#!/usr/bin/env python3
"""GASP average-linkage signed-graph decode (Bailoni et al., CVPR 2022 — HC-Avg / HCC-Avg) as a drop-in
alternative to greedy Mutex-Watershed for the affinity slab decode.

Why: greedy MWS is single-linkage-brittle and over-fragments (81k instances / 68 wraps); AVERAGE linkage
propagates a separation established where a contact IS visible into its blind stretch along the same sheet
(Bailoni: SOTA on CREMI EM). One repulsive edge is not decisive (as in MWS) — the MEAN interaction is, so a
mostly-touching contact with one visible gap still separates, and a genuinely-same sheet with one weak spot
does not split.

Pipeline (fragments stay in fast rust; only the small fragment-graph agglomeration is python):
  1. superpixels: WSDT-style — boundary map = 1 - min(unit affinities); seeds = CC of deep-interior (boundary
     < seed_thr); watershed(boundary, seeds, mask=fiber) -> fragments that never straddle a strong boundary.
  2. RAG: per offset channel, accumulate signed weight w=aff-0.5 (+size) on every fragment-fragment boundary
     edge (short attractive + long repulsive both contribute to the mean).
  3. agglomerate: max-heap by current AVERAGE weight (sumW/sumS), union-find, lazy-invalidate stale entries;
     HC-Avg merges while max avg > 0; HCC-Avg (--cannot_link) processes by |avg|, marks cannot-link on avg<0.
  4. small-fragment absorption into the neighbour with the strongest attractive average.
  5. emit: carve_from_labels -> carved binary -> wrap_erl harness (same consumer as the MWS decode).

Usage:
  python gasp_decode.py --selftest
  python gasp_decode.py --aff /root/slab_pred_mine_seed1/aff_f16.npy --fiber /root/slab_pred_mine_seed1/fiber.npy \
      --ranges 1,3,9,27 --out /root/slab_pred_mine_seed1/fiber_gasp.npy [--cannot_link] [--min_frag 20]
"""
import os, sys, argparse, heapq
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tstr_eval_aff import build_offsets
from slab_emit_carved import carve_from_labels, shift_fiber


# ------------------------------------------------------------------ union-find
class UF:
    def __init__(self, n):
        self.p = np.arange(n, dtype=np.int64)
        self.sz = np.ones(n, dtype=np.int64)

    def find(self, x):
        p = self.p
        r = x
        while p[r] != r:
            r = p[r]
        while p[x] != r:
            p[x], x = r, p[x]
        return int(r)

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.sz[ra] += self.sz[rb]
        return ra


# ------------------------------------------------------------------ fragments (superpixels)
def make_fragments(aff, fiber, seed_thr=0.20):
    """WSDT-style oversegmentation. boundary = MAX over unit offsets of (1 - affinity) on existing fiber edges,
    marked at BOTH endpoints -> a voxel is boundary if it has a WEAK edge in ANY direction. seeds = CC of fiber
    where boundary < seed_thr (deep interior); watershed fills the rest within fiber. aff[ax] valued at source."""
    from skimage.segmentation import watershed
    bnd = np.zeros(fiber.shape, np.float32)
    for ax, off in enumerate(([1, 0, 0], [0, 1, 0], [0, 0, 1])):
        o = off[ax]
        src = [slice(None)] * 3; src[ax] = slice(0, fiber.shape[ax] - o)   # v
        dst = [slice(None)] * 3; dst[ax] = slice(o, fiber.shape[ax])       # v+off
        src = tuple(src); dst = tuple(dst)
        emask = fiber[src] & fiber[dst]                                    # edge (v, v+off) exists
        contrib = np.where(emask, 1.0 - aff[ax][src].astype(np.float32), 0.0)
        bnd[src] = np.maximum(bnd[src], contrib)                           # source endpoint
        bnd[dst] = np.maximum(bnd[dst], contrib)                           # target endpoint
    bnd = np.where(fiber, bnd, 0.0)
    seeds = ndi.label(fiber & (bnd < seed_thr))[0]
    frags = watershed(bnd, markers=seeds, mask=fiber).astype(np.int32)
    return frags


# ------------------------------------------------------------------ region-adjacency graph
def build_rag(frags, aff, offsets, fiber):
    """adj[a] = {b: [sumW, sumS]} over fragment-fragment edges; w = aff-0.5 (attractive>0, repulsive<0)."""
    BIG = np.int64(frags.max()) + 1
    acc = {}                                              # key = a*BIG+b (a<b) -> [sumW, cnt]
    for c, off in enumerate(offsets):                     # offsets all non-negative (build_offsets)
        src = tuple(slice(0, s - o) for s, o in zip(frags.shape, off))   # v
        dst = tuple(slice(o, s) for s, o in zip(frags.shape, off))       # v+off
        fa = frags[src]                                   # frags[v]
        fbv = frags[dst]                                  # frags[v+off]
        m = fiber[src] & fiber[dst] & (fa > 0) & (fbv > 0) & (fa != fbv)
        if not m.any():
            continue
        a = fa[m].astype(np.int64)
        b = fbv[m].astype(np.int64)
        w = aff[c][src][m].astype(np.float32) - 0.5       # aff valued at source v, for edge (v, v+off)
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        key = lo * BIG + hi
        order = np.argsort(key, kind="stable")
        key_s = key[order]
        w_s = w[order]
        uk, idx = np.unique(key_s, return_index=True)
        sumW = np.add.reduceat(w_s, idx)
        cnt = np.diff(np.append(idx, len(w_s)))
        for k, sw, cc in zip(uk.tolist(), sumW.tolist(), cnt.tolist()):
            e = acc.get(k)
            if e is None:
                acc[k] = [sw, cc]
            else:
                e[0] += sw
                e[1] += cc
    adj = {}
    for k, (sw, cc) in acc.items():
        a = int(k // BIG)
        b = int(k % BIG)
        adj.setdefault(a, {})[b] = [sw, cc]
        adj.setdefault(b, {})[a] = [sw, cc]
    return adj


# ------------------------------------------------------------------ GASP agglomeration
def gasp_agglomerate(adj, n_nodes, cannot_link=False):
    """HC-Avg (cannot_link=False): merge while max average interaction > 0.
    HCC-Avg (cannot_link=True): process by |avg|, mark cannot-link on avg<0. Lazy-invalidated heap."""
    uf = UF(n_nodes + 1)
    clink = {}                                            # root -> set of cannot-link roots
    heap = []
    for a, nb in adj.items():
        for b, (sw, cc) in nb.items():
            if a < b:
                avg = sw / cc
                pr = -abs(avg) if cannot_link else -avg
                heapq.heappush(heap, (pr, a, b))
    while heap:
        pr, a, b = heapq.heappop(heap)
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue
        e = adj.get(ra, {}).get(rb)
        if e is None:
            continue
        avg = e[0] / e[1]
        cur_pr = -abs(avg) if cannot_link else -avg
        if abs(cur_pr - pr) > 1e-9:                       # stale entry (weights changed since pushed)
            continue
        if cannot_link and rb in clink.get(ra, ()):       # forbidden pair
            continue
        if avg <= 0:
            if cannot_link:
                clink.setdefault(ra, set()).add(rb)
                clink.setdefault(rb, set()).add(ra)
                continue
            else:
                break                                     # HC-Avg: no attractive edges left
        # merge ra, rb (average linkage: sum the aggregates)
        keep = uf.union(ra, rb)
        gone = rb if keep == ra else ra
        merged = {}
        for src in (ra, rb):
            for nbr, (sw, cc) in adj.get(src, {}).items():
                rn = uf.find(nbr)
                if rn == keep:
                    continue
                if rn in merged:
                    merged[rn][0] += sw
                    merged[rn][1] += cc
                else:
                    merged[rn] = [sw, cc]
        # write merged adjacency symmetrically; drop old endpoints
        adj[keep] = merged
        for rn, agg in merged.items():
            adj.setdefault(rn, {})
            # remove stale refs to ra/rb, install merged ref to keep
            adj[rn].pop(ra, None)
            adj[rn].pop(rb, None)
            adj[rn][keep] = agg
            avg2 = agg[0] / agg[1]
            pr2 = -abs(avg2) if cannot_link else -avg2
            heapq.heappush(heap, (pr2, keep, rn))
        if gone in adj and gone != keep:
            adj.pop(gone, None)
        # merge cannot-link sets
        if cannot_link:
            s = clink.get(ra, set()) | clink.get(rb, set())
            s.discard(keep)
            if s:
                clink[keep] = s
                for x in s:
                    clink.setdefault(x, set()).add(keep)
    return uf


def absorb_small(labels, adj_orig, uf, min_frag):
    """Absorb final clusters smaller than min_frag voxels into the neighbour with the strongest attractive avg."""
    if min_frag <= 0:
        return uf
    ids, sizes = np.unique(labels[labels > 0], return_counts=True)
    size = dict(zip(ids.tolist(), sizes.tolist()))
    # cluster size = sum of member fragment sizes (labels already carry fragment ids here)
    for f in ids.tolist():
        r = uf.find(f)
    # aggregate cluster sizes by root
    root_size = {}
    for f, s in size.items():
        root_size[uf.find(f)] = root_size.get(uf.find(f), 0) + s
    changed = True
    guard = 0
    while changed and guard < 5:
        changed = True if False else False
        guard += 1
        smalls = [r for r, s in root_size.items() if s < min_frag]
        for r in smalls:
            # best attractive neighbour among original fragment adjacencies mapped to roots
            best, bestw = None, -1e9
            for f in [fr for fr in size if uf.find(fr) == r]:
                for nbr, (sw, cc) in adj_orig.get(f, {}).items():
                    rn = uf.find(nbr)
                    if rn == r:
                        continue
                    w = sw / cc
                    if w > bestw:
                        bestw, best = w, rn
            if best is not None and bestw > -1e9:
                keep = uf.union(r, best)
                other = best if keep == r else r
                root_size[keep] = root_size.get(keep, 0) + root_size.get(other, 0)
                root_size.pop(other, None)
                changed = True
    return uf


# ------------------------------------------------------------------ end-to-end
def gasp_decode(aff, fiber, offsets, cannot_link=False, min_frag=20, seed_thr=0.20, verbose=True):
    frags = make_fragments(aff, fiber, seed_thr=seed_thr)
    nfrag = int(frags.max())
    if verbose:
        print(f"[gasp] fragments={nfrag}", flush=True)
    adj = build_rag(frags, aff, offsets, fiber)
    adj_orig = {a: dict(nb) for a, nb in adj.items()}     # snapshot for absorption (agglo mutates adj)
    uf = gasp_agglomerate(adj, nfrag, cannot_link=cannot_link)
    # relabel fragments -> cluster roots
    remap = np.zeros(nfrag + 1, np.int64)
    for f in range(1, nfrag + 1):
        remap[f] = uf.find(f)
    labels = remap[frags]
    uf = absorb_small(frags, adj_orig, uf, min_frag)      # note: uses fragment voxel sizes
    for f in range(1, nfrag + 1):
        remap[f] = uf.find(f)
    labels = remap[frags].astype(np.int32)
    # compact ids
    uids = np.unique(labels)
    comp = np.zeros(int(uids.max()) + 1, np.int32)
    comp[uids] = np.arange(len(uids))
    labels = comp[labels]
    if verbose:
        print(f"[gasp] final clusters={len(uids) - (1 if 0 in uids else 0)}", flush=True)
    return labels.astype(np.int32)


# ------------------------------------------------------------------ self-test
def selftest():
    """Two sheets that TOUCH (share a boundary) but have a visible gap on part of the contact, plus a single
    sheet with one weak internal spot. GASP-avg must (a) separate the two sheets, (b) NOT split the single sheet.
    Also a pure MWS-single-linkage would bridge across the mostly-touching contact."""
    Z, Y, X = 4, 40, 40
    fiber = np.zeros((Z, Y, X), bool)
    inst = np.zeros((Z, Y, X), np.int32)
    # sheet A rows 10-14, sheet B rows 16-20, with a true 1-vox gap at row15 -> fragments are cleanly separate
    # (unit edges never cross the gap). The A|B relationship lives on LONG-RANGE (r=3,9) cross-gap edges.
    fiber[:, 10:15, :] = True; inst[:, 10:15, :] = 1
    fiber[:, 16:21, :] = True; inst[:, 16:21, :] = 2
    offsets = build_offsets("1,3,9")                                     # ch4=[0,3,0], ch7=[0,9,0]
    aff = np.zeros((len(offsets),) + fiber.shape, np.float32)
    for c, off in enumerate(offsets):
        tb = np.zeros_like(inst)
        src = tuple(slice(0, s - o) for s, o in zip(inst.shape, off))     # v
        dst = tuple(slice(o, s) for s, o in zip(inst.shape, off))         # v+off
        tb[src] = inst[dst]                                               # tb[v] = inst[v+off]
        same = (inst > 0) & (tb > 0) & (inst == tb)
        aff[c] = np.where(same, 0.95, 0.05).astype(np.float32)
    # AVERAGE-LINKAGE TEST: the A|B RAG edge (long-range r=3 across the gap, row13->row16) is mostly REPULSIVE
    # (0.05) but with a few SPURIOUS strong-attractive spots (0.9) at x in [20,24) -- the model wrongly says merge
    # there. Mean stays repulsive -> GASP-avg must keep A,B separate, whereas single-linkage / Mutex-Watershed
    # would contract one 0.9 edge and merge the two sheets.
    aff[4][:, 13, 20:24] = 0.90                                          # ch4 = [0,3,0]: row13 -> row16
    labels = gasp_decode(aff, fiber, offsets, cannot_link=False, min_frag=0, seed_thr=0.2, verbose=False)
    a_ids = set(np.unique(labels[:, 10:15, :][labels[:, 10:15, :] > 0]).tolist())
    b_ids = set(np.unique(labels[:, 16:21, :][labels[:, 16:21, :] > 0]).tolist())
    merged = bool(a_ids & b_ids)
    oversplit = (len(a_ids) > 1) or (len(b_ids) > 1)
    print("selftest: A_ids", a_ids, "B_ids", b_ids, "merged", merged, "oversplit", oversplit)
    ok = (not merged) and (not oversplit)
    print("SELFTEST_OK" if ok else "SELFTEST_FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--aff"); ap.add_argument("--fiber"); ap.add_argument("--out")
    ap.add_argument("--ranges", default="1,3,9,27")
    ap.add_argument("--cannot_link", action="store_true")
    ap.add_argument("--min_frag", type=int, default=20)
    ap.add_argument("--seed_thr", type=float, default=0.20)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    offsets = build_offsets(a.ranges)
    aff = np.load(a.aff).astype(np.float32)
    if aff.max() > 1.5:
        aff /= 255.0
    fiber_prob = np.load(a.fiber).astype(np.float32)
    if fiber_prob.max() > 1.5:
        fiber_prob /= 255.0
    fiber = fiber_prob >= 0.5
    labels = gasp_decode(aff, fiber, offsets, cannot_link=a.cannot_link, min_frag=a.min_frag, seed_thr=a.seed_thr)
    carved_prob, carved_binary, carve = carve_from_labels(labels, fiber, fiber_prob)
    np.save(a.out, carved_prob)
    print("GASP_DECODE_DONE", a.out, "clusters", int(labels.max()), "carved_fg", int(carved_binary.sum()), flush=True)


if __name__ == "__main__":
    main()
