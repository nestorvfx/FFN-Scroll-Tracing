"""Block-wise decode with mid-halo mutual-maximal-overlap stitching (DESIGN item 19, research/09 S2).

WHY THIS FILE EXISTS. `inference.block_grid` / `inference.stitch_blocks` are **dead code called by
nothing** (FINDINGS 3.4): `evaluate.py` and `inline_eval.py` both decode whole volumes, so there is
no whole-scroll decode at all -- `flood_fill_object` allocates a [Z,Y,X] float POM canvas and a
[Z,Y,X] bool `wrote` mask PER FILL, which at 118 Gvox is not a tuning problem. And the stitch rule
that does exist is a **one-sided greedy** "≥ 50% overlap -> take the majority global label",
whereas the rule measured at **-84% mergers / +28% splits** (Januszewski et al. 2018, songbird) is:

    discard the outer envelope of each subvolume, take a 1-voxel plane from the MIDDLE of each
    overlap, and link two segments only if they are MUTUALLY MAXIMAL on that plane -- A's best
    partner is B *and* B's best partner is A -- then reconcile globally with union-find.

Mutual maximality is the whole point. One-sided greedy links A->B whenever B is A's best partner,
so a segment that leaks across a blind contact in ONE block drags its neighbour's identity across
every block it touches. Requiring agreement in both directions makes a single bad block local.

GEOMETRY. Each block owns a `core` region and decodes on `core + halo`. The halo IS the discarded
envelope: fills there are truncated by the block boundary and are used only for stitching, never
for output. For two blocks adjacent along an axis, their extended regions overlap on
[c-halo, c+halo) where c is the shared core boundary -- so the middle of the overlap is exactly the
plane a = c, where the lower block has halo labels and the upper block has core labels. Both are
present, which is what makes the comparison possible.

Halo must satisfy halo >= fov//2 + delta so a fill starting anywhere in the core can take at least
one step and still see a full FOV (`flood_fill_object` refuses centres closer than fov//2 to a face).
"""
from __future__ import annotations

import numpy as np
import torch

from .inference import segment_block


def block_grid(shape, block, halo):
    """Tile `shape` into core blocks with halos.

    Returns a list of dicts: core_lo/core_hi (the region this block OWNS) and ext_lo/ext_hi (the
    region it DECODES). Cores tile the volume exactly and never overlap; extended regions do.
    """
    Z, Y, X = shape
    bz, by, bx = block
    out = []
    for iz, z0 in enumerate(range(0, Z, bz)):
        for iy, y0 in enumerate(range(0, Y, by)):
            for ix, x0 in enumerate(range(0, X, bx)):
                core_lo = (z0, y0, x0)
                core_hi = (min(z0 + bz, Z), min(y0 + by, Y), min(x0 + bx, X))
                ext_lo = tuple(max(0, c - halo) for c in core_lo)
                ext_hi = (min(Z, core_hi[0] + halo), min(Y, core_hi[1] + halo),
                          min(X, core_hi[2] + halo))
                out.append(dict(idx=(iz, iy, ix), core_lo=core_lo, core_hi=core_hi,
                                ext_lo=ext_lo, ext_hi=ext_hi))
    return out


def _sl(lo, hi):
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def mutual_max_pairs(a_plane: np.ndarray, b_plane: np.ndarray, min_overlap: int = 10):
    """Segment pairs that are MUTUALLY maximal on a shared plane.

    a_plane/b_plane are two labelings of the SAME voxels. Returns [(a, b), ...] where b is a's
    largest-overlap partner AND a is b's. This is the -84%-merger rule; a one-sided argmax is what
    the current `stitch_blocks` does and is what lets one bad block propagate.
    """
    m = (a_plane > 0) & (b_plane > 0)
    if not m.any():
        return []
    aa = a_plane[m].astype(np.int64)
    bb = b_plane[m].astype(np.int64)
    nb = int(bb.max()) + 1
    uk, cnt = np.unique(aa * nb + bb, return_counts=True)
    ai, bi = uk // nb, uk % nb
    best_a, best_b = {}, {}
    for a_, b_, c in zip(ai.tolist(), bi.tolist(), cnt.tolist()):
        if c < min_overlap:
            continue
        if a_ not in best_a or c > best_a[a_][0]:
            best_a[a_] = (c, b_)
        if b_ not in best_b or c > best_b[b_][0]:
            best_b[b_] = (c, a_)
    return [(a_, b_) for a_, (c, b_) in best_a.items()
            if b_ in best_b and best_b[b_][1] == a_]


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def decode_blocked(predict_fns, image_gpu, seeds, cfg, block=(64, 384, 384), halo=40,
                   K=128, min_overlap=10, fibre_mask=None, progress=False,
                   devices=None, threads=True):
    """Decode a volume block-wise and reconcile with mid-halo mutual-max stitching.

    predict_fns: one callable, or a list (round-robin over devices -- DESIGN 6.4).
    devices: optional list parallel to `predict_fns` giving each one's device. REQUIRED when the
        models live on different GPUs: each block's sub-image must be moved to the device of the
        model that will consume it, or conv3d raises "weight is on cuda:1, other tensors on cuda:0".
    threads: run blocks concurrently, one worker per predict_fn. Blocks are independent
        ("embarrassingly parallel with no dependencies between subvolumes" -- google/ffn manual),
        so this is real parallelism, not just round-robin.
    seeds: [M,3] global seed coordinates; each block decodes the seeds inside its CORE.
    fibre_mask: optional bool [Z,Y,X]; when given, non-fibre voxels are marked claimed so the
        fill cannot move through them (the FAFB tissue-mask restrictor -- free, several-fold
        cheaper, and it cannot create a merge because it only ever BLOCKS growth).

    Returns (labels [Z,Y,X] int32, stats dict).
    """
    if not isinstance(predict_fns, (list, tuple)):
        predict_fns = [predict_fns]
    if devices is None:
        devices = [image_gpu.device] * len(predict_fns)
    devices = [torch.device(d) for d in devices]
    Z, Y, X = tuple(int(s) for s in image_gpu.shape)
    blocks = block_grid((Z, Y, X), block, halo)
    seeds = np.asarray(seeds, np.int64).reshape(-1, 3)
    def _one(bi):
        b = blocks[bi]
        esl = _sl(b["ext_lo"], b["ext_hi"])
        w = bi % len(predict_fns)
        shape = tuple(int(h - l) for l, h in zip(b["ext_lo"], b["ext_hi"]))
        lo = np.array(b["ext_lo"])
        in_core = np.all((seeds >= np.array(b["core_lo"])) & (seeds < np.array(b["core_hi"])), 1)
        s_local = seeds[in_core] - lo
        if len(s_local) == 0:
            return bi, np.zeros(shape, np.int32)
        # move the sub-image to the device of the model that will consume it
        sub_img = image_gpu[esl].contiguous().to(devices[w])
        with torch.no_grad():
            lab = segment_block(predict_fns[w], sub_img, s_local, cfg, K=K)
        if fibre_mask is not None:
            lab[~fibre_mask[esl]] = 0
        if progress:
            print(f"  block {bi+1}/{len(blocks)} {b['core_lo']} "
                  f"objects={len(np.unique(lab))-1}", flush=True)
        return bi, lab.astype(np.int32)

    ext_labels = {}
    if threads and len(predict_fns) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(predict_fns)) as ex:
            for bi, lab in ex.map(_one, range(len(blocks))):
                ext_labels[bi] = lab
    else:
        for bi in range(len(blocks)):
            k, lab = _one(bi)
            ext_labels[k] = lab
    offset = 0
    offsets = {}
    for bi in range(len(blocks)):
        offsets[bi] = offset
        offset += int(ext_labels[bi].max()) + 1
    # ---- link across every shared core-boundary plane (the middle of each overlap) ----
    uf = _UF()
    n_links = 0
    by_idx = {b["idx"]: bi for bi, b in enumerate(blocks)}
    for bi, b in enumerate(blocks):
        for ax in range(3):
            nb_idx = list(b["idx"])
            nb_idx[ax] += 1
            nb_idx = tuple(nb_idx)
            if nb_idx not in by_idx:
                continue
            bj = by_idx[nb_idx]
            c = b["core_hi"][ax]                     # shared core boundary == middle of overlap
            other = blocks[bj]
            if c <= max(b["ext_lo"][ax], other["ext_lo"][ax]) or \
               c >= min(b["ext_hi"][ax], other["ext_hi"][ax]):
                continue
            # the plane's extent = intersection of the two extended regions on the other two axes
            lo = [max(b["ext_lo"][a], other["ext_lo"][a]) for a in range(3)]
            hi = [min(b["ext_hi"][a], other["ext_hi"][a]) for a in range(3)]
            lo[ax], hi[ax] = c, c + 1
            if any(h <= l for l, h in zip(lo, hi)):
                continue
            pa = ext_labels[bi][_sl([lo[a] - b["ext_lo"][a] for a in range(3)],
                                    [hi[a] - b["ext_lo"][a] for a in range(3)])]
            pb = ext_labels[bj][_sl([lo[a] - other["ext_lo"][a] for a in range(3)],
                                    [hi[a] - other["ext_lo"][a] for a in range(3)])]
            for (la, lb) in mutual_max_pairs(pa, pb, min_overlap):
                uf.union(offsets[bi] + la, offsets[bj] + lb)
                n_links += 1
    # ---- write CORE regions only, relabelled through union-find ----
    out = np.zeros((Z, Y, X), np.int32)
    remap, nxt = {}, 1
    for bi, b in enumerate(blocks):
        lab = ext_labels[bi]
        csl_g = _sl(b["core_lo"], b["core_hi"])
        csl_l = _sl([b["core_lo"][a] - b["ext_lo"][a] for a in range(3)],
                    [b["core_hi"][a] - b["ext_lo"][a] for a in range(3)])
        core = lab[csl_l]
        ids = [int(i) for i in np.unique(core) if i]
        if not ids:
            continue
        lut = np.zeros(int(core.max()) + 1, np.int32)
        for i in ids:
            r = uf.find(offsets[bi] + i)
            if r not in remap:
                remap[r] = nxt
                nxt += 1
            lut[i] = remap[r]
        out[csl_g] = lut[core]
    return out, dict(n_blocks=len(blocks), n_links=n_links, n_objects=nxt - 1,
                     block=tuple(block), halo=int(halo))
