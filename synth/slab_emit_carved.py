#!/usr/bin/env python3
"""Carved-emission decode for the affinity arm on the GT slab (the production artifact).

Production consumes predictions as nonzero-binarize -> thin -> ridge polylines, so EXPLICIT ZEROS at wrap-wrap
seams are the only separation mechanism the tracer can see. This script turns the dual-head model (fiber prob +
12 affinity channels) into carved fields two independent ways:

  affcarve : threshold the SHORT-RANGE affinities directly -- zero any fiber voxel on a fiber-fiber edge whose
             r=1 affinity says "different object" (p < aff_carve_thr). No instances involved. This is the pure
             readout of the learned representation: if the head knows where invisible seams are, this shows it;
             if this shows nothing, no clustering can conjure it.
  mws      : Mutex-Watershed instances (short-range attractive, long-range REPULSIVE-ONLY, deterministic grid
             strides) -> absorb fragments < min_inst voxels into the neighbor with the highest mean short-range
             affinity (the model's own merge evidence, NOT contact area) -> zero fiber voxels 26-adjacent to a
             different surviving instance.

Why v1 failed (2026-07-18, dip screen 0.016 < 0.046 intensity floor): MWS over-fragmented (81,475 instances /
68 wraps -- long-range 2p-1 turns every uncertain within-wrap affinity into a mutex edge; the affinity head is
supervised only on synthetic cases so real-domain long-range affs are noisy) and the carve then zeroed
intra-wrap fragment boundaries, i.e. wrap INTERIORS. Fixes: repulsive-only long range, affinity-aware
absorption, plus the instance-free affcarve baseline.

The network pass (~1h at slab scale) caches fiber.npy + aff_f16.npy in the output dir; decode-parameter
iteration reuses them for free.

Outputs to /root/slab_pred_<tag>/: fiber.npy (raw prob), fiber_carved.npy + carved_binary.npy (mws decode),
fiber_affcarve.npy + affcarve_binary.npy (direct decode), labels_raw.npy/labels_absorbed.npy. Score each field
with wrap_erl.py to isolate each mechanism's contribution.

Usage:
  python slab_emit_carved.py --model_dir <...AffLongG__...3d_fullres> --ckpt checkpoint_final.pth \
         --ranges 1,3,9,27 --tag afflong [--gpu 0] [--mode both|mws|affcarve] [--min_inst 100000] \
         [--aff_carve_thr 0.3] [--stride_scale 3]
  python slab_emit_carved.py --selftest        # phantom-based plumbing test, no GPU/model/slab needed
"""
import os, sys, argparse
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tstr_eval_aff import build_offsets, build_aff_network, preprocess, sliding_window_logits, \
    mws_instances, shift_fiber


def pair_boundary_stats(labels, aff):
    """Boundary statistics between different positive labels over the 3 unit offsets:
    {(lo,hi): [n_edges, sum_shortrange_aff]}. aff channels 0..2 must be the r=1 (z,y,x) affinities,
    valued at the SOURCE voxel (the affinity_gt convention)."""
    lab = labels.astype(np.int64)
    n = int(lab.max()) + 1
    stats = {}
    for ax in range(3):
        s_src = [slice(None)] * 3; s_dst = [slice(None)] * 3
        s_src[ax] = slice(None, -1); s_dst[ax] = slice(1, None)
        a = lab[tuple(s_src)].ravel(); b = lab[tuple(s_dst)].ravel()
        av = aff[ax][tuple(s_src)].ravel()
        m = (a > 0) & (b > 0) & (a != b)
        if not m.any():
            continue
        key = np.minimum(a[m], b[m]) * n + np.maximum(a[m], b[m])
        order = np.argsort(key, kind='stable')
        k_sorted = key[order]; av_sorted = av[m][order].astype(np.float64)
        uniq, start = np.unique(k_sorted, return_index=True)
        cnts = np.diff(np.append(start, k_sorted.size))
        sums = np.add.reduceat(av_sorted, start)
        for ki, ci, si in zip(uniq.tolist(), cnts.tolist(), sums.tolist()):
            e = stats.setdefault(divmod(ki, n), [0, 0.0])
            e[0] += ci; e[1] += si
    return stats


def repair_fragments(labels, aff, merge_thr=0.6, min_size=0):
    """Two-stage region-graph repair of MWS over-fragmentation, driven by the model's own merge evidence.

    Stage 1 (agglomerate, any size): repeatedly merge the adjacent pair with the highest mean short-range
    boundary affinity while that mean >= merge_thr -- a false intra-wrap boundary has high mean r=1 affinity,
    a real (learned) seam has low. Classic waterz-style mean agglomeration.
    Stage 2 (absorb): merge leftover fragments < min_size into the neighbor with the highest mean affinity
    (contact area as tiebreak), so no tiny fragment survives to ring itself with carved zeros."""
    lab = labels.astype(np.int64)
    n = int(lab.max()) + 1
    sizes = np.bincount(lab.ravel(), minlength=n)
    stats = pair_boundary_stats(lab, aff)
    adj = {}                                              # node -> {node: [n_edges, aff_sum]}
    for (i, j), (c, s) in stats.items():
        e = [c, s]                                        # ONE shared list per pair: both directions stay in sync
        adj.setdefault(i, {})[j] = e
        adj.setdefault(j, {})[i] = e
    parent = np.arange(n)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    import heapq

    def merge(r, t):
        """Union r into t; fold r's adjacency into t's. Returns list of t's neighbors whose stats changed."""
        parent[r] = t
        sizes[t] += sizes[r]; sizes[r] = 0
        touched = []
        merged = adj.pop(r, {})
        for nb, (c, s) in merged.items():
            d = adj.get(nb)
            if d is not None:
                d.pop(r, None)                            # drop the reverse reference to the dead node
            nbr = find(nb)
            if nbr == t:
                continue
            et = adj.setdefault(t, {}).get(nbr)
            if et is None:
                et = [0, 0.0]
                adj[t][nbr] = et
                adj.setdefault(nbr, {})[t] = et           # shared list keeps both directions in sync
            et[0] += c; et[1] += s
            touched.append(nbr)
        adj.get(t, {}).pop(r, None)
        return touched

    # stage 1: global mean-affinity agglomeration (lazy heap: entries carry the mean they were pushed with;
    # stale entries -- merged nodes or changed stats -- are re-validated on pop)
    heap = [(-(s / c), i, j) for (i, j), (c, s) in stats.items() if s / c >= merge_thr]
    heapq.heapify(heap)
    while heap:
        negm, i, j = heapq.heappop(heap)
        r, t = find(i), find(j)
        if r == t:
            continue
        e = adj.get(r, {}).get(t)
        if e is None:
            continue
        mean = e[1] / e[0]
        if mean < merge_thr:
            continue
        if abs(-negm - mean) > 1e-9:                      # stats changed since push: re-queue with fresh mean
            heapq.heappush(heap, (-mean, r, t)); continue
        keep, gone = (r, t) if sizes[r] >= sizes[t] else (t, r)
        for nbr in merge(gone, keep):
            e2 = adj[keep][nbr]
            m2 = e2[1] / e2[0]
            if m2 >= merge_thr:
                heapq.heappush(heap, (-m2, keep, nbr))

    # stage 2: size-based absorption of what's left
    if min_size > 0:
        sheap = [(int(sizes[find(i)]), find(i)) for i in range(1, n) if 0 < sizes[find(i)] < min_size]
        heapq.heapify(sheap)
        while sheap:
            sz, i = heapq.heappop(sheap)
            r = find(i)
            if r != i or sizes[r] != sz or sizes[r] >= min_size or not adj.get(r):
                continue                                  # stale entry, or grown big / isolated
            tgt = max(adj[r], key=lambda nb: (adj[r][nb][1] / adj[r][nb][0], adj[r][nb][0]))
            t = find(tgt)
            if t == r:                                    # stale neighbor id already merged into us
                del adj[r][tgt]
                heapq.heappush(sheap, (int(sizes[r]), r)); continue
            merge(r, t)
            if 0 < sizes[t] < min_size:
                heapq.heappush(sheap, (int(sizes[t]), t))
    root_of = np.array([find(i) for i in range(n)], dtype=np.int64)
    return root_of[lab].astype(np.int32)


def carve_from_labels(labels, fiber, fiber_prob):
    """Zero fiber voxels 26-adjacent (3^3 box) to a DIFFERENT positive instance -> ~2vox-wide explicit seam."""
    big = np.where(labels > 0, labels, np.iinfo(np.int32).max)
    mn = ndi.minimum_filter(big, size=3)
    mx = ndi.maximum_filter(np.where(labels > 0, labels, 0), size=3)
    carve = (labels > 0) & (mn < np.iinfo(np.int32).max) & (mx != mn)
    carved_binary = fiber & ~carve
    return np.where(carved_binary, fiber_prob, 0.0).astype(np.float32), carved_binary, carve


def aff_carve(aff, fiber, fiber_prob, thr):
    """Instance-free decode: zero BOTH endpoints of any fiber-fiber unit edge whose r=1 affinity < thr.
    aff channels 0..2 = unit (z,y,x) offsets, valued at the source voxel."""
    carve = np.zeros_like(fiber)
    for ax, off in enumerate(([1, 0, 0], [0, 1, 0], [0, 0, 1])):
        tgt_fiber = shift_fiber(fiber, off)
        low = fiber & tgt_fiber & (aff[ax] < thr)         # low-affinity fiber-fiber edge, marked at source
        carve |= low
        carve |= shift_fiber(low, [-c for c in off])      # and its target endpoint
    carve &= fiber
    carved_binary = fiber & ~carve
    return np.where(carved_binary, fiber_prob, 0.0).astype(np.float32), carved_binary, carve


def aff_carve_rank(aff, fiber, fiber_prob, budget_frac, min_keep=0.5):
    """"Cut on doubt", RANK-based: sever the lowest `budget_frac` of fiber->fiber unit edges by affinity.

    This replaces the absolute threshold as the default decode, for two reasons.

    1. It is IMMUNE to a global calibration shift. The ep20 head emitted a max affinity of 0.30, so an absolute
       thr=0.3 carve severed essentially every edge and `AFFCARVE thr=0.3: fg 0.1818 -> 0.0000` erased the entire
       fiber -- a total decode failure produced by a monotone shift that leaves the RANKING untouched. Any
       rank-based rule (this, or Mutex Watershed) cannot fail that way, because it depends only on edge order.
    2. It expresses the actual cost structure. Splits under ~40-60px are interpolated away downstream and are
       effectively free, while one merge shifts the winding number of every outer wrap. So the operating point we
       want is not "where is P(same) = 0.5" but "how many splits am I willing to buy" -- and at a blind contact a
       CALIBRATED affinity falls to its class prior, which lands it in the cut fraction and severs it. That is the
       whole point of training with a proper scoring rule: doubt becomes actionable instead of being a coin flip.

    `min_keep` is a guard, not a tuning knob: if a carve would destroy more than half the fiber, something is
    structurally wrong (as above) and we raise rather than silently emit an empty volume for the tracer."""
    vals, masks = [], []
    for ax, off in enumerate(([1, 0, 0], [0, 1, 0], [0, 0, 1])):
        m = fiber & shift_fiber(fiber, off)               # valid fiber->fiber unit edges, marked at source
        vals.append(aff[ax][m]); masks.append(m)
    allv = np.concatenate(vals) if vals else np.array([0.0])
    thr = float(np.quantile(allv, budget_frac)) if allv.size else 0.0
    carve = np.zeros_like(fiber)
    for ax, off in enumerate(([1, 0, 0], [0, 1, 0], [0, 0, 1])):
        low = masks[ax] & (aff[ax] < thr)
        carve |= low
        carve |= shift_fiber(low, [-c for c in off])      # and the target endpoint
    carve &= fiber
    carved_binary = fiber & ~carve
    kept = float(carved_binary.sum()) / max(1.0, float(fiber.sum()))
    if kept < min_keep:
        raise RuntimeError(f"rank carve at budget {budget_frac} kept only {kept:.1%} of the fiber (thr={thr:.4f}). "
                           f"That is a decode failure, not a segmentation: refusing to emit. "
                           f"Check the affinity distribution with synth/diag_affinity.py.")
    return np.where(carved_binary, fiber_prob, 0.0).astype(np.float32), carved_binary, carve, thr


def run_decodes(fiber_prob, aff, offsets, out_dir, a):
    fiber = fiber_prob >= a.fiber_thr
    print("fiber fg frac", float(fiber.mean()), flush=True)
    if a.mode in ("both", "affcarve"):
        budget = getattr(a, 'carve_budget', 0.0)
        if budget > 0:
            fc, cb, carve, thr = aff_carve_rank(aff, fiber, fiber_prob, budget)
            print(f"AFFCARVE-RANK budget={budget:.3f} -> thr={thr:.4f}: carved {int(carve.sum())} "
                  f"({float(carve.mean()):.4f} of vol); fg {float(fiber.mean()):.4f} -> {float(cb.mean()):.4f}",
                  flush=True)
        else:
            fc, cb, carve = aff_carve(aff, fiber, fiber_prob, a.aff_carve_thr)
            print(f"AFFCARVE thr={a.aff_carve_thr}: carved {int(carve.sum())} ({float(carve.mean()):.4f} of vol); "
                  f"fg {float(fiber.mean()):.4f} -> {float(cb.mean()):.4f}", flush=True)
        np.save(f"{out_dir}/fiber_affcarve.npy", fc)
        np.save(f"{out_dir}/affcarve_binary.npy", cb.astype(np.uint8))
    if a.mode in ("both", "mws"):
        labels = mws_instances(aff.astype(np.float32), fiber, offsets,
                               stride_scale=a.stride_scale, long_repulsive_only=True)
        np.save(f"{out_dir}/labels_raw.npy", labels.astype(np.int32))
        print("MWS instances:", int(len(np.unique(labels[labels > 0]))), flush=True)
        if a.min_inst > 0 or a.merge_thr < 1.0:
            labels = repair_fragments(labels, aff, merge_thr=a.merge_thr, min_size=a.min_inst)
            np.save(f"{out_dir}/labels_absorbed.npy", labels)
            print(f"after repair(merge_thr={a.merge_thr}, min_inst={a.min_inst}):",
                  int(len(np.unique(labels[labels > 0]))), flush=True)
        fc, cb, carve = carve_from_labels(labels, fiber, fiber_prob)
        np.save(f"{out_dir}/fiber_carved.npy", fc)
        np.save(f"{out_dir}/carved_binary.npy", cb.astype(np.uint8))
        print(f"MWS-CARVE: carved {int(carve.sum())} ({float(carve.mean()):.4f} of vol); "
              f"fg {float(fiber.mean()):.4f} -> {float(cb.mean()):.4f}", flush=True)
    print("EMIT_CARVED_DONE", a.tag)


def selftest():
    """Phantom plumbing test, no GPU/model/slab: two 6-vox wraps in seamless contact + a third across an air
    gap. Ideal-but-noisy affinities (seam known only to the aff channels, NOT to intensity). Asserts:
    (1) MWS separates the two touching wraps and does not merge across the air gap;
    (2) absorb_small repairs injected over-fragmentation without erasing the true seam;
    (3) both carve modes place >=80% of carved voxels within 1 voxel of the true seam;
    (4) neither carve digs into wrap cores away from the seam."""
    rng = np.random.default_rng(0)
    Z, Y, X = 32, 40, 64
    gt = np.zeros((Z, Y, X), np.int32)
    gt[:, 4:10, :] = 1                                    # wrap 1
    gt[:, 10:16, :] = 2                                   # wrap 2, zero-gap contact at y=10 plane
    gt[:, 20:26, :] = 3                                   # wrap 3 across a 4-vox air gap
    fiber_prob = np.where(gt > 0, 0.95, 0.03).astype(np.float32)   # intensity shows NO seam at y=10
    offsets = build_offsets("1,3,9")
    aff = np.zeros((len(offsets),) + gt.shape, np.float32)
    for i, off in enumerate(offsets):
        g = np.zeros_like(gt)
        dz, dy, dx = off
        z1, y1, x1 = Z - dz, Y - dy, X - dx
        g[:z1, :y1, :x1] = gt[dz:, dy:, dx:]
        same = (gt > 0) & (g == gt)
        aff[i] = np.where(same, 0.9, 0.1)
        aff[i] += rng.normal(0, 0.05, aff[i].shape).astype(np.float32)   # calibration noise
    aff = np.clip(aff, 0.0, 1.0)
    # real-domain nastiness: patches of noisy long-range affs INSIDE wraps (the v1 fragmentation trigger)
    for i, off in enumerate(offsets):
        if sum(abs(c) for c in off) > 1:
            noise_patch = rng.random(aff[i].shape) < 0.10
            aff[i][noise_patch & (gt > 0)] = rng.random(int((noise_patch & (gt > 0)).sum())) * 0.4

    class A: pass
    a = A(); a.fiber_thr = 0.5; a.mode = "both"; a.min_inst = 800; a.aff_carve_thr = 0.3
    a.stride_scale = 0; a.tag = "selftest"
    fiber = fiber_prob >= a.fiber_thr
    labels = mws_instances(aff, fiber, offsets, stride_scale=0, long_repulsive_only=True)
    n_raw = len(np.unique(labels[labels > 0]))
    labels_abs = repair_fragments(labels, aff, merge_thr=0.6, min_size=a.min_inst)
    ids = np.unique(labels_abs[labels_abs > 0])
    print(f"raw instances {n_raw} -> repaired {len(ids)}")
    # (1)+(2): each wrap one dominant id; touching wraps have DIFFERENT ids; gap wrap different from both
    dom = {}
    for w in (1, 2, 3):
        vals, cnts = np.unique(labels_abs[(gt == w) & (labels_abs > 0)], return_counts=True)
        dom[w] = int(vals[np.argmax(cnts)])
        purity = cnts.max() / cnts.sum()
        assert purity > 0.9, f"wrap {w} fragmented after absorb (purity {purity:.2f})"
    assert dom[1] != dom[2], "MWS merged the two seamlessly-touching wraps"
    assert dom[3] not in (dom[1], dom[2]), "MWS merged across the air gap"
    # (3)+(4): carve localization for both modes
    seam_band = np.zeros_like(fiber)
    seam_band[:, 8:12, :] = True                          # +-2 vox of the true y=10 seam plane
    for name, (fc, cb, carve) in (
            ("mws", carve_from_labels(labels_abs, fiber, fiber_prob)),
            ("affcarve", aff_carve(aff, fiber, fiber_prob, a.aff_carve_thr))):
        frac_at_seam = float((carve & seam_band).sum()) / max(1, int(carve.sum()))
        core = (gt > 0) & ~seam_band
        core_damage = float((carve & core).sum()) / float(core.sum())
        print(f"{name}: carved {int(carve.sum())}, at-seam {frac_at_seam:.2f}, core damage {core_damage:.4f}")
        assert frac_at_seam >= 0.8, f"{name}: carve not localized at seam ({frac_at_seam:.2f})"
        assert core_damage < 0.02, f"{name}: carve digs into wrap cores ({core_damage:.4f})"
        seam_open = (fc[:, 10, :] < 0.5).mean()
        assert seam_open > 0.5, f"{name}: seam plane not opened ({seam_open:.2f})"

    # RANK carve ("cut on doubt"): must (a) find the same seam as the absolute threshold, and (b) survive the exact
    # failure that erased the real slab -- a monotone shift of every affinity below the absolute threshold. The
    # ep20 head maxed out at 0.30, so `aff_carve(thr=0.3)` severed everything; a rank rule sees identical ORDER.
    # The budget is a REAL knob and must be matched to how prevalent true seam edges actually are: it names the
    # fraction of edges to sever, so over-budgeting necessarily digs past the seam into wrap cores. Measure the
    # true seam prevalence here and spend exactly that, which is also how it should be set on real data (from a
    # split budget), never as a fixed constant.
    # NOTE the budget must come from the TRUE cross-wrap edges (gt differs on the two endpoints), not from the
    # seam_band -- that band is a +-2 voxel tolerance window for the localization assertion and is 4 planes of 20
    # wide, so using it asks for a 23% carve and (correctly) trips the min_keep guard.
    def _shift_gt(g, off):
        out = np.zeros_like(g)
        dz, dy, dx = off
        out[:g.shape[0] - dz, :g.shape[1] - dy, :g.shape[2] - dx] = g[dz:, dy:, dx:]
        return out
    n_edge = n_cross = 0
    for o in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        valid = fiber & shift_fiber(fiber, o)
        gt_b = _shift_gt(gt, o)
        n_edge += int(valid.sum())
        n_cross += int((valid & (gt > 0) & (gt_b > 0) & (gt != gt_b)).sum())
    budget = n_cross / n_edge
    fcr, cbr, carver, thr_r = aff_carve_rank(aff, fiber, fiber_prob, budget)
    frac_at_seam = float((carver & seam_band).sum()) / max(1, int(carver.sum()))
    core_dmg = float((carver & ((gt > 0) & ~seam_band)).sum()) / float(((gt > 0) & ~seam_band).sum())
    assert frac_at_seam >= 0.8, f"rank carve not localized at seam ({frac_at_seam:.2f}) at budget {budget:.4f}"
    assert core_dmg < 0.02, f"rank carve digs into wrap cores ({core_dmg:.4f})"
    aff_shift = aff * 0.30                                # squash everything below the absolute 0.3 threshold
    _, cb_abs, carve_abs = aff_carve(aff_shift, fiber, fiber_prob, 0.3)
    kept_abs = float(cb_abs.sum()) / float(fiber.sum())
    _, cb_rank, carve_rank, _ = aff_carve_rank(aff_shift, fiber, fiber_prob, budget)
    kept_rank = float(cb_rank.sum()) / float(fiber.sum())
    assert kept_abs < 0.05, f"the shift should destroy the ABSOLUTE carve (kept {kept_abs:.1%}) -- test is invalid"
    assert kept_rank > 0.8, f"rank carve must be shift-invariant, kept only {kept_rank:.1%}"
    assert np.array_equal(carve_rank, carver), "a monotone shift must not change the rank carve AT ALL"
    print(f"RANK carve: budget {budget:.4f} (= true seam prevalence) at-seam {frac_at_seam:.2f} "
          f"core-damage {core_dmg:.4f} thr={thr_r:.4f} | under a global x0.30 squash: "
          f"absolute keeps {kept_abs:.1%} (destroyed), rank keeps {kept_rank:.1%} (identical carve)")
    print("SELFTEST_OK")


def swtest():
    """Gate for the transfer-optimized sliding window: compare against a brute-force fp32 reference on a
    small random volume + fixed conv net. Asserts max |prob diff| < 2e-3 (fp16 transfer precision), i.e.
    far below every downstream decision (0.1 dip resolution, 0.5 threshold)."""
    import torch
    from tstr_eval_aff import sliding_window_logits
    torch.manual_seed(0)
    n_out, patch = 6, [32, 32, 32]
    net = torch.nn.Conv3d(1, n_out, 3, padding=1)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(dev).eval()
    data = np.random.default_rng(0).normal(size=(1, 20, 70, 75)).astype(np.float32)  # z < patch -> padded
    fast = sliding_window_logits(net, data.copy(), patch, n_out, dev)
    # brute-force reference: identical tiling math, fp32 end-to-end, no cropping tricks
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    dpad, slicer = pad_nd_image(data.copy(), patch, 'constant', {'constant_values': 0}, True)
    t = torch.from_numpy(dpad).float()
    Z, Y, X = t.shape[1:]
    steps = compute_steps_for_sliding_window((Z, Y, X), patch, 0.5)
    g = compute_gaussian(tuple(patch), sigma_scale=1. / 8, device=dev, dtype=torch.float32).cpu()
    acc = torch.zeros((n_out, Z, Y, X)); wacc = torch.zeros((1, Z, Y, X))
    pz, py, px = patch
    with torch.no_grad():
        for sz in steps[0]:
            for sy in steps[1]:
                for sx in steps[2]:
                    with torch.autocast(dev.type, enabled=(dev.type == 'cuda')):
                        out = net(t[:, sz:sz+pz, sy:sy+py, sx:sx+px][None].to(dev))[0].float()
                    acc[:, sz:sz+pz, sy:sy+py, sx:sx+px] += out.cpu() * g
                    wacc[:, sz:sz+pz, sy:sy+py, sx:sx+px] += g
    ref = (acc / wacc).numpy()[(slice(None),) + tuple(slicer[1:])]
    d_logit = float(np.abs(fast - ref).max())
    d_prob = float(np.abs(1/(1+np.exp(-fast)) - 1/(1+np.exp(-ref))).max())
    print(f"swtest: shapes {fast.shape}=={ref.shape}, max|dlogit|={d_logit:.2e}, max|dprob|={d_prob:.2e}")
    assert fast.shape == ref.shape, "optimized window changed output shape"
    assert d_prob < 2e-3, f"optimized window drifted: {d_prob}"
    print("SWTEST_OK")


def main():
    if "--selftest" in sys.argv:
        selftest(); return
    if "--swtest" in sys.argv:
        swtest(); return
    import torch, nrrd
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True); ap.add_argument("--ckpt", default="checkpoint_final.pth")
    ap.add_argument("--ranges", default="1,3,9,27"); ap.add_argument("--tag", default="afflong")
    ap.add_argument("--gpu", default="0", help="GPU id, or comma list e.g. 0,1 -> tile-sharded multi-GPU emit")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--fiber_thr", type=float, default=0.5)
    ap.add_argument("--stride_scale", type=int, default=3)
    ap.add_argument("--mode", choices=["both", "mws", "affcarve"], default="both")
    ap.add_argument("--min_inst", type=int, default=100000,
                    help="absorb instances smaller than this (voxels) before carving; 0 disables")
    ap.add_argument("--merge_thr", type=float, default=0.6,
                    help="agglomerate adjacent instances while mean r=1 boundary affinity >= this; 1.0 disables")
    ap.add_argument("--aff_carve_thr", type=float, default=0.3,
                    help="r=1 affinity below this on a fiber-fiber edge => carve both endpoints (affcarve mode)")
    ap.add_argument("--carve_budget", type=float, default=0.0,
                    help="if >0, use the RANK carve instead of the absolute threshold: sever this FRACTION of the "
                         "lowest-affinity fiber-fiber edges. Rank-based => immune to a global calibration shift "
                         "(an absolute thr wiped the entire fiber when the head's max affinity fell to 0.30), and "
                         "it states the cost structure directly: splits under ~40-60px are free, merges are not.")
    a = ap.parse_args()
    offsets = build_offsets(a.ranges); n_aff = len(offsets)
    gpu_ids = [int(g) for g in str(a.gpu).split(",") if str(g).strip() != ""]
    devices = [torch.device(f"cuda:{g}") for g in gpu_ids]
    device = devices[0]
    out_dir = f"/root/slab_pred_{a.tag}"
    os.makedirs(out_dir, exist_ok=True)

    # the network pass is the expensive part (~1h at slab scale); cache it so decode params iterate for free
    if os.path.exists(f"{out_dir}/aff_f16.npy") and os.path.exists(f"{out_dir}/fiber.npy"):
        fiber_prob = np.load(f"{out_dir}/fiber.npy")
        aff = np.load(f"{out_dir}/aff_f16.npy").astype(np.float32)
        print("reusing cached fiber/aff", aff.shape, flush=True)
    else:
        vol = nrrd.read("/root/data/slab/volume.nrrd")[0].astype(np.float32)
        p, net = build_aff_network(a.model_dir, a.fold, a.ckpt, device)
        # one INDEPENDENTLY-built net per extra device (deepcopy is unsafe: the aff-head forward_pre_hook lambda
        # closes over its own instance, so a copy would write features into the original net)
        nets = [net] + [build_aff_network(a.model_dir, a.fold, a.ckpt, d)[1] for d in devices[1:]]
        patch = list(p.configuration_manager.patch_size)
        n_seg = int(p.label_manager.num_segmentation_heads)   # medial+BW head = softmax over {bg,fiber}
        data = preprocess(p, vol)
        print("slab preprocessed", data.shape, "patch", patch, "n_seg", n_seg, "n_aff", n_aff,
              "gpus", gpu_ids, flush=True)
        logits = sliding_window_logits(net, data, patch, n_seg + n_aff, device, devices=devices, nets=nets)
        seg = logits[:n_seg]
        aff = (1.0 / (1.0 + np.exp(-logits[n_seg:]))).astype(np.float32)
        if n_seg >= 2:                                        # softmax: fiber = P(ch1)
            e = np.exp(seg - seg.max(0, keepdims=True))
            fiber_prob = (e[1] / (e.sum(0) + 1e-9)).astype(np.float32)
        else:                                                 # single-channel sigmoid head
            fiber_prob = (1.0 / (1.0 + np.exp(-seg[0]))).astype(np.float32)
        np.save(f"{out_dir}/fiber.npy", fiber_prob)
        np.save(f"{out_dir}/aff_f16.npy", aff.astype(np.float16))
    run_decodes(fiber_prob, aff, offsets, out_dir, a)


if __name__ == "__main__":
    main()
