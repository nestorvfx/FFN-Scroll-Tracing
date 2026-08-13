#!/usr/bin/env python3
"""Metric v2: expected corrupted-text fraction (CTF) -- the downstream-faithful merge cost.

Fixes three verified defects of wrap_erl v1 (kept alongside for continuity):
  1. BRIDGE DEFINITION: v1 counted 8-adjacent skeleton pixel pairs with different wrap labels -- adjacency,
     not connection. Two legitimately separate ridges 1px apart false-positive; cost of a real bridge is
     non-local. v2: a bridge = a CONNECTED COMPONENT of the thresholded-then-thinned skeleton that spans
     >=2 GT wraps (exactly what makes the spiral cross onto the wrong wrap).
  2. COST WEIGHTING: v1 used L^2 (wrap length). The true cost is the RADIAL BLAST RADIUS: a bridge between
     wraps (k, k+1) shifts the winding number of every wrap OUTWARD of k, corrupting all their text. v2 cost
     of a slice = arc length of all wraps radially outward of the innermost spanning bridge, plus the
     spanned wraps themselves. CTF = corrupted arc / total arc (what fraction of text gets the wrong wrap).
  3. RANKING STATISTIC: CTF is honestly heavy-tailed (inner bridges ARE that bad) -> also reports a tamed
     per-slice statistic sum(log1p(blast_wraps)) over spanning components, for ranking under seed noise;
     plus bootstrap-over-slices CIs for both.

Radial wrap order is computed from mean distance to the umbilicus per wrap label (never assume label order).

CALIBRATIONS (run these before trusting any model number):
  --pred truth        : GT binary as prediction = the ceiling of any intensity-faithful model WITHOUT
                        carving. If this scores high CTF, no non-carving model can do better -- quantifies
                        exactly how much headroom lives in explicit seam zeros.
  --pred truth_carved : GT binary with 1-vox carve between different wraps = the ideal carved output.
                        MUST score CTF ~ 0; validates the metric itself.

Usage: python wrap_erl2.py --slab /root/data/slab --pred <fiber.npy|truth|truth_carved> --tag <tag>
       [--thrs 0.5,0.7] [--selftest]
"""
import os, sys, json, argparse
import numpy as np
from scipy import ndimage as ndi

CONN8 = np.ones((3, 3), bool)


def radial_order(truth, umb_y, umb_x):
    """wrap label -> radial rank (0 = innermost), from mean distance to umbilicus over all slices."""
    labels = np.unique(truth[truth > 0])
    yy = np.arange(truth.shape[1])[:, None] - umb_y
    xx = np.arange(truth.shape[2])[None, :] - umb_x
    rr = np.sqrt(yy * yy + xx * xx)
    mean_r = ndi.mean(np.broadcast_to(rr, truth.shape), labels=truth, index=labels)
    rank = {int(l): int(i) for i, l in enumerate(labels[np.argsort(mean_r)])}
    return rank, labels


def slice_ctf(binary2d, truth2d, rank):
    """One z-slice -> (corrupted_arc, total_arc, tamed_bridge_stat, n_spanning, per-wrap skel lengths dict).

    skeleton components (8-conn) of the binarized prediction; component label-span from GT overlap
    (truth 0 = unlabeled, ignored); innermost spanning bridge corrupts every wrap radially outward."""
    from skimage.morphology import medial_axis
    skel = medial_axis(binary2d)
    lab = np.where(skel, truth2d, 0)
    wraps, counts = np.unique(lab[lab > 0], return_counts=True)
    L = {int(w): int(c) for w, c in zip(wraps, counts)}
    total_arc = float(sum(L.values()))
    if total_arc == 0:
        return 0.0, 0.0, 0.0, 0, L
    comp, n_comp = ndi.label(skel, structure=CONN8)
    # component -> set of wrap labels it overlaps (pairs via unique on combined key)
    on = skel & (truth2d > 0)
    pair = comp[on].astype(np.int64) * (int(truth2d.max()) + 1) + truth2d[on].astype(np.int64)
    uniq = np.unique(pair)
    comp_wraps = {}
    for p in uniq.tolist():
        c, w = divmod(p, int(truth2d.max()) + 1)
        comp_wraps.setdefault(c, set()).add(w)
    spanning = [ws for ws in comp_wraps.values() if len(ws) >= 2]
    n_ranks = len(rank)
    tamed = 0.0
    innermost = None
    for ws in spanning:
        rmin = min(rank[w] for w in ws)
        blast = n_ranks - rmin                     # wraps corrupted by this bridge (itself + all outward)
        tamed += np.log1p(blast)
        innermost = rmin if innermost is None else min(innermost, rmin)
    if innermost is None:
        return 0.0, total_arc, 0.0, 0, L
    corrupted = float(sum(l for w, l in L.items() if rank[w] >= innermost))
    return corrupted, total_arc, tamed, len(spanning), L


def _ctf_job(args):
    binary2d, truth2d, rank, z = args
    c, t, tamed, nsp, _ = slice_ctf(binary2d, truth2d, rank)
    return dict(z=z, corrupted=c, total=t, tamed=tamed, n_spanning=nsp)


def evaluate_ctf(prob, truth, rank, thr, jobs=None):
    Z = truth.shape[0]
    tasks = [(prob[z] >= thr, truth[z], rank, z) for z in range(Z)]
    jobs = jobs if jobs is not None else min(16, os.cpu_count() or 1, Z)
    if jobs <= 1:
        return [_ctf_job(t) for t in tasks]
    from multiprocessing import Pool
    with Pool(jobs) as pool:
        per_slice = list(pool.imap(_ctf_job, tasks, chunksize=2))
    per_slice.sort(key=lambda r: r["z"])
    return per_slice


def aggregate(per_slice):
    c = sum(r["corrupted"] for r in per_slice)
    t = sum(r["total"] for r in per_slice)
    return dict(ctf=round(c / t, 4) if t else 0.0,
                tamed_bridges=round(float(np.mean([r["tamed"] for r in per_slice])), 3),
                spanning_total=int(sum(r["n_spanning"] for r in per_slice)))


def bootstrap(per_slice, n=500, seed=0):
    rng = np.random.default_rng(seed)
    ctfs, tameds = [], []
    Z = len(per_slice)
    for _ in range(n):
        pick = rng.integers(0, Z, Z)
        rs = [per_slice[i] for i in pick]
        t = sum(r["total"] for r in rs)
        ctfs.append(sum(r["corrupted"] for r in rs) / t if t else 0.0)
        tameds.append(float(np.mean([r["tamed"] for r in rs])))
    pct = lambda v: [round(float(np.percentile(v, q)), 4) for q in (2.5, 97.5)]
    return dict(ctf_ci=pct(ctfs), tamed_ci=pct(tameds))


def make_truth_pred(truth, carved=False):
    """truth binary; optionally with a 1-vox carve wherever a voxel touches a DIFFERENT wrap (26-conn 2D-safe:
    per-slice 8-conn is what the per-slice skeleton sees)."""
    binary = (truth > 0).astype(np.float32)
    if not carved:
        return binary
    out = binary.copy()
    for z in range(truth.shape[0]):
        t = truth[z]
        big = np.where(t > 0, t, np.iinfo(np.int32).max)
        mn = ndi.minimum_filter(big, size=3)
        mx = ndi.maximum_filter(np.where(t > 0, t, 0), size=3)
        carve = (t > 0) & (mn < np.iinfo(np.int32).max) & (mx != mn)
        out[z][carve] = 0.0
    return out


def selftest():
    """Phantom: 4 concentric rings (truth), predictions with known bridges -> known CTF.
    Ring radii 10,16,22,28, width 2; umbilicus at center of a 64x64 slice."""
    H = W = 64; cy = cx = 32
    yy, xx = np.mgrid[:H, :W]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    truth2d = np.zeros((H, W), np.int32)
    for i, r0 in enumerate((10, 16, 22, 28)):
        truth2d[(rr >= r0 - 1) & (rr <= r0 + 1)] = i + 1
    truth = truth2d[None]                                  # single-slice volume
    rank, _ = radial_order(truth, cy, cx)
    assert [rank[i] for i in (1, 2, 3, 4)] == [0, 1, 2, 3], "radial order wrong"
    clean = (truth2d > 0).astype(np.float32)[None]
    per = evaluate_ctf(clean, truth, rank, 0.5)
    agg = aggregate(per)
    assert agg["spanning_total"] == 0 and agg["ctf"] == 0.0, f"clean rings must have no bridges: {agg}"
    # bridge rings 3-4 (outermost pair): corrupted = rings 3+4 arcs
    b34 = clean.copy(); b34[0, cy - 28:cy - 20, cx] = 1.0   # radial spoke crossing rings 3 and 4
    per = evaluate_ctf(b34, truth, rank, 0.5)
    agg34 = aggregate(per)
    assert agg34["spanning_total"] >= 1, "spoke must create a spanning component"
    # bridge rings 1-2 (innermost pair): corrupted = everything outward of ring 1 = all four rings ~ CTF ~1
    b12 = clean.copy(); b12[0, cy - 17:cy - 8, cx] = 1.0
    agg12 = aggregate(evaluate_ctf(b12, truth, rank, 0.5))
    assert agg12["ctf"] > agg34["ctf"] + 0.2, f"inner bridge must cost more: {agg12} vs {agg34}"
    assert agg12["ctf"] > 0.9, f"innermost bridge corrupts ~all outward text: {agg12}"
    # v1-style false positive check: two separate rings 1px apart must NOT count (they are separate comps)
    tight = np.zeros((H, W), np.int32)
    tight[(rr >= 12) & (rr <= 13)] = 1
    tight[(rr >= 15) & (rr <= 16)] = 2                     # ~1px gap, no connection
    ttruth = tight[None]
    trank, _ = radial_order(ttruth, cy, cx)
    tagg = aggregate(evaluate_ctf((tight > 0).astype(np.float32)[None], ttruth, trank, 0.5))
    assert tagg["spanning_total"] == 0, f"adjacent-but-disconnected ridges must not bridge: {tagg}"
    # truth_carved calibration on a FUSED phantom (rings touching): carved must uncorrupt
    fused2d = np.zeros((H, W), np.int32)
    fused2d[(rr >= 10) & (rr < 13)] = 1
    fused2d[(rr >= 13) & (rr <= 16)] = 2                   # zero-gap contact at r=13
    ftruth = fused2d[None]
    frank, _ = radial_order(ftruth, cy, cx)
    raw = make_truth_pred(ftruth, carved=False)
    carved = make_truth_pred(ftruth, carved=True)
    agg_raw = aggregate(evaluate_ctf(raw, ftruth, frank, 0.5))
    agg_car = aggregate(evaluate_ctf(carved, ftruth, frank, 0.5))
    assert agg_raw["ctf"] > 0.9, f"fused truth binary must bridge (this is the whole problem): {agg_raw}"
    assert agg_car["ctf"] == 0.0, f"carved truth must be clean: {agg_car}"
    print(f"selftest: clean {agg} | outer-bridge {agg34} | inner-bridge {agg12} | "
          f"tight-no-FP {tagg} | fused raw {agg_raw} -> carved {agg_car}")
    print("SELFTEST_OK")


def main():
    if "--selftest" in sys.argv:
        selftest(); return
    ap = argparse.ArgumentParser()
    ap.add_argument("--slab", required=True); ap.add_argument("--pred", required=True)
    ap.add_argument("--tag", default="pred"); ap.add_argument("--thrs", default="0.5,0.7")
    a = ap.parse_args()
    import nrrd
    truth = nrrd.read(os.path.join(a.slab, "truth.nrrd"))[0].astype(np.int32)
    bb = json.load(open(os.path.join(a.slab, "bbox.json")))
    rank, labels = radial_order(truth, bb["umb_local_y"], bb["umb_local_x"])
    if a.pred == "truth":
        prob = make_truth_pred(truth, carved=False); thrs = [0.5]
    elif a.pred == "truth_carved":
        prob = make_truth_pred(truth, carved=True); thrs = [0.5]
    else:
        prob = np.load(a.pred).astype(np.float32)
        if prob.max() > 1.5:
            prob /= 255.0
        thrs = [float(t) for t in a.thrs.split(",")]
    results = {"tag": a.tag, "n_wraps": int(len(labels))}
    for thr in thrs:
        per = evaluate_ctf(prob, truth, rank, thr)
        entry = aggregate(per)
        entry.update(bootstrap(per))
        results[f"thr_{thr}"] = entry
        print(a.tag, "thr", thr, json.dumps(entry), flush=True)
    json.dump(results, open(f"/root/wrap_ctf_{a.tag}.json", "w"), indent=1)
    print("WRAP_CTF_DONE", a.tag)


if __name__ == "__main__":
    main()
