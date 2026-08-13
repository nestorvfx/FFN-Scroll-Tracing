#!/usr/bin/env python3
"""Wrap-ERL: the production-faithful surface-detection metric.

The tracer consumes predictions as: binarize (nonzero) -> DT-based thinning -> ridge polylines -> snap.
One ridge BRIDGE between two wraps shifts the winding number for every wrap outward (unbounded cost);
sub-40px gaps are interpolated by regularizers (nearly free). So the metric that matches downstream reality:

  wrap-ERL = length-weighted expected error-free arc length of a wrap's ridge before a merge bridge,
  computed on the DT skeleton (mirrors Thinning.cpp: cv2 L1 distance transform ridge) of the thresholded
  prediction, per z-slice (the winding plane), scored against the 68-wrap slab truth.

Bridge event = 8-adjacent skeleton pixel pair with DIFFERENT truth wrap labels (both >0). Merge-biased:
a bridge terminates runs on BOTH wraps. ERL uses the equal-split approximation L/(B+1) per wrap-slice,
aggregated length-weighted: ERL = sum_w L_w^2/(B_w+1) / sum_w L_w.

Also reported: fragments/wrap-slice + skeleton coverage (split/continuity guardrail), contact-dip stats on
the probability field (fast screen), sector bootstrap CIs (12 angular sectors around the umbilicus),
held-out odd sectors, threshold sweep, and a blur-degradation sanity gate.

Usage: python wrap_erl.py --slab /root/data/slab --pred /root/slab_pred_new/fiber.npy --tag new
       [--thrs 0.3,0.5,0.7,0.9] [--degrade]
"""
import os, sys, json, argparse
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def slice_metrics(binary, truth2d, sector2d):
    """One z-slice: DT skeleton -> per-wrap lengths, bridge counts (per sector), fragments."""
    from skimage.morphology import medial_axis
    skel = medial_axis(binary)
    lab = np.where(skel, truth2d, 0)
    # bridge edges: 8-adjacent skeleton pixels with different positive labels
    bridges = []           # (wrap_a, wrap_b, sector)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = lab[max(0, -dy):lab.shape[0] - max(0, dy), max(0, -dx):lab.shape[1] - max(0, dx)]
        b = lab[max(0, dy):lab.shape[0] + min(0, dy) or None, max(0, dx):lab.shape[1] + min(0, dx) or None]
        s = sector2d[max(0, -dy):sector2d.shape[0] - max(0, dy), max(0, -dx):sector2d.shape[1] - max(0, dx)]
        m = (a > 0) & (b > 0) & (a != b)
        if m.any():
            for wa, wb, sec in zip(a[m].ravel(), b[m].ravel(), s[m].ravel()):
                bridges.append((int(wa), int(wb), int(sec)))
    # per-wrap skeleton length + sector assignment (majority) + fragment count
    out = {}
    for w in np.unique(lab[lab > 0]):
        m = lab == w
        L = int(m.sum())
        secs, cnts = np.unique(sector2d[m], return_counts=True)
        sec = int(secs[np.argmax(cnts)])
        ncomp = ndi.label(m, structure=np.ones((3, 3)))[1]
        out[int(w)] = dict(L=L, sector=sec, frags=int(ncomp))
    return out, bridges


def erl_from_records(recs, keep_sectors=None):
    """recs: list of per-(slice,wrap) dicts with L, sector, bridges count. Length-weighted ERL."""
    num = den = 0.0
    for r in recs:
        if keep_sectors is not None and r["sector"] not in keep_sectors:
            continue
        num += r["L"] ** 2 / (r["B"] + 1.0)
        den += r["L"]
    return (num / den) if den else 0.0


def _slice_job(args):
    binary, truth2d, sector, z = args
    per_wrap, bridges = slice_metrics(binary, truth2d, sector)
    return z, per_wrap, bridges


def evaluate(prob, truth, sector, thr, jobs=None):
    Z = truth.shape[0]
    tasks = [(prob[z] >= thr, truth[z], sector, z) for z in range(Z)]
    jobs = jobs if jobs is not None else min(16, os.cpu_count() or 1, Z)
    if jobs <= 1:
        results = [_slice_job(t) for t in tasks]
    else:
        from multiprocessing import Pool
        with Pool(jobs) as pool:
            results = list(pool.imap(_slice_job, tasks, chunksize=2))
    recs, all_bridges = [], []
    for z, per_wrap, bridges in sorted(results, key=lambda r: r[0]):
        bcount = {}
        for wa, wb, sec in bridges:
            bcount[wa] = bcount.get(wa, 0) + 1
            bcount[wb] = bcount.get(wb, 0) + 1
        for w, d in per_wrap.items():
            recs.append(dict(L=d["L"], sector=d["sector"], B=bcount.get(w, 0), frags=d["frags"], z=z, w=w))
        all_bridges += bridges
    return recs, all_bridges


def bootstrap_ci(recs, sectors, n=500, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    slist = sorted(sectors)
    for _ in range(n):
        pick = set(rng.choice(slist, size=len(slist), replace=True).tolist())
        vals.append(erl_from_records(recs, keep_sectors=pick))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def contact_screen(prob, truth):
    """Contact-dip screen on the probability field: for real wrap-wrap near-contacts (from truth), does the
    prediction produce a separating minimum? Uses measure_fusion.contact_dips on prob*255."""
    from measure_fusion import contact_dips
    x = (prob * 255.0).astype(np.float32)
    dips = np.concatenate([contact_dips(x, truth, axis=ax, grain_sigma=1.0)[0] for ax in (1, 2)])
    if dips.size == 0:
        return {}
    d = dips / 255.0                                        # prob units
    return dict(n=int(d.size), mean_dip=round(float(d.mean()), 4),
                resolved_at_0p1=round(float((d > 0.1).mean()), 4),
                resolved_at_0p3=round(float((d > 0.3).mean()), 4),
                resolved_at_0p5=round(float((d > 0.5).mean()), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slab", required=True); ap.add_argument("--pred", required=True)
    ap.add_argument("--tag", default="pred"); ap.add_argument("--thrs", default="0.3,0.5,0.7,0.9")
    ap.add_argument("--degrade", action="store_true", help="also score a blurred copy (sanity: must be worse)")
    a = ap.parse_args()
    import nrrd
    truth = nrrd.read(os.path.join(a.slab, "truth.nrrd"))[0].astype(np.int32)
    bb = json.load(open(os.path.join(a.slab, "bbox.json")))
    yy = np.arange(truth.shape[1])[:, None] - bb["umb_local_y"]
    xx = np.arange(truth.shape[2])[None, :] - bb["umb_local_x"]
    sector = ((np.arctan2(yy, xx) + np.pi) / (2 * np.pi) * 12).astype(int) % 12
    prob = np.load(a.pred).astype(np.float32)
    if prob.max() > 1.5:
        prob /= 255.0

    results = {"tag": a.tag, "screen": contact_screen(prob, truth)}
    for thr in [float(t) for t in a.thrs.split(",")]:
        recs, bridges = evaluate(prob, truth, sector, thr)
        odd = {s for s in range(12) if s % 2 == 1}
        lo, hi = bootstrap_ci(recs, set(range(12)))
        entry = dict(erl_whole=round(erl_from_records(recs), 1),
                     erl_ci=[round(lo, 1), round(hi, 1)],
                     erl_odd_sectors=round(erl_from_records(recs, odd), 1),
                     bridges_total=len(bridges),
                     mean_frags=round(float(np.mean([r["frags"] for r in recs])), 2),
                     skel_len_total=int(sum(r["L"] for r in recs)))
        results[f"thr_{thr}"] = entry
        print(a.tag, "thr", thr, json.dumps(entry), flush=True)
    if a.degrade:
        pd = ndi.gaussian_filter(prob, 2.0)
        recs, bridges = evaluate(pd, truth, sector, 0.5)
        results["degraded_thr_0.5"] = dict(erl_whole=round(erl_from_records(recs), 1), bridges=len(bridges))
        print(a.tag, "DEGRADED", json.dumps(results["degraded_thr_0.5"]), flush=True)
    json.dump(results, open(f"/root/wrap_erl_{a.tag}.json", "w"), indent=1)
    print("WRAP_ERL_DONE", a.tag)


if __name__ == "__main__":
    main()
