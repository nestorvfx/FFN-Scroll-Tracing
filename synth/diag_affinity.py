#!/usr/bin/env python3
"""DIRECT affinity-quality diagnostic -- the gate that must pass BEFORE any decode is trusted.

Motivation (2026-07-19): the FT+MALIS arm trained fine (loss down, pseudo-dice 0.616->0.636) yet the decoded
wrap-ERL did not improve, and the GASP decode changed the bridge count by 0.09% -- i.e. it carved essentially
nothing. That is only possible if the predicted affinities do not actually separate wraps. We had NO measurement
of affinity quality itself: we went straight from the training loss to end-to-end ERL, so a dead affinity head
was indistinguishable from a good one that the decode wasted.

This measures the ONE thing that must be true for any affinity decode to work:

    predicted affinity must be LOW across a true wrap-wrap contact and HIGH within a wrap.

For each offset channel, using the slab truth as ground truth, we report over edges whose SOURCE is fiber:
  * mean affinity on WITHIN-wrap edges  (target 1) -> want high
  * mean affinity on CROSS-wrap edges   (target 0) -> want low
  * separation = within_mean - cross_mean          -> the usable signal; ~0 means the head learned nothing
  * AUC (probability a random within-edge scores above a random cross-edge) -> 0.5 = chance
  * the same restricted to ZERO-EVIDENCE contacts (CT dip below --dip_thr), which is the regime that matters
A head that is useful for separation needs AUC comfortably above 0.5 (and above all on the zero-evidence subset).

Usage:
  python diag_affinity.py --slab /root/data/slab --aff /root/slab_pred_X/aff_f16.npy --ranges 1,3,9,27
"""
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tstr_eval_aff import build_offsets


def auc_mannwhitney(pos, neg, cap=2_000_000, seed=0):
    """AUC = P(random pos > random neg), via rank statistic. Subsampled for tractability."""
    rng = np.random.default_rng(seed)
    if pos.size > cap:
        pos = rng.choice(pos, cap, replace=False)
    if neg.size > cap:
        neg = rng.choice(neg, cap, replace=False)
    if pos.size == 0 or neg.size == 0:
        return float('nan')
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind='stable')
    ranks = np.empty(allv.size, np.float64)
    ranks[order] = np.arange(1, allv.size + 1)
    # average ranks over ties so a constant predictor scores exactly 0.5
    uniq, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    rsum = np.zeros(uniq.size, np.float64)
    np.add.at(rsum, inv, ranks)
    ranks = (rsum / cnt)[inv]
    r_pos = ranks[:pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slab", required=True)
    ap.add_argument("--aff", required=True)
    ap.add_argument("--ranges", default="1,3,9,27")
    ap.add_argument("--dip_thr", type=float, default=8.0,
                    help="CT dip (gray levels /255) below which a contact counts as ZERO-EVIDENCE")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import nrrd
    truth = nrrd.read(os.path.join(a.slab, "truth.nrrd"))[0].astype(np.int32)
    vol = nrrd.read(os.path.join(a.slab, "volume.nrrd"))[0].astype(np.float32)
    aff = np.load(a.aff)
    if aff.dtype != np.float32:
        aff = aff.astype(np.float32)
    offsets = build_offsets(a.ranges)
    assert aff.shape[0] == len(offsets), f"aff has {aff.shape[0]} channels but ranges give {len(offsets)}"
    Z, Y, X = truth.shape
    res = {"aff": a.aff, "ranges": a.ranges, "channels": []}
    all_within, all_cross, all_cross_blind = [], [], []
    for c, (dz, dy, dx) in enumerate(offsets):
        z0, z1 = max(0, -dz), Z - max(0, dz)
        y0, y1 = max(0, -dy), Y - max(0, dy)
        x0, x1 = max(0, -dx), X - max(0, dx)
        s = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
        t = (slice(z0 + dz, z1 + dz), slice(y0 + dy, y1 + dy), slice(x0 + dx, x1 + dx))
        ta, tb = truth[s], truth[t]
        av = aff[c][s]
        within = (ta > 0) & (tb > 0) & (ta == tb)
        cross = (ta > 0) & (tb > 0) & (ta != tb)
        w_v = av[within]; c_v = av[cross]
        # ZERO-EVIDENCE subset: a contact is "blind" when the CT does not DIP along the path between the two
        # endpoints -- i.e. the wraps are fused with no visible air gap. NOTE this must be the minimum along the
        # intervening voxels, NOT |vol[v]-vol[v+off]| (an endpoint brightness difference says nothing about a gap,
        # and is meaningless at r=27 where the endpoints are far apart). r=1 has no intervening voxel, so the dip
        # is undefined there and the blind subset is reported as None.
        r = max(abs(dz), abs(dy), abs(dx))
        if r >= 2:
            u = (np.sign(dz), np.sign(dy), np.sign(dx))
            pathmin = None
            for k in range(1, r):                          # intervening voxels v + k*unit
                ks = (slice(z0 + u[0] * k, z1 + u[0] * k),
                      slice(y0 + u[1] * k, y1 + u[1] * k),
                      slice(x0 + u[2] * k, x1 + u[2] * k))
                vk = vol[ks]
                pathmin = vk if pathmin is None else np.minimum(pathmin, vk)
            dip = np.minimum(vol[s], vol[t]) - pathmin     # >0 = a real intensity gap between the wraps
            blind = cross & (dip <= a.dip_thr)
            b_v = av[blind]
        else:
            blind = np.zeros_like(cross)
            b_v = av[blind]
        entry = dict(
            offset=[int(dz), int(dy), int(dx)],
            n_within=int(within.sum()), n_cross=int(cross.sum()), n_cross_blind=int(blind.sum()),
            within_mean=round(float(w_v.mean()), 4) if w_v.size else None,
            cross_mean=round(float(c_v.mean()), 4) if c_v.size else None,
            separation=round(float(w_v.mean() - c_v.mean()), 4) if (w_v.size and c_v.size) else None,
            auc=round(auc_mannwhitney(w_v, c_v), 4) if (w_v.size and c_v.size) else None,
            auc_blind=round(auc_mannwhitney(w_v, b_v), 4) if (w_v.size and b_v.size) else None,
        )
        res["channels"].append(entry)
        print(f"off {str(entry['offset']):>12}  within {entry['within_mean']}  cross {entry['cross_mean']}  "
              f"sep {entry['separation']}  AUC {entry['auc']}  AUC_zeroevid {entry['auc_blind']}  "
              f"(n {entry['n_within']}/{entry['n_cross']}/{entry['n_cross_blind']})", flush=True)
        if w_v.size:
            all_within.append(w_v[:: max(1, w_v.size // 400_000)])
        if c_v.size:
            all_cross.append(c_v[:: max(1, c_v.size // 400_000)])
        if b_v.size:
            all_cross_blind.append(b_v[:: max(1, b_v.size // 400_000)])
    W = np.concatenate(all_within) if all_within else np.array([])
    C = np.concatenate(all_cross) if all_cross else np.array([])
    B = np.concatenate(all_cross_blind) if all_cross_blind else np.array([])
    res["overall"] = dict(
        within_mean=round(float(W.mean()), 4) if W.size else None,
        cross_mean=round(float(C.mean()), 4) if C.size else None,
        separation=round(float(W.mean() - C.mean()), 4) if (W.size and C.size) else None,
        auc=round(auc_mannwhitney(W, C), 4) if (W.size and C.size) else None,
        auc_zeroevidence=round(auc_mannwhitney(W, B), 4) if (W.size and B.size) else None,
    )
    o = res["overall"]
    print(f"\nOVERALL within {o['within_mean']} vs cross {o['cross_mean']} | separation {o['separation']} | "
          f"AUC {o['auc']} | AUC_zeroevidence {o['auc_zeroevidence']}")
    # PER-CHANNEL GATE. Pooling AUC over all channels is misleading: for long y/x offsets the target is ~always 0
    # (a sheet is only within-wrap tangentially over 9-27 vox), so those channels are trivially separable and drag
    # the pooled number up -- a head that is DEAD at r=1 can pass a pooled gate. The r=1 channels (0,1,2) are the
    # ones aff_carve and GASP's fragment stage depend on, so they are gated hard and separately.
    unit = [c for c in res["channels"] if max(abs(v) for v in c["offset"]) == 1]
    unit_auc = [c["auc"] for c in unit if c["auc"] is not None]
    long_auc = [c["auc"] for c in res["channels"] if c["auc"] is not None
                and max(abs(v) for v in c["offset"]) > 1]
    res["gate"] = dict(unit_auc_min=round(min(unit_auc), 4) if unit_auc else None,
                       unit_auc_mean=round(float(np.mean(unit_auc)), 4) if unit_auc else None,
                       long_auc_mean=round(float(np.mean(long_auc)), 4) if long_auc else None)
    print(f"GATE r=1 channels: min AUC {res['gate']['unit_auc_min']} mean {res['gate']['unit_auc_mean']} | "
          f"long-range mean AUC {res['gate']['long_auc_mean']}")
    # distribution percentiles -> pick aff_carve_thr from DATA instead of the hardcoded 0.3
    if W.size and C.size:
        res["percentiles"] = dict(
            within=[round(float(np.percentile(W, q)), 4) for q in (5, 25, 50, 75, 95)],
            cross=[round(float(np.percentile(C, q)), 4) for q in (5, 25, 50, 75, 95)])
        print(f"within p5/25/50/75/95 {res['percentiles']['within']}")
        print(f"cross  p5/25/50/75/95 {res['percentiles']['cross']}")
    ok = bool(unit_auc) and min(unit_auc) > 0.6 and (o["auc"] is not None and o["auc"] > 0.6)
    print("AFFINITY_USABLE" if ok else
          "AFFINITY_DEAD -- no decode can separate wraps from these affinities (check the r=1 channels first)")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print("DIAG_AFFINITY_DONE")


if __name__ == "__main__":
    main()
