#!/usr/bin/env python3
"""Validate CC-DERIVED partial instance labels against REAL wrap truth before trusting them for training.

The proposal (deep-audit #1): real cubes have only BINARY fiber labels, so they currently contribute ZERO
wrap-identity supervision (inst=0 -> all-invalid affinity GT). But connected components of the binary label are
PARTIAL instance labels: cross-component edges are "different wrap" negatives, within-component edges are "same
wrap" positives -- IF two caveats are handled:
  (a) two wraps FUSED into one component would make within-component "same" labels WRONG exactly at the decisive
      contacts (teaching merges). Mitigation: remove locally-THICK voxels (fused blobs are ~2x sheet thickness)
      before labeling, so components split at fused bridges and the ambiguous voxels become ignore (label 0).
  (b) one wrap clipped into two components would make cross-component "different" labels wrong. Expected rare in
      a 192-scale window (a winding's circumference >> window), and a false "different" only buys a cheap split.

This script MEASURES both failure rates against ground truth instead of arguing about them. It runs the exact
derivation on the slab's binary (truth>0), then scores the derived labels against the true wrap ids, per offset
of the training offset set and per DT threshold:
    same_purity  = P(truth-same  | cc-same)     <- caveat (a): must be ~1.0 or the labels teach merges
    diff_purity  = P(truth-diff  | cc-diff)     <- caveat (b): false splits are cheap but should be known
    coverage     = labeled fraction of all fiber-fiber edges
    cross_recall = of TRUE cross-wrap edges, fraction labeled "different"  <- the supervision value
    toxic        = of TRUE cross-wrap edges, fraction labeled "SAME"       <- the disqualifier if high

Usage: python cc_validate.py --slab /root/data/slab [--thr 3.0,3.5,4.0,5.0] [--out cc_validate.json]
"""
import os, sys, json, argparse
import numpy as np
from scipy import ndimage as ndi

OFFSETS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 4, 0], [0, 0, 4], [0, 3, 3], [0, 8, 0], [0, 0, 8],
           [0, 6, 6], [0, 6, -6], [0, 16, 0], [0, 11, 11], [0, 11, -11], [0, 24, 0]]


def derive_cc(binary, dt_thr):
    """The exact derivation training would use: drop thick voxels, 26-connectivity CC on what remains."""
    dt = ndi.distance_transform_edt(binary)
    thin = binary & (dt <= dt_thr)
    cc, n = ndi.label(thin, structure=np.ones((3, 3, 3), bool))
    return cc.astype(np.int32), n, float(thin.sum()) / max(1, int(binary.sum()))


def pair_views(vol, off):
    dz, dy, dx = off
    Z, Y, X = vol.shape
    z0, z1 = max(0, -dz), Z - max(0, dz)
    y0, y1 = max(0, -dy), Y - max(0, dy)
    x0, x1 = max(0, -dx), X - max(0, dx)
    a = vol[z0:z1, y0:y1, x0:x1]
    b = vol[z0 + dz:z1 + dz, y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slab", required=True)
    ap.add_argument("--thr", default="3.0,3.5,4.0,5.0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import nrrd
    truth = nrrd.read(os.path.join(a.slab, "truth.nrrd"))[0].astype(np.int32)
    binary = truth > 0
    print(f"slab {truth.shape}, fiber {binary.mean():.3f}, wraps {truth.max()}", flush=True)
    results = {}
    for thr in [float(x) for x in a.thr.split(",")]:
        cc, n_cc, keep_frac = derive_cc(binary, thr)
        rows = []
        for off in OFFSETS:
            ca, cb = pair_views(cc, off)
            ta, tb = pair_views(truth, off)
            fib = (ta > 0) & (tb > 0)                      # all fiber-fiber edges at this offset
            lab = (ca > 0) & (cb > 0)                      # edges the derivation supervises
            n_fib = int(fib.sum())
            if not n_fib:
                continue
            cc_same = lab & (ca == cb)
            cc_diff = lab & (ca != cb)
            tr_same = ta == tb
            n_ccs, n_ccd = int(cc_same.sum()), int(cc_diff.sum())
            same_pur = float((cc_same & tr_same).sum()) / max(1, n_ccs)
            diff_pur = float((cc_diff & ~tr_same).sum()) / max(1, n_ccd)
            cross = fib & ~tr_same                         # TRUE cross-wrap edges (the decisive ones)
            n_cross = int(cross.sum())
            cross_rec = float((cross & cc_diff).sum()) / max(1, n_cross)
            toxic = float((cross & cc_same).sum()) / max(1, n_cross)
            rows.append(dict(off=off, n_fib=n_fib, coverage=round((n_ccs + n_ccd) / n_fib, 4),
                             same_purity=round(same_pur, 5), diff_purity=round(diff_pur, 5),
                             n_cross=n_cross, cross_recall=round(cross_rec, 4), toxic=round(toxic, 5)))
        w = np.array([r["n_fib"] for r in rows], np.float64)
        agg = {k: round(float(np.average([r[k] for r in rows], weights=w)), 5)
               for k in ("coverage", "same_purity", "diff_purity", "cross_recall", "toxic")}
        results[thr] = dict(n_components=n_cc, thin_keep_frac=round(keep_frac, 4), pooled=agg, per_offset=rows)
        print(f"\nTHR={thr}: components={n_cc}  kept(thin) {keep_frac:.1%} of fiber")
        print(f"  pooled: coverage {agg['coverage']:.1%}  same_purity {agg['same_purity']:.4f}  "
              f"diff_purity {agg['diff_purity']:.4f}  cross_recall {agg['cross_recall']:.1%}  "
              f"TOXIC {agg['toxic']:.4%}")
        worst = max(rows, key=lambda r: r["toxic"])
        print(f"  worst-toxic offset {worst['off']}: toxic {worst['toxic']:.4%} "
              f"(cross_recall {worst['cross_recall']:.1%}, n_cross {worst['n_cross']})", flush=True)
    print("\nDECISION RULE: usable if same_purity ~1 AND toxic << the cross-wrap base error a model would make "
          "anyway; cross_recall is the payoff (real 'different' supervision at real contacts).")
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1, default=float)
    print("CC_VALIDATE_DONE")


if __name__ == "__main__":
    main()
