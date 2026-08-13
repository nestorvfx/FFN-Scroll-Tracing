#!/usr/bin/env python3
"""GRADED per-contact merge-rate on the slab -- the instrument for ranking merge-averse progress.

CTF is saturated on this dense slab (truth-uncarved already 0.9957), so it can only reward near-perfect carving and
gives no gradient for partial improvement. This measures the thing that moves continuously: of all ADJACENT truth-
wrap pairs, the fraction that share a predicted connected component (a MERGE). Lower = better. Reported per
threshold, since a merge-averse model should trade more splits (free) for fewer merges as the threshold rises.

Usage: python merge_rate.py --slab /root/data/slab --pred fiber.npy --tag mergeaverse --thrs 0.5,0.7,0.85
"""
import os, sys, argparse
import numpy as np
from scipy import ndimage as ndi


def merger_rate(fg, truth, min_shared=40):
    """fg: bool predicted foreground. truth: int wrap labels. Returns (merged, total_pairs, rate, n_instances)."""
    lab, _ = ndi.label(fg, structure=np.ones((3, 3, 3), bool))
    n = int(truth.max()) + 1
    m = (lab > 0) & (truth > 0)
    key = lab[m].astype(np.int64) * n + truth[m].astype(np.int64)
    kk, cc = np.unique(key, return_counts=True)
    cover = {}
    for k, c in zip(kk, cc):
        if c >= min_shared:                                   # this pred instance substantially covers this wrap
            cover.setdefault(int(k // n), set()).add(int(k % n))
    pairs = set()
    for ax in range(3):
        s1 = [slice(None)] * 3; s2 = [slice(None)] * 3
        s1[ax] = slice(0, -1); s2[ax] = slice(1, None)
        a = truth[tuple(s1)]; b = truth[tuple(s2)]
        mm = (a > 0) & (b > 0) & (a != b)
        for u, v in zip(a[mm], b[mm]):
            pairs.add((min(int(u), int(v)), max(int(u), int(v))))
    merged = set()
    for tids in cover.values():
        tl = sorted(tids)
        for i in range(len(tl)):
            for j in range(i + 1, len(tl)):
                if (tl[i], tl[j]) in pairs:                   # one pred instance covers BOTH wraps of an adjacent pair
                    merged.add((tl[i], tl[j]))
    ninst = int(len(np.unique(lab[lab > 0])))
    return len(merged), len(pairs), (len(merged) / max(1, len(pairs))), ninst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slab", required=True); ap.add_argument("--pred", required=True)
    ap.add_argument("--tag", default="pred"); ap.add_argument("--thrs", default="0.5,0.7,0.85")
    a = ap.parse_args()
    import nrrd
    truth = nrrd.read(os.path.join(a.slab, "truth.nrrd"))[0].astype(np.int32)
    prob = np.load(a.pred).astype(np.float32)
    if prob.max() > 1.5:
        prob /= 255.0
    for thr in [float(x) for x in a.thrs.split(",")]:
        merged, total, rate, ninst = merger_rate(prob >= thr, truth)
        print(f"{a.tag} thr {thr}: merge_rate {rate:.4f} ({merged}/{total} pairs merged)  instances {ninst}  "
              f"fg_frac {float((prob >= thr).mean()):.4f}", flush=True)
    print("MERGE_RATE_DONE", a.tag)


if __name__ == "__main__":
    main()
