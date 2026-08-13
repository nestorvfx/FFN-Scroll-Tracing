#!/usr/bin/env python3
"""Carve-width ablation: for a corpus of instance tifs, recompute BOTH carve modes (symmetric 2-vox vs
single 1-vox seam) from the SAME instances and report, per mode: post-carve sheet thickness (p50) and the
LABEL-BRIDGE count (CC spanning >=2 wraps, >40 vox) -- the one metric that must stay 0 (unbounded merge cost).
Answers the user's carve-thickness question: how thin can the seam get before merges appear."""
import os, sys, glob
import numpy as np
from scipy import ndimage as ndi
import tifffile

MAXI = np.iinfo(np.int32).max


def carve(inst, mode):
    fg = inst > 0
    big = np.where(fg, inst, MAXI)
    mn = ndi.minimum_filter(big, size=3)
    mx = ndi.maximum_filter(np.where(fg, inst, 0), size=3)
    boundary = fg & (mn < MAXI) & (mx != mn)
    c = boundary & (inst < mx) if mode == "single" else boundary
    return (fg & ~c).astype(np.uint8)


def sheet_p50(inst, lab, min_vox=300):
    """post-carve median sheet thickness over surviving sheets (2*EDT on isolated post-carve mask)."""
    survivor = np.where(lab > 0, inst, 0)
    ids = np.unique(survivor[survivor > 0])
    locs = ndi.find_objects(survivor)
    meds = []
    for i in ids:
        if i - 1 >= len(locs) or locs[i - 1] is None:
            continue
        sl = tuple(slice(max(0, s.start - 1), s.stop + 1) for s in locs[i - 1])
        m = survivor[sl] == i
        if m.sum() < min_vox:
            continue
        edt = ndi.distance_transform_edt(m)
        t = 2.0 * edt[m]
        meds.append(np.median(t[t >= np.percentile(t, 50)]))
    return (float(np.median(meds)) if meds else 0.0), len(meds)


def bridges(inst, lab):
    """CC on the carved binary; count components whose voxels span >=2 distinct wrap ids (>40 vox overlap each)."""
    cc, n = ndi.label(lab > 0)
    nb = 0
    for c in range(1, n + 1):
        m = cc == c
        if m.sum() < 80:
            continue
        wraps = inst[m]
        u, cnt = np.unique(wraps[wraps > 0], return_counts=True)
        if (cnt > 40).sum() >= 2:
            nb += 1
    return nb


def main():
    corp = os.environ.get("CORP", sys.argv[1] if len(sys.argv) > 1 else "/root/val_deep_fix")
    insts = sorted(glob.glob(f"{corp}/labelsTr_inst/*.tif"))[:60]
    for mode in ("symmetric", "single"):
        p50s, nsurv, nbr, ncube_bridge = [], [], 0, 0
        for f in insts:
            inst = tifffile.imread(f).astype(np.int32)
            lab = carve(inst, mode)
            p, ns = sheet_p50(inst, lab)
            b = bridges(inst, lab)
            if p > 0:
                p50s.append(p); nsurv.append(ns)
            nbr += b; ncube_bridge += (b > 0)
        print(f"{mode:10s}: post-carve p50={np.median(p50s):.2f} vox  survivors/cube={np.mean(nsurv):.1f}  "
              f"LABEL-BRIDGES={nbr} in {ncube_bridge}/{len(insts)} cubes")
    print("ABLATION_DONE")


if __name__ == "__main__":
    main()
