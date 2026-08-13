#!/usr/bin/env python
"""Vectorized defect gates (D1 foreign-in-gap / D2 traces / D3 thin columns).

The loop versions cost minutes per cycle (per-column Python over millions of runs); this is the
same measurement fully vectorized: run-length extraction via a single diff over column-padded
flattened masks, gap/foreign/thin statistics via bincounts and per-column cumsums. ~10 s per batch.

Usage: fastgates.py <labels_glob> [--ax N|-1(heuristic)] [--d2]
"""
import argparse
import glob

import numpy as np
import tifffile
from scipy import ndimage as ndi


def runs_2d(mf):
    """(col, start, end) of True-runs for every row of a 2-D bool array, fully vectorized."""
    C, N = mf.shape
    pad = np.zeros((C, N + 2), np.int8)
    pad[:, 1:-1] = mf
    d = np.diff(pad.reshape(-1))
    st = np.flatnonzero(d == 1)
    en = np.flatnonzero(d == -1)
    col = st // (N + 2)
    s = st - col * (N + 2)
    e = en - col * (N + 2)          # start/end in padded coords; run = [s, e) in real coords [s, e)
    return col, s, e


def gate_cube(inst, ax_forced):
    tg = fo = th = tc = 0
    for iid in np.unique(inst[inst > 0]):
        sel = inst == iid
        ax = ax_forced if ax_forced is not None else \
            int(np.argmax([np.abs(np.diff(sel.astype(np.int8), axis=a)).sum() for a in range(3)]))
        m = np.moveaxis(sel, ax, -1)
        o = np.moveaxis((inst > 0) & ~sel, ax, -1)
        N = m.shape[-1]
        mf = m.reshape(-1, N)
        of = o.reshape(-1, N)
        occ = mf.any(1)
        if occ.sum() < 100:
            continue
        col, s, e = runs_2d(mf)
        # per-column totals
        width = np.bincount(col, weights=(e - s).astype(np.float64), minlength=mf.shape[0])
        ncols = int(occ.sum())
        tc += ncols
        th += int(((width > 0) & (width < 2)).sum())
        # gaps between consecutive runs of the SAME column
        same = col[1:] == col[:-1]
        g = s[1:] - e[:-1]
        ok = same & (g >= 1) & (g <= 12)
        tg += int(ok.sum())
        if ok.any():
            cs = np.cumsum(of, axis=1)
            c2 = col[1:][ok]
            a2 = e[:-1][ok]          # gap = [a2, b2)
            b2 = s[1:][ok]
            fsum = cs[c2, b2 - 1] - np.where(a2 > 0, cs[c2, a2 - 1], 0)
            fo += int((fsum > 0).sum())
    return fo, tg, th, tc


def cover_cube(inst, ct):
    """Label coverage: fraction of Otsu-foreground carrying NO instance label, stratified by
    distance to the nearest label. The single most important label-quality statistic for FFN
    training -- D1/D2/D3 structurally cannot see it (advisor 4.1). Real GP GT ~24%, old synth
    ~22.5%; the gate15 regression measured 41-55%."""
    from skimage.filters import threshold_otsu
    x = ct.astype(np.float32)
    fg = x >= threshold_otsu(x)
    unl = fg & (inst == 0)
    d = ndi.distance_transform_edt(inst == 0)
    tot = max(int(fg.sum()), 1)
    return dict(unl=float(unl.sum()) / tot,
                near=float((unl & (d <= 1)).sum()) / tot,
                mid=float((unl & (d > 1) & (d <= 3)).sum()) / tot,
                far=float((unl & (d > 3)).sum()) / tot)


def straight_cube(ct, ax=2):
    """Longest boundary transition ALONG the stacking axis held perfectly straight ACROSS an
    in-plane axis (1-voxel precision): a sheet face at constant depth across tens of voxels is
    physically impossible. Sheet edges running along the scroll axis are naturally straight and
    are NOT counted (that channel is confounded -- advisor 1.2)."""
    from skimage.filters import threshold_otsu
    x = np.moveaxis(ct.astype(np.float32), ax, -1)
    thr = 0.62 * float(threshold_otsu(x))
    m = x >= thr
    best = 0
    for a in range(2):
        S = np.moveaxis(m, a, 0)[m.shape[a] // 2]        # (other in-plane axis, stacking)
        B = (S[:, :-1] != S[:, 1:])                      # transitions along stacking
        col, s, e = runs_2d(np.ascontiguousarray(B.T))   # runs across the in-plane axis
        if s.size:
            best = max(best, int((e - s).max()))
    return best


def d2_cube(inst, ct):
    v = ct[ct > 0]
    thr = np.percentile(v, 60)
    far = ndi.distance_transform_edt(inst == 0) > 2.5
    return float(((ct > thr) & (inst == 0) & far).sum()) / max((ct > thr).sum(), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels_glob")
    ap.add_argument("--ax", type=int, default=2)      # -1 = per-instance heuristic (for real crops)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--d2", action="store_true")
    ap.add_argument("--cover", action="store_true")
    ap.add_argument("--straight", action="store_true")
    a = ap.parse_args()
    axf = None if a.ax < 0 else a.ax
    FO = TG = TH = TC = 0
    d2s, covs, strs = [], [], []
    for p in sorted(glob.glob(a.labels_glob))[:a.n]:
        inst = tifffile.imread(p).astype(np.int32)
        fo, tg, th, tc = gate_cube(inst, axf)
        FO += fo; TG += tg; TH += th; TC += tc
        if a.d2 or a.cover or a.straight:
            ct = tifffile.imread(p.replace("labelsTr_inst", "imagesTr")
                                 .replace(".tif", "_0000.tif")).astype(np.float32)
            if a.d2:
                d2s.append(d2_cube(inst, ct))
            if a.cover:
                covs.append(cover_cube(inst, ct))
            if a.straight:
                strs.append(straight_cube(ct))
    print("D1 foreign-in-gap: %d/%d (%.2f%%)   D3 thin-cols: %d/%d (%.2f%%)"
          % (FO, TG, 100.0 * FO / max(TG, 1), TH, TC, 100.0 * TH / max(TC, 1)))
    if d2s:
        print("D2 traces mean: %.3f%%  per-cube: %s"
              % (100 * float(np.mean(d2s)), [round(100 * x, 3) for x in d2s]))
    if covs:
        print("COVER unlabelled-fg mean: %.1f%% (near %.1f / mid %.1f / far %.1f)  per-cube: %s"
              % (100 * float(np.mean([c["unl"] for c in covs])),
                 100 * float(np.mean([c["near"] for c in covs])),
                 100 * float(np.mean([c["mid"] for c in covs])),
                 100 * float(np.mean([c["far"] for c in covs])),
                 [round(100 * c["unl"], 1) for c in covs]))
    if strs:
        print("STRAIGHT longest-run: max %d  per-cube: %s" % (max(strs), strs))


if __name__ == "__main__":
    main()
