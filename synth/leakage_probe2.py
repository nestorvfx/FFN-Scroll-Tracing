#!/usr/bin/env python3
"""Composer-specific leakage probe (advisor 4.2): the old leakage_probe.py exercises the RETIRED
v1 mechanics (synth_merge gap_collapse/crease_warp), so its gate was vacuous for sheet_compose.

Question: can a texture-only classifier tell composer-specific zones apart from native ones inside
the SAME saved corpus cubes? Classes probed:
  contact : papyrus within 2 vox of an inter-instance boundary (fusion/core-swap zone)
            vs papyrus >= 8 vox from any boundary (native interior).
  air     : air 2-6 vox from papyrus (donor pad-air envelope) vs air > 10 vox (base-air field).

Descriptors are geometry-suppressed (high-pass residual; robust MAD + normalized high-freq radial
spectrum shape), same design as the v1 probe. The honest gate is RELATIVE: run the same probe on
REAL labeled cubes and compare AUCs -- near-sheet air legitimately differs from far air in real CT
too (PV ramp), so absolute AUC alone over-flags.

Usage:
  leakage_probe2.py --corpus <dir with imagesTr/ labelsTr_inst/> [--n 10 --per_cube 60 --half 10]
  add --real <dir> to also run the real baseline (same layout).
"""
import argparse
import glob
import os

import numpy as np
import tifffile
from scipy import ndimage as ndi


def radial_nps(p):
    F = np.abs(np.fft.fftn(p - p.mean())) ** 2 / p.size
    fr = [np.fft.fftfreq(s) for s in p.shape]
    g = np.meshgrid(*fr, indexing="ij")
    r = np.sqrt(sum(x * x for x in g))
    nb = 12
    bi = np.minimum((r * 2 * nb).astype(int), nb - 1)
    nps = (np.bincount(bi.ravel(), F.ravel(), minlength=nb)
           / np.maximum(np.bincount(bi.ravel(), minlength=nb), 1))
    return None, nps


def sample_centers(mask, n, half, shape, rng):
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return pts
    ok = np.all((pts >= half) & (pts < (np.array(shape) - half)), axis=1)
    pts = pts[ok]
    if len(pts) == 0:
        return pts
    sel = rng.choice(len(pts), size=min(n, len(pts)), replace=False)
    return pts[sel]


def patch_feats(hf, centers, half):
    feats = []
    for z, y, x in centers:
        p = hf[z - half:z + half, y - half:y + half, x - half:x + half]
        _, nps = radial_nps(p)
        hi = nps[len(nps) // 3:]
        shape = hi / (np.nansum(hi) + 1e-8)
        mad = float(np.median(np.abs(p - np.median(p)))) * 1.4826
        feats.append(np.concatenate([[mad], np.nan_to_num(shape)]))
    return np.asarray(feats, np.float32)


def auc(scores, labels):
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    alls = np.concatenate([pos, neg])
    ranks = np.empty(len(alls))
    ranks[np.argsort(alls, kind="mergesort")] = np.arange(1, len(alls) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fit_logreg(X, y, iters=400, lr=0.2, l2=1e-2):
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -30, 30)))
        g = p - y
        w -= lr * (X.T @ g / len(y) + l2 * w)
        b -= lr * float(g.mean())
    return w, b


def cv_auc(X, y, seed, k=5):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, k)
    aucs = []
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        w, b = fit_logreg((X[tr] - mu) / sd, y[tr])
        aucs.append(auc(((X[te] - mu) / sd) @ w + b, y[te]))
    return float(np.mean(aucs)), float(np.std(aucs))


def masks_for(inst, ct):
    fg = inst > 0
    big = np.where(fg, inst, np.iinfo(np.int32).max)
    mn = ndi.minimum_filter(big, size=3)
    mx = ndi.maximum_filter(np.where(fg, inst, 0), size=3)
    boundary = fg & (mn < np.iinfo(np.int32).max) & (mx != mn)
    d_b = ndi.distance_transform_edt(~boundary)
    d_fg = ndi.distance_transform_edt(~fg)
    # v2.1: BOTH contact classes are through-material (no air within 2.5 vox). Open-gap faces are
    # genuine edges whose PREVALENCE differs between synth (calibrated patch statistics) and real
    # (annotation adjacency) -- with them in the class, AUC tracks composition, not texture
    # (measured: four disjoint composer knobs all moved pooled AUC to ~0.85 identically).
    d_air = ndi.distance_transform_edt(fg)
    pos_c = fg & (d_b <= 2) & (d_air > 2.5)
    neg_c = fg & (d_b >= 8) & (d_air > 2.5)
    pos_a = (~fg) & (d_fg >= 2) & (d_fg <= 6)
    neg_a = (~fg) & (d_fg > 10)
    return dict(contact=(pos_c, neg_c), air=(pos_a, neg_a))


def probe_dir(d, n_cubes, per_cube, half, seed, label_sub="labelsTr_inst", prefix=""):
    rng = np.random.default_rng(seed)
    X = {k: [] for k in ("contact", "air")}
    Y = {k: [] for k in ("contact", "air")}
    used = 0
    files = [f for f in sorted(glob.glob(os.path.join(d, label_sub, "*.tif")))
             if os.path.basename(f).startswith(prefix)]
    rng.shuffle(files)
    for p in files[: n_cubes * 3]:
        if used >= n_cubes:
            break
        inst = tifffile.imread(p).astype(np.int32)
        cp = p.replace(label_sub, "imagesTr").replace(".tif", "_0000.tif")
        if not os.path.exists(cp):
            continue
        ct = tifffile.imread(cp).astype(np.float32)
        hf = ct - ndi.gaussian_filter(ct, 0.8)
        got = False
        for k, (pm, nm) in masks_for(inst, ct).items():
            pc = sample_centers(pm, per_cube, half, ct.shape, rng)
            nc = sample_centers(nm, per_cube, half, ct.shape, rng)
            if len(pc) < 8 or len(nc) < 8:
                continue
            X[k].append(patch_feats(hf, pc, half))
            Y[k].append(np.ones(len(pc)))
            X[k].append(patch_feats(hf, nc, half))
            Y[k].append(np.zeros(len(nc)))
            got = True
        used += int(got)
    out = {}
    for k in X:
        if X[k]:
            xx, yy = np.concatenate(X[k]), np.concatenate(Y[k])
            out[k] = cv_auc(xx, yy, seed) + (len(yy),)
    return out, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--real")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--per_cube", type=int, default=60)
    ap.add_argument("--half", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real_labels", default="labelsTr_inst")
    ap.add_argument("--real_prefix", default="real_")
    a = ap.parse_args()
    syn, us = probe_dir(a.corpus, a.n, a.per_cube, a.half, a.seed)
    for k, (m, s, n) in syn.items():
        print(f"SYNTH {k:8s} AUC {m:.3f} +/- {s:.3f}  ({us} cubes, {n} patches)")
    if a.real:
        rl, ur = probe_dir(a.real, a.n, a.per_cube, a.half, a.seed, a.real_labels, a.real_prefix)
        for k, (m, s, n) in rl.items():
            print(f"REAL  {k:8s} AUC {m:.3f} +/- {s:.3f}  ({ur} cubes, {n} patches)")
        for k in syn:
            if k in rl:
                d = syn[k][0] - rl[k][0]
                tag = "PASS" if d <= 0.05 else ("OK" if d <= 0.10 else "TELL")
                print(f"GATE  {k:8s} synth-real ΔAUC {d:+.3f}   {tag}")


if __name__ == "__main__":
    main()
