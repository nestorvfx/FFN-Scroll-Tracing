#!/usr/bin/env python3
"""Quantify the texture gap between the real and synth panels of a blind round.

The independent judge separated v29 perfectly (0/12 = inverted 12/12) on ONE axis: synth panels read
crisper/finer-grained with bright specks; real 138px crops read softer/blobbier. This measures that axis
numerically on the exact panels of a round (reproduced deterministically from the seed), so the PSF/grain
parameters can be tuned against a target band instead of eyeballs.

Usage: python measure_blind.py --cubes /root/data/cubes --tif /root/data/10192.tif --seed 61 \
          [--key /root/blind_key9.json] [--n 6]
"""
import os, sys, json, argparse
import numpy as np
import tifffile
import multiprocessing as mp
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_blind import real_crops, ref_windows, one


def panel_stats(p):
    x = p.astype(np.float32)
    lap = ndi.laplace(ndi.gaussian_filter(x, 0.5))
    gx = ndi.sobel(x, 0); gy = ndi.sobel(x, 1)
    grad = float(np.mean(np.hypot(gx, gy)))
    # attribution: where does fine-scale energy live? bright-phase interior / dark interior / boundary band
    from skimage.filters import threshold_otsu
    thr = float(threshold_otsu(x))
    bm = x >= thr
    bi_ = ndi.binary_erosion(bm, iterations=3); di = ndi.binary_erosion(~bm, iterations=3)
    bb = ~(bi_ | di)
    l2 = lap ** 2
    lv_pap = float(l2[bi_].mean()) if bi_.sum() > 50 else 0.0
    lv_air = float(l2[di].mean()) if di.sum() > 50 else 0.0
    lv_bnd = float(l2[bb].mean()) if bb.sum() > 50 else 0.0
    # radial FFT power split: DC-free fraction above 0.30 Nyquist and in 0.15-0.30 (structure vs grain bands)
    F = np.abs(np.fft.fftshift(np.fft.fft2(x - x.mean()))) ** 2
    h, w = x.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - h / 2, xx - w / 2) / (min(h, w) / 2)   # 0..~1 of Nyquist
    tot = float(F[r > 0.02].sum()) + 1e-6
    hf = float(F[r > 0.30].sum()) / tot
    mf = float(F[(r > 0.15) & (r <= 0.30)].sum()) / tot
    lc = float((x - ndi.gaussian_filter(x, 3.0)).std())
    return dict(lapvar=float(lap.var()), lv_pap=lv_pap, lv_air=lv_air, lv_bnd=lv_bnd, grad=grad,
                hf=hf, mf=mf, lc=lc,
                b215=float((p >= 215).mean()), b230=float((p >= 230).mean()),
                mean=float(x.mean()), std=float(x.std()))


def band(vals):
    return f"{min(vals):.4g}..{max(vals):.4g} (med {np.median(vals):.4g})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubes", required=True); ap.add_argument("--tif", required=True)
    ap.add_argument("--seed", type=int, default=61); ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--key", default=None, help="verify reproduced shuffle against this hidden-key JSON")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    img = tifffile.imread(a.tif).astype(np.float32)
    lo, hi = np.percentile(img, [1, 99.5])
    img = np.clip((img - lo) / (hi - lo + 1e-6) * 255, 0, 255)

    reals = [r.astype(np.uint8) for r in real_crops(img, a.n, rng)]
    refs = ref_windows(img, a.n, np.random.default_rng(a.seed + 5000))
    with mp.Pool(a.n) as p:
        synths = p.map(one, [(a.cubes, 200 + i, refs[i % len(refs)]) for i in range(a.n)])

    if a.key:
        panels = [(r, "real") for r in reals] + [(s, "synth") for s in synths]
        order = rng.permutation(len(panels))
        mykey = {str(i + 1): panels[oi][1] for i, oi in enumerate(order)}
        truth = json.load(open(a.key))
        print(f"key reproduction vs {a.key}: {'MATCH' if mykey == truth else 'MISMATCH ' + json.dumps(mykey)}")

    rs = [panel_stats(r) for r in reals]
    ss = [panel_stats(s) for s in synths]
    # 3-PLANE consistency (audit issue 3: everything was certified on ct[:,mid,:] only; the network trains on
    # all planes). Compose once more and compare the three orthogonal mid-planes' fine-scale energy: a plane
    # that sticks out means the 3D statistics are anisotropic in a way the certified plane never showed.
    from sheet_compose import compose as _compose
    ct3, _, _ = _compose(a.cubes, 777, hist_ref=refs[0])
    for ax, nm in ((0, "z-plane"), (1, "y-plane"), (2, "x-plane")):
        cs3 = np.take(ct3, ct3.shape[ax] // 2, axis=ax).astype(np.float32)
        lv3 = float(ndi.laplace(ndi.gaussian_filter(cs3, 0.5)).var())
        print(f"  plane-check {nm}: lapvar={lv3:.0f} std={cs3.std():.1f} mean={cs3.mean():.1f}")
    keys = ["lapvar", "lv_pap", "lv_air", "lv_bnd", "grad", "hf", "mf", "lc", "b215", "b230", "mean", "std"]
    print(f"{'stat':8s}  {'REAL band':34s}  {'SYNTH band':34s}  gap")
    for k in keys:
        rv = [d[k] for d in rs]; sv = [d[k] for d in ss]
        gap = ""
        if min(sv) > max(rv): gap = "SYNTH ABOVE"
        elif max(sv) < min(rv): gap = "SYNTH BELOW"
        print(f"{k:8s}  {band(rv):34s}  {band(sv):34s}  {gap}")
    print("\nper-panel (synth):")
    for i, d in enumerate(ss):
        print(f"  s{i}: " + " ".join(f"{k}={d[k]:.4g}" for k in keys))
    print("per-panel (real):")
    for i, d in enumerate(rs):
        print(f"  r{i}: " + " ".join(f"{k}={d[k]:.4g}" for k in keys))


if __name__ == "__main__":
    main()
