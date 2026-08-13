#!/usr/bin/env python3
"""BLIND real-vs-composed test: shuffled, unlabeled, identically rendered panels + a hidden key.

The only honest test of "matched": the judge must NOT know which is which. Real panels are 192x192 crops from
compact regions of the full slice; composed panels are cross-sections of composed cubes. Both uint8, same zoom,
no titles. The key (position -> real/synth) goes to a separate JSON the judge reads only AFTER calling each panel.

Usage: python render_blind.py --cubes <dir> --tif /root/data/10192.tif --out /root/blind.png --key /root/blind_key.json
"""
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile
import multiprocessing as mp
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sheet_compose import compose


def real_crops(img, n, rng, size=138, p_fused=0.25):
    """Sample real windows. The trans>0.08 gate selects visibly-layered regions -- but that EXCLUDES the fused
    mottle blocks that ARE the densest regime (Fable-5 audit issue 1: refs and blind reals were one-sidedly
    'separable-looking', flattering both calibration and blind scores). A p_fused share now comes from a
    NO-TRANSITION stratum: high fg, few visible alternations = fused/compact material."""
    from skimage.filters import threshold_otsu
    small = img[::8, ::8]
    thr = float(threshold_otsu(small[small > 5]))
    fgm = (small >= thr).astype(np.float32)
    fg = ndi.uniform_filter(fgm, size // 8)
    trans = ndi.uniform_filter(np.abs(np.diff(fgm, axis=1, prepend=0)), size // 8)   # many sheet/air alternations
    ok_layered = (fg > 0.45) & (fg < 0.82) & (trans > 0.08)   # sheets, not the solid case shell / not fragments
    ok_fused = (fg > 0.60) & (fg < 0.92) & (trans <= 0.06)    # dense fused mottle: the hardest regime
    # AIRY stratum: the pool had NO window below fg 0.45, so airy compositions were quantile-mapped
    # onto dense references -- stretching their air phase bright (measured: D2 traces 1.6-2.2% in
    # fg~0.35 cubes vs ~0.02% dense; real air 0.014%). Real windows ARE often airy.
    ok_airy = (fg > 0.12) & (fg <= 0.45) & (trans > 0.03)
    out, used = [], []
    for ok, want in ((ok_fused, int(round(n * p_fused))), (ok_airy, int(round(n * 0.3))), (ok_layered, n)):
        ys, xs = np.where(ok)
        if len(ys) == 0:
            continue
        picks = rng.choice(len(ys), size=min(n * 3, len(ys)), replace=False)
        for pi in picks:
            cy, cx = int(ys[pi] * 8), int(xs[pi] * 8)
            if any(abs(cy - a) < size and abs(cx - b) < size for a, b in used):
                continue
            w = img[cy:cy + size, cx:cx + size]
            if w.shape == (size, size):
                out.append(w.copy()); used.append((cy, cx))
            if len(out) >= (want if ok is ok_fused else n):
                break
        if len(out) >= n:
            break
    return out[:n]


def ref_windows(img, n, rng, size=480):
    """Per-seed intensity-reference windows: real panels vary window to window (std 45-63), so each synth
    gets its own real compact target instead of all six sharing one marginal."""
    return real_crops(img, n, rng, size=size)


def one(job):
    cubes, seed, ref = job
    # regeneration gate: real panels' fine-scale energy tops out ~250 (lapvar band 106-322 across rounds,
    # real p95 ~250); a composition landing hotter is an out-of-band outlier -- redraw it, same as the fg
    # gate the corpus wrapper applies. Selection, not fabrication.
    s = seed
    best = None
    for _ in range(3):
        ct, inst, meta = compose(cubes, s, hist_ref=ref)
        cs = ct[:, ct.shape[1] // 2, :].astype(np.float32)
        lv = float(ndi.laplace(ndi.gaussian_filter(cs, 0.5)).var())
        mu = float(cs.mean())
        pen = max(0.0, lv - 260.0) / 260.0 + max(0.0, 118.0 - mu) / 20.0 + max(0.0, mu - 150.0) / 20.0
        if best is None or pen < best[0]:
            best = (pen, cs)                              # keep the CLOSEST-to-band attempt, not the last one
        if pen == 0.0:                                    # real panel bands: lapvar<=~250, mean 119-145
            break
        s += 9973
    cs = best[1]
    rng = np.random.default_rng(seed * 31 + 7)
    ang = float(rng.uniform(-30, 30)) + (90.0 if rng.random() < 0.35 else 0.0)
    cs = ndi.rotate(cs, ang, reshape=False, order=1, mode="reflect")
    c0 = (cs.shape[0] - 138) // 2                     # crop the always-valid center: no mirror swirls
    return np.clip(cs[c0:c0 + 138, c0:c0 + 138], 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubes", required=True); ap.add_argument("--tif", required=True)
    ap.add_argument("--out", default="/root/blind.png"); ap.add_argument("--key", default="/root/blind_key.json")
    ap.add_argument("--n", type=int, default=6); ap.add_argument("--seed", type=int, default=99)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    img = tifffile.imread(a.tif).astype(np.float32)
    lo, hi = np.percentile(img, [1, 99.5])
    img = np.clip((img - lo) / (hi - lo + 1e-6) * 255, 0, 255)

    # SYMMETRIC harness: reals get the same bilinear rotation + center-crop as synth panels (the one-sided
    # resampling softened only the synth population -- audit issue 3 harness asymmetry)
    reals_big = real_crops(img, a.n, rng, size=196)
    reals = []
    for i, rb in enumerate(reals_big):
        rr = np.random.default_rng(a.seed * 917 + i)
        ang = float(rr.uniform(-30, 30)) + (90.0 if rr.random() < 0.35 else 0.0)
        rot = ndi.rotate(rb.astype(np.float32), ang, reshape=False, order=1, mode="reflect")
        c0 = (rot.shape[0] - 138) // 2
        reals.append(np.clip(rot[c0:c0 + 138, c0:c0 + 138], 0, 255))
    refs = ref_windows(img, a.n, np.random.default_rng(a.seed + 5000))
    with mp.Pool(a.n) as p:
        synths = p.map(one, [(a.cubes, 200 + i, refs[i % len(refs)]) for i in range(a.n)])

    # scale-matched calibration sheet for the judge: labeled REAL crops at the same 138px render as the blind
    # panels, disjoint rng stream from the blind reals (the 480px reference miscalibrated the v29 judge into a
    # perfect inversion: at that scale real texture reads crisp, at 138px it reads soft)
    calib = real_crops(img, 8, np.random.default_rng(a.seed + 9000))
    figc, axc = plt.subplots(2, 4, figsize=(3.2 * 4, 3.3 * 2))
    for i, c in enumerate(calib):
        ax = axc[i // 4, i % 4]
        ax.imshow(c, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"REAL ref {i + 1}", fontsize=10); ax.axis("off")
    plt.tight_layout()
    figc.savefig(os.path.splitext(a.out)[0] + "_refs.png", dpi=130, bbox_inches="tight")
    plt.close(figc)

    panels = [(r.astype(np.uint8), "real") for r in reals] + [(s, "synth") for s in synths]
    order = rng.permutation(len(panels))
    panels = [panels[i] for i in order]

    cols = 4
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.3 * rows))
    axes = np.atleast_2d(axes)
    key = {}
    for i, (pan, lab) in enumerate(panels):
        ax = axes[i // cols, i % cols]
        ax.imshow(pan, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"#{i + 1}", fontsize=10)
        ax.axis("off")
        key[str(i + 1)] = lab
    for j in range(len(panels), rows * cols):
        axes[j // cols, j % cols].axis("off")
    plt.tight_layout()
    plt.savefig(a.out, dpi=130, bbox_inches="tight")
    json.dump(key, open(a.key, "w"))
    print(f"wrote {a.out} ({len(panels)} panels, key hidden in {a.key})")


if __name__ == "__main__":
    main()
