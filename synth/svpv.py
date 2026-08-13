#!/usr/bin/env python3
"""Rank-5 primitive: SUB-VOXEL PARTIAL-VOLUME (SVPV) — simulation-supervision for the un-resolvable gap.

THE PROBLEM. The failure case is two sheets parted by LESS than one voxel of air. The detector integrates air and
papyrus into a single partial-volume-averaged voxel, so the image shows a dip of a few CT units or nothing at all,
yet the sheets are genuinely two. This is a GT-HARD regime: nobody can hand-label a boundary the instrument never
recorded, so it cannot be learned from annotation — only from simulation where the answer is known by construction.

THE PARADIGM (established, not invented here). "Physics-based machine learning for subcellular segmentation in
living cells", Nature Machine Intelligence 3:1071 (2021) solves exactly this class -- resolution-limited,
noise-afflicted, GT-deficient -- with SIMULATION-SUPERVISION using PHYSICS-BASED GT: define the structure where it
is unambiguous, push it through the instrument's forward model (PSF, blur, noise), and train on the result; the
physics-based GT "resolves the GT-hardness" and lets the network "compensate for the limitations of the
instrument". The same trick is standard in microscopy super-resolution as semi-synthetic pairs (degrade known
high-fidelity GT to synthesise the low-fidelity input, e.g. PSSR). Partial-volume theory (the tissue-fraction
effect; PET/SPECT PVC literature) gives the quantitative model: a voxel's value is the fraction-weighted mix of
the materials inside it, and quantification degrades once a structure is below ~2-3x the system FWHM.

WHY THE EXISTING PIPELINE CANNOT REACH THIS REGIME (measured, not assumed). gap_collapse closes gaps by MOVING
voxels on the native grid via a per-column cumsum warp. Even at keep_air=0 a column reads

    fiber : .#######.#######.#######.        <- a 1-voxel seam SURVIVES full collapse
    labels: .5555555.4444444.3333333.

because the warp is resampled on integer voxels: the collapsed air run still occupies one output sample. So the
thinnest gap the pipeline can express is QUANTIZED to ~1 voxel, and it can never produce a genuinely fused contact
(measured: contacts = 0 at keep_air = 0). The whole sub-voxel band 0 < g < 1 -- the band that actually breaks the
model -- is unreachable by moving voxels. It must be RENDERED.

THE METHOD -- two steps, because the seam the pipeline emits is an OBJECT artifact, not an image.
A genuine g-voxel gap would appear in the image as  papyrus - Delta*(rect_g (*) PSF): a shallow, PSF-smeared dip.
What gap_collapse actually leaves is a HARD, air-valued voxel (it resamples material; it never applies a PSF), i.e.
a sharp rect_t hole that no scanner could ever produce. So we:
    1. FILL the quantized seam back to papyrus  ( + Delta on the t seam voxels ) -> a genuinely fused contact,
       and the unphysical sharp seam is gone;
    2. CARVE the sub-voxel film through the forward model ( - Delta*g*(sheet (*) PSF) ), depositing a total mass of
       Delta*g per unit area spread over the seam's t voxels (amplitude Delta*g/t each).
    out = ct + Delta*thin - PSF (*) [ Delta*g*thin/t ]
The result is, by construction, the image a scanner WOULD have recorded for a g-voxel gap:
peak dip = Delta*g/(sigma*sqrt(2pi)) -- e.g. Delta=150, g=0.3, sigma=2.5 -> ~7 CT units. Verified against an
independently rendered ground-truth scan of a true g-gap (test T1). The real cube is never re-blurred and never
resampled; we add one smooth, physically derived field, so its own recon grain is preserved.

PSF SCALE from the MEASURED real edge-spread band (this box: ESF-L 1.45/2.38/3.54 vox at p25/50/90). Paganin/TIE-Hom
ESF is Lorentzian, 10-90 rise = 2*ln5*L = 3.219*L; a Gaussian has 10-90 rise = 2.563*sigma -> sigma = 1.256*L.

PATCHY BY DESIGN: g varies smoothly in-plane over a self-affine field, so one interface carries a fused patch
(g~0.1, invisible), a faintly parted patch (g~0.9), and a smooth transition -- the exact target morphology. The
label always says two sheets, so the only viable strategy is to propagate the boundary from where it is visible
into where it is not.
"""
import numpy as np
from scipy import ndimage as ndi

ESF_L_P25, ESF_L_P50, ESF_L_P90 = 1.45, 2.38, 3.54     # measured real edge-spread band (native voxels)
LORENTZ_TO_GAUSS = 2 * np.log(5) / 2.563                # = 1.256; matches 10-90 rise of Lorentzian-L to Gaussian-sigma


def interior_air(ct, axis, air_thresh):
    """Air voxels BETWEEN papyrus along the cross-sheet axis (the real inter-sheet seams), excluding open air
    outside the stack. Same definition as weld.py / gap_collapse_contact so all mechanics agree."""
    v = np.moveaxis(ct, axis, -1)
    air = v < air_thresh
    pap = ~air
    seen_pre = np.cumsum(pap, axis=-1) > 0
    seen_post = np.cumsum(pap[..., ::-1], axis=-1)[..., ::-1] > 0
    return np.moveaxis(air & seen_pre & seen_post, -1, axis)


def seam_thickness(gap, axis):
    """Per-voxel thickness t of the inter-sheet air RUN it belongs to (runs connected only along `axis`).
    This is the quantized gap the pipeline produced; SVPV re-renders it to a continuous sub-voxel value."""
    g = np.moveaxis(gap, axis, -1)
    st = np.zeros((3, 3, 3), int); st[1, 1, :] = 1                  # connect ONLY along the last (cross-sheet) axis
    lab, n = ndi.label(g, structure=st)
    if n == 0:
        return np.zeros(gap.shape, np.float32)
    sizes = np.bincount(lab.ravel())
    # run length along the axis == voxel count (runs are 1-D by construction)
    t = np.where(lab > 0, sizes[lab], 0).astype(np.float32)
    return np.moveaxis(t, -1, axis)


def self_affine_2d(shape2d, rng, hurst=0.75):
    """Unit-RMS isotropic self-affine field (fibrous/paper H~0.75) — the in-plane patchiness of the adhesion."""
    ny, nx = shape2d
    qx = 2 * np.pi * np.fft.rfftfreq(nx); qy = 2 * np.pi * np.fft.fftfreq(ny)
    QX, QY = np.meshgrid(qx, qy); q = np.hypot(QX, QY)
    amp = np.zeros_like(q); m = q > 0
    amp[m] = q[m] ** (-(hurst + 1.0)); amp[0, 0] = 0.0
    spec = amp * (rng.normal(size=amp.shape) + 1j * rng.normal(size=amp.shape))
    h = np.fft.irfft2(spec, s=(ny, nx)); h -= h.mean()
    return (h / (h.std() + 1e-8)).astype(np.float32)


def svpv_film(ct, axis, rng, g_max=None, esf_L=None, air_thresh=None, t_max=4.0, patch_scale=None,
              patchy=True, return_meta=False):
    """Re-render the pipeline's QUANTIZED thin seams as CONTINUOUS SUB-VOXEL gaps.
      g_max      : max residual gap in NATIVE voxels (default sampled 0.15-0.9 = sub-resolution by construction)
      esf_L      : PSF edge-spread L in native vox (default sampled from the measured real band)
      t_max      : only seams this thin (vox) are converted; wider gaps are real, resolved, and left alone
      patch_scale: in-plane correlation length of the residual-gap field (default sampled 6-22 vox)
      patchy     : False -> uniform g == g_max everywhere (used to validate the forward model against theory)
    CT only — the labels already encode two sheets, so GT is exact by construction. Run BEFORE to_outputs so the
    now-fused fibre is partitioned to the nearest instance and carved."""
    ct = np.asarray(ct, np.float32)
    if air_thresh is None:
        from skimage.filters import threshold_otsu
        air_thresh = float(threshold_otsu(ct))
    if g_max is None:
        g_max = float(rng.uniform(0.15, 0.9))                      # SUB-RESOLUTION by construction
    if esf_L is None:
        esf_L = float(rng.uniform(ESF_L_P25, ESF_L_P90))
    if patch_scale is None:
        patch_scale = float(rng.uniform(6.0, 22.0))
    sigma = esf_L * LORENTZ_TO_GAUSS

    gap = interior_air(ct, axis, air_thresh)
    if not gap.any():
        return (ct, dict(seam_vox=0, converted=0)) if return_meta else ct
    t = seam_thickness(gap, axis)
    thin = gap & (t > 0) & (t <= t_max)                            # only the quantized seams are ours to re-render
    if not thin.any():
        return (ct, dict(seam_vox=int(gap.sum()), converted=0)) if return_meta else ct

    pap = ct >= air_thresh
    I_pap = float(np.median(ct[pap])) if pap.any() else 180.0
    I_air = float(np.median(ct[~pap])) if (~pap).any() else 30.0
    delta = max(I_pap - I_air, 1.0)

    # target residual gap g(x,y) in [0, g_max]
    ip = [s for i, s in enumerate(ct.shape) if i != axis]
    if not patchy:
        g2 = np.full(tuple(ip), g_max, np.float32)
    else:
        h = ndi.gaussian_filter(self_affine_2d(tuple(ip), rng), patch_scale)
        # RANK (empirical-CDF) mapping, not min-max: after a strong low-pass the field is nearly constant and a
        # min-max rescale would stretch numerical noise back across the full range, silently randomising g per
        # column (measured: it made every column an arbitrary g and broke the forward-model check). The rank map
        # is scale-free, so g is uniformly distributed over [0, g_max] whatever the smoothing.
        order = h.ravel().argsort().argsort().astype(np.float32)
        g2 = (order / max(order.size - 1, 1)).reshape(h.shape) * g_max
    g = np.broadcast_to(np.expand_dims(g2.astype(np.float32), axis), ct.shape)

    # 1. FILL the quantized (unphysically sharp) seam back to papyrus -> genuinely fused contact
    out = ct + delta * thin.astype(np.float32)
    # 2. CARVE the sub-voxel film through the forward model: total mass Delta*g per unit area, spread over the
    #    seam's t voxels, convolved once with the measured PSF -> the dip a real g-voxel gap would produce
    src = np.zeros(ct.shape, np.float32)
    src[thin] = (delta * g[thin] / np.maximum(t[thin], 1e-6)).astype(np.float32)
    out = np.clip(out - ndi.gaussian_filter(src, sigma), 0, 255)
    if return_meta:
        return out, dict(seam_vox=int(gap.sum()), converted=int(thin.sum()), g_max=round(g_max, 3),
                         esf_L=round(esf_L, 2), sigma=round(sigma, 2), delta=round(delta, 1),
                         mean_t=round(float(t[thin].mean()), 2), mean_g=round(float(g[thin].mean()), 3))
    return out


if __name__ == "__main__":
    ok = True
    sig_ref = 2.0 * LORENTZ_TO_GAUSS

    K = 16   # super-sampling factor used ONLY to render the reference truth at sub-voxel fidelity

    def scanned(gap_t, Z=96, noise=0.0, seed=0):
        """REFERENCE TRUTH: the image a scanner would record for a gap of EXACTLY gap_t voxels (gap_t may be
        fractional). Built on a K-times finer grid so a sub-voxel gap is representable, convolved with the PSF,
        then block-averaged to the native grid == the partial-volume/tissue-fraction operator. This is the
        independent yardstick svpv_film must reproduce; it is NOT how svpv works."""
        zf = np.arange(Z * K, dtype=np.float32) / K
        col = np.where((zf >= 20) & (zf < 48.0), 180.0,
                       np.where((zf >= 48.0 + gap_t) & (zf < 76), 180.0, 30.0))
        col = ndi.gaussian_filter1d(col, sig_ref * K)           # PSF on the fine grid
        col = col.reshape(Z, K).mean(axis=1)                     # block-average -> native (partial volume)
        img = np.broadcast_to(col[:, None, None], (Z, 48, 48)).astype(np.float32).copy()
        if noise:
            img = img + np.random.default_rng(seed).standard_normal(img.shape).astype(np.float32) * noise
        return np.clip(img, 0, 255)

    def pipeline_like(gap_t, Z=96, noise=0.0, seed=0):
        """What gap_collapse ACTUALLY emits: papyrus slabs with a HARD, air-valued seam of gap_t (integer) voxels.
        No PSF is applied at the seam -- the warp resamples material, so the seam is an unphysically sharp hole.
        This is svpv's input."""
        col = np.full(Z, 30.0, np.float32)
        col[20:48] = 180.0
        col[48 + int(gap_t):76] = 180.0                          # seam = [48, 48+gap_t) left at air
        img = np.broadcast_to(col[:, None, None], (Z, 48, 48)).astype(np.float32).copy()
        if noise:
            img = img + np.random.default_rng(seed).standard_normal(img.shape).astype(np.float32) * noise
        return np.clip(img, 0, 255)

    # T1: THE forward-model test -- svpv must turn the pipeline's hard 1-vox seam into the SAME image a scanner
    # would have recorded for a true sub-voxel g-gap (reference rendered independently on a 16x finer grid).
    print("T1 forward-model fidelity: svpv(hard 1-vox seam -> g)  vs  independently rendered true g-gap scan")
    for g in (0.2, 0.4, 0.8):
        src = pipeline_like(1)
        got = svpv_film(src, 0, np.random.default_rng(0), g_max=g, esf_L=2.0, air_thresh=100.0,
                        patchy=False)          # uniform g -> directly comparable to the flat reference gap
        ref = scanned(g)
        dip_got = 180.0 - got[:, 24, 24][44:54].min()
        dip_ref = 180.0 - ref[:, 24, 24][44:54].min()
        err = abs(dip_got - dip_ref) / max(dip_ref, 1e-6)
        print(f"   g={g:.1f}: dip rendered={dip_got:5.1f} vs true={dip_ref:5.1f} CT-units  (rel.err {err:5.1%})  "
              f"{'PASS' if err < 0.35 else 'FAIL'}")
        ok &= err < 0.35

    # T2: monotone, meaningful curriculum across the sub-voxel band
    src = pipeline_like(1)
    dips = []
    for g in (0.1, 0.3, 0.6, 0.9):
        o = svpv_film(src, 0, np.random.default_rng(1), g_max=g, esf_L=2.0, air_thresh=100.0, patchy=False)
        dips.append(round(float(180.0 - o[:, 24, 24][44:54].min()), 2))
    mono = all(dips[i] < dips[i + 1] for i in range(len(dips) - 1))
    print(f"T2 curriculum: residual dip {dips} CT-units for g=0.1/0.3/0.6/0.9 -> "
          f"{'PASS' if mono and dips[-1] - dips[0] > 3 else 'FAIL'}")
    ok &= mono and dips[-1] - dips[0] > 3

    # T3: the hard seam FUSES (rises above air threshold) -> fibre becomes one blob -> partition+carve can act
    o = svpv_film(src, 0, np.random.default_rng(2), g_max=0.3, esf_L=2.0, air_thresh=100.0, patchy=False)
    before_air = int((src[:, 24, 24][44:54] < 100).sum()); after_air = int((o[:, 24, 24][44:54] < 100).sum())
    print(f"T3 seam fuses: sub-threshold voxels in seam {before_air} -> {after_air}  "
          f"{'PASS' if before_air > 0 and after_air == 0 else 'FAIL'} (fibre continuous; labels still say 2 sheets)")
    ok &= before_air > 0 and after_air == 0

    # T4: sub-resolution -- residual dip is a small fraction of a fully-resolved gap's contrast
    frac = (180.0 - o[:, 24, 24][44:54].min()) / 150.0
    print(f"T4 sub-resolution: g=0.3 dip is {frac:.1%} of a fully-resolved gap  {'PASS' if frac < 0.25 else 'FAIL'}")
    ok &= frac < 0.25

    # T5: patchiness -- one interface carries fused AND parted patches simultaneously
    o5, m5 = svpv_film(pipeline_like(1, noise=2.0, seed=4), 0, np.random.default_rng(5), g_max=0.9, esf_L=2.0,
                       air_thresh=100.0, patch_scale=10.0, return_meta=True)
    prof = 180.0 - o5[44:54].min(axis=0)
    lo, hi = np.percentile(prof, 10), np.percentile(prof, 90)
    print(f"T5 patchy: per-column residual dip p10={lo:.1f} p90={hi:.1f} CT-units (mean g={m5['mean_g']}, "
          f"mean t={m5['mean_t']}) -> {'PASS' if hi - lo > 2 else 'FAIL'} (fused & parted coexist in one cube)")
    ok &= hi - lo > 2

    # T6: wide REAL gaps are left alone (only quantized thin seams are ours to re-render)
    wide = pipeline_like(10)
    o6, m6 = svpv_film(wide, 0, np.random.default_rng(6), g_max=0.5, esf_L=2.0, air_thresh=100.0, t_max=4.0,
                       return_meta=True)
    untouched = np.allclose(wide, o6, atol=1e-3)
    print(f"T6 wide gaps untouched: 10-vox gap -> converted={m6['converted']} voxels, CT identical={untouched} "
          f"{'PASS' if untouched else 'FAIL'}")
    ok &= untouched

    # T7: robustness with noise
    o7 = svpv_film(pipeline_like(1, noise=4.0, seed=7), 0, np.random.default_rng(8), air_thresh=100.0)
    print(f"T7 robustness: range=({o7.min():.0f},{o7.max():.0f}) finite={np.isfinite(o7).all()} "
          f"{'PASS' if o7.min() >= 0 and o7.max() <= 255 and np.isfinite(o7).all() else 'FAIL'}")
    ok &= o7.min() >= 0 and o7.max() <= 255 and np.isfinite(o7).all()

    print("SVPV SELF-TEST:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
