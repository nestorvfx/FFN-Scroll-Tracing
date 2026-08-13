#!/usr/bin/env python3
"""Structured CT-artifact augmentation (rank-A): rings + Paganin-residual streaks + anisotropic (cylindrical) NPS.

WHY (physics, not generic noise): Herculaneum scans are PARALLEL-BEAM MONOCHROMATIC synchrotron uCT with
single-distance Paganin reconstruction. The real structured artifacts are therefore:
  * RING artifacts -- miscalibrated/nonlinear detector columns -> sinogram stripes -> concentric intensity bands
    around the rotation axis (Vo, sarepy; arXiv 2402.05983). In a local cube at radius R from the axis they are
    gently-curved bands, tangential, near-constant along z (detector-column persistence). CRITICALLY: papyrus wraps
    are ALSO roughly concentric around the scroll axis -> ring bands run QUASI-PARALLEL to real sheets = a false
    sheet-like intensity cue exactly in our decision space.
  * PAGANIN-RESIDUAL STREAKS -- single-distance TIE-Hom assumes one homogeneous material; at strong phase
    gradients/interfaces the linearization fails, leaving long-range oriented streaks (Eikonal Phase Retrieval,
    IUCr J.Synchrotron Rad 2025 / arXiv 2601.22793). Strictly IN-PLANE (all projection lines lie in the slice).
  * ANISOTROPIC noise texture -- FBP noise is correlated in-plane (ramp-filter shaped) and differently along z;
    a 3D radially-averaged NPS (our current grain model) cannot represent that anisotropy. Standard CT practice is
    the cylindrically-averaged NPS(k_perp, k_z) (image-domain 3D-NPS noise insertion, e.g. PMC12070530).
  * NO beam hardening: monochromatic beam -> cupping/BH dark bands are the WRONG physics here; deliberately absent.

HONESTY MODEL: synthetic cubes are moved REAL voxels, so they already carry real artifacts (warped). This module is
therefore a DOMAIN-RANDOMIZATION augmentation -- teach the net invariance to structured false cues (especially
sheet-parallel ring bands) -- with amplitudes CALIBRATED to (and bounded by) levels measured in real cubes
(measure_artifact_params), never exceeding them. GT untouched (intensity-only). Gates: leakage probe must not
increase over the un-injected floor, and the faint-gap survival check must pass (the injected structure must not
erase the residual sub-voxel gap cue beyond what real artifacts do).
"""
import numpy as np
from scipy import ndimage as ndi


# ---------------------------------------------------------------- anisotropic (cylindrical) NPS
def cylindrical_nps(res, axis=0, n_perp=40, n_ax=17):
    """Cylindrically-averaged noise power spectrum NPS(k_perp, k_ax) of a residual volume; `axis` = tomographic
    rotation axis (z). Returns (kp_centers, ka_centers, nps[n_perp, n_ax])."""
    r = (res - res.mean()).astype(np.float32)
    P = np.abs(np.fft.rfftn(r)) ** 2 / r.size
    fr = [np.fft.fftfreq(s) for s in r.shape]
    fr[-1] = np.fft.rfftfreq(r.shape[-1])
    F = np.meshgrid(*fr, indexing="ij")
    ka = np.abs(F[axis])
    ip = [F[a] for a in range(3) if a != axis]
    kp = np.sqrt(ip[0] ** 2 + ip[1] ** 2)
    pb = np.linspace(0, np.sqrt(0.5), n_perp + 1)
    ab = np.linspace(0, 0.5, n_ax + 1)
    ip_idx = np.clip(np.digitize(kp.ravel(), pb) - 1, 0, n_perp - 1)
    ax_idx = np.clip(np.digitize(ka.ravel(), ab) - 1, 0, n_ax - 1)
    flat = ip_idx * n_ax + ax_idx
    nps = np.bincount(flat, weights=P.ravel(), minlength=n_perp * n_ax)
    cnt = np.bincount(flat, minlength=n_perp * n_ax)
    nps = (nps / np.maximum(cnt, 1)).reshape(n_perp, n_ax)
    return 0.5 * (pb[:-1] + pb[1:]), 0.5 * (ab[:-1] + ab[1:]), nps


def colored_noise_cyl(shape, kp_c, ka_c, nps, rng, axis=0):
    """Unit-variance Gaussian noise colored to a cylindrical NPS(k_perp, k_ax) via bilinear spectral interpolation."""
    fr = [np.fft.fftfreq(s) for s in shape]
    fr[-1] = np.fft.rfftfreq(shape[-1])
    F = np.meshgrid(*fr, indexing="ij")
    ka = np.abs(F[axis])
    ip = [F[a] for a in range(3) if a != axis]
    kp = np.sqrt(ip[0] ** 2 + ip[1] ** 2)
    pi = np.clip(np.searchsorted(kp_c, kp) - 1, 0, len(kp_c) - 2)
    ai = np.clip(np.searchsorted(ka_c, ka) - 1, 0, len(ka_c) - 2)
    wp = np.clip((kp - kp_c[pi]) / (kp_c[pi + 1] - kp_c[pi] + 1e-12), 0, 1)
    wa = np.clip((ka - ka_c[ai]) / (ka_c[ai + 1] - ka_c[ai] + 1e-12), 0, 1)
    v = (nps[pi, ai] * (1 - wp) * (1 - wa) + nps[pi + 1, ai] * wp * (1 - wa)
         + nps[pi, ai + 1] * (1 - wp) * wa + nps[pi + 1, ai + 1] * wp * wa)
    amp = np.sqrt(np.maximum(v, 0.0))
    G = np.fft.rfftn(rng.standard_normal(shape).astype(np.float32)) * amp
    g = np.fft.irfftn(G, s=shape).astype(np.float32)
    g -= g.mean()
    return g / (g.std() + 1e-8)


# ---------------------------------------------------------------- measurement from REAL cubes (calibration)
def measure_artifact_params(ct, axis=0, center_inplane=None):
    """Estimate structured-artifact levels in a real cube: ring-band radial profile (tangential coherence around
    `center_inplane`, absolute in-plane coords; None -> unknown center, rings reported 0) and the anisotropic grain
    NPS. Returns a dict usable by inject_artifacts (amplitudes in CT units)."""
    ct = ct.astype(np.float32)
    hi = ct - ndi.gaussian_filter(ct, 1.2)
    air = ct < np.percentile(ct, 35)
    out = {"grain_std": float(hi[air].std()) if air.any() else 3.0}
    kp, ka, nps = cylindrical_nps(np.where(air, hi, 0.0), axis=axis)
    out["nps"] = (kp, ka, nps)
    ring_amp = 0.0
    if center_inplane is not None:
        ipax = [a for a in range(3) if a != axis]
        g = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in ct.shape], indexing="ij")
        rho = np.hypot(g[ipax[0]] - center_inplane[0], g[ipax[1]] - center_inplane[1])
        rbin = rho.astype(np.int64)
        m = air & np.isfinite(hi)
        prof = np.bincount(rbin[m], weights=hi[m]) / np.maximum(np.bincount(rbin[m]), 1)
        prof = prof[np.bincount(rbin[m]) > 200]                       # tangentially well-averaged radii only
        if prof.size > 16:
            ring_amp = float(np.std(prof - ndi.uniform_filter1d(prof, 15)))   # banding above the smooth trend
    out["ring_amp"] = ring_amp
    # STREAK amplitude, same construction as rings but linear: directional mean-profiles of the
    # air high-pass, banding above the smooth trend, worst in-plane direction. Without this the
    # injector had no measured streak level and defaulted to 1.0 CT units regardless of the scan.
    ipax = [a for a in range(3) if a != axis]
    sa = 0.0
    for d in ipax:
        m2 = np.where(air, hi, np.nan)
        prof = np.nanmean(m2, axis=tuple(a for a in range(3) if a != d))
        prof = prof[np.isfinite(prof)]
        if prof.size > 32:
            sa = max(sa, float(np.nanstd(prof - ndi.uniform_filter1d(prof, 15))))
    out["streak_amp"] = sa
    return out


# ---------------------------------------------------------------- injection (intensity-only; GT untouched)
def inject_rings(ct, rng, axis=0, amp=1.5, R_range=(600.0, 4000.0), n_bands_scale=1.0):
    """Concentric intensity bands around a virtual rotation axis at distance R (sampled from the plausible scroll
    range) -- locally: gently-curved tangential bands, near-z-invariant with slow angular/z decorrelation, 1-3 vox
    radial correlation (detector-pixel scale) + occasional wider partial arcs."""
    shp = ct.shape
    ipax = [a for a in range(3) if a != axis]
    g = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in shp], indexing="ij")
    R = float(rng.uniform(*R_range))
    phi = float(rng.uniform(0, 2 * np.pi))
    cy = shp[ipax[0]] / 2 - R * np.cos(phi)
    cx = shp[ipax[1]] / 2 - R * np.sin(phi)                           # axis center outside the cube at distance R
    rho = np.hypot(g[ipax[0]] - cy, g[ipax[1]] - cx)
    rmin = float(rho.min())
    n_r = int(rho.max() - rmin) + 3
    # radial banding profile: pink-ish detector-response noise + sparse stronger single-column rings
    prof = ndi.gaussian_filter1d(rng.standard_normal(n_r).astype(np.float32), 1.2)
    n_strong = rng.poisson(2.0 * n_bands_scale)
    for _ in range(int(n_strong)):
        c = rng.integers(0, n_r); wdt = float(rng.uniform(0.7, 1.8)); a = float(rng.uniform(1.5, 3.0)) * rng.choice([-1, 1])
        xs = np.arange(n_r, dtype=np.float32)
        prof += a * np.exp(-0.5 * ((xs - c) / wdt) ** 2)
    prof /= (np.abs(prof).std() + 1e-8)
    band = np.interp(rho - rmin, np.arange(n_r, dtype=np.float32), prof)
    # slow z modulation (detector column drift) + partial-arc angular envelope
    zprof = 0.75 + 0.25 * ndi.gaussian_filter1d(rng.standard_normal(shp[axis]).astype(np.float32), 12)
    zmod = zprof.reshape([-1 if a == axis else 1 for a in range(3)])
    ang = np.arctan2(g[ipax[1]] - cx, g[ipax[0]] - cy)
    aenv = 0.8 + 0.2 * np.cos(ang * float(rng.uniform(30, 90)) + float(rng.uniform(0, 6.28)))
    return np.clip(ct + amp * band * zmod * aenv, 0, 255).astype(np.float32)


def inject_streaks(ct, rng, axis=0, amp=1.2, n_dirs=3, length=80.0, width=1.6):
    """Paganin-residual style long-range oriented IN-PLANE streaks seeded at strong in-plane gradients (sheet
    edges/dense interfaces). Zero-mean dark/bright banding (DoG across the streak), correlated over a few slices."""
    ct = ct.astype(np.float32)
    sm = ndi.gaussian_filter(ct, 1.5)
    gr = np.zeros_like(ct)
    ipax = [a for a in range(3) if a != axis]
    for a in ipax:
        gr += np.gradient(sm, axis=a) ** 2
    seed = np.sqrt(gr)
    seed = np.where(seed > np.percentile(seed, 97), seed, 0.0)        # strongest interfaces only
    seed /= (seed.max() + 1e-8)
    field = np.zeros_like(ct)
    # oriented line blur IS a convolution: the old 150-shift quadrature cost ~25 s/injection and
    # aliased at its 3-voxel step. One FFT of the seed + an analytic transfer function per
    # direction (sinc along the line x gaussian-DoG across it x gaussian along z) is ~80x faster
    # and exact -- length/width become continuous parameters instead of loop counts.
    try:
        from scipy import fft as _sfft
        _rfftn, _irfftn = _sfft.rfftn, _sfft.irfftn
    except Exception:
        _rfftn, _irfftn = np.fft.rfftn, np.fft.irfftn
    S = _rfftn(seed)
    fr = [np.fft.fftfreq(seed.shape[0]), np.fft.fftfreq(seed.shape[1]),
          np.fft.rfftfreq(seed.shape[2])]
    fgrid = [fr[0][:, None, None], fr[1][None, :, None], fr[2][None, None, :]]
    for _ in range(int(n_dirs)):
        th = float(rng.uniform(0, np.pi))
        d = np.zeros(3, np.float32); d[ipax[0]] = np.cos(th); d[ipax[1]] = np.sin(th)
        perp = np.zeros(3, np.float32); perp[ipax[0]] = -np.sin(th); perp[ipax[1]] = np.cos(th)
        f_par = d[0] * fgrid[0] + d[1] * fgrid[1] + d[2] * fgrid[2]
        f_perp = perp[0] * fgrid[0] + perp[1] * fgrid[1] + perp[2] * fgrid[2]
        f_z = fgrid[axis]
        H_line = np.sinc(np.clip(2.0 * length * f_par, -1e6, 1e6))          # box of length 2L along d
        g_n = np.exp(-2.0 * (np.pi * width * f_perp) ** 2)                  # gaussian across the streak
        g_w = np.exp(-2.0 * (np.pi * width * 2.5 * f_perp) ** 2)
        g_z = np.exp(-2.0 * (np.pi * 2.0 * f_z) ** 2)                       # slight z coherence
        field += _irfftn(S * (H_line * (g_n - g_w) * g_z).astype(np.complex64),
                         s=seed.shape).astype(np.float32)
    if field.std() > 1e-6:
        field *= 1.0 / field.std()
    return np.clip(ct + amp * field, 0, 255).astype(np.float32)


def inject_artifacts(ct, rng, axis=0, params=None, ring_amp=None, streak_amp=None, grain_frac=0.0):
    """Compose the artifact augmentation. Amplitudes default to (and are clamped by) measured-real `params`.
    grain_frac>0 additionally adds a small cylindrical-NPS-matched grain top-up (use sparingly; real voxels
    already carry real grain)."""
    p = params or {}
    # NO artificial floor: the old `max(measured, 0.6)` injected fake rings precisely when the real
    # reference measured none -- the user-visible "white lines that are nothing" (measured: traces
    # 1.10% of bright voxels vs real 0.014%). Measured-real means measured-real.
    ra = ring_amp if ring_amp is not None else float(p.get("ring_amp", 1.2))
    sa = streak_amp if streak_amp is not None else float(p.get("streak_amp", 1.0))
    out = ct.astype(np.float32)
    if ra > 0:
        out = inject_rings(out, rng, axis=axis, amp=float(rng.uniform(0.5, 1.0)) * ra)
    if sa > 0:
        out = inject_streaks(out, rng, axis=axis, amp=float(rng.uniform(0.5, 1.0)) * sa)
    if grain_frac > 0 and "nps" in p:
        kp, ka, nps = p["nps"]
        g = colored_noise_cyl(out.shape, kp, ka, nps, rng, axis=axis)
        out = out + grain_frac * p.get("grain_std", 2.0) * g
    return np.clip(out, 0, 255)


# ---------------------------------------------------------------- gates
def faint_gap_survival(rng, gap_vox=1.5, n=24, axis=0, **inj_kw):
    """GATE: a known sub-voxel/1-2vox dark gap between two bright slabs must remain detectable after injection.
    Measures the gap-contrast (slab minus gap mean) before/after; returns (mean retention ratio, pass_bool)."""
    ret = []
    for i in range(n):
        r = np.random.default_rng(1000 + i)
        cube = np.full((64, 64, 64), 40.0, np.float32)
        z0 = 32
        cube[z0 - 8:z0, :, :] = 170.0
        cube[z0 + int(np.ceil(gap_vox)):z0 + 8, :, :] = 170.0        # gap of ~gap_vox dark voxels at z0..
        cube += r.standard_normal(cube.shape).astype(np.float32) * 3.0
        gslice = cube[z0:z0 + int(np.ceil(gap_vox))].mean()
        slab = cube[z0 - 6:z0 - 1].mean()
        c0 = slab - gslice
        aug = inject_artifacts(cube, np.random.default_rng(2000 + i), axis=axis, **inj_kw)
        g1 = aug[z0:z0 + int(np.ceil(gap_vox))].mean()
        s1 = aug[z0 - 6:z0 - 1].mean()
        ret.append((s1 - g1) / (c0 + 1e-6))
    m = float(np.mean(ret))
    return m, m > 0.85


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # smoke: inject on a synthetic slab cube, check ranges + gap survival
    m, ok = faint_gap_survival(rng)
    print(f"GATE faint-gap survival: retention={m:.3f} (need >0.85) -> {'PASS' if ok else 'FAIL'}")
    cube = (np.clip(np.cumsum(np.random.default_rng(1).standard_normal((96, 96, 96)), 0) * 4 + 100, 0, 255)).astype(np.float32)
    p = measure_artifact_params(cube)
    aug = inject_artifacts(cube, rng, params=p)
    d = aug - cube
    print(f"smoke: delta std={d.std():.2f} CT-units, minmax=({aug.min():.0f},{aug.max():.0f}); "
          f"grain_std_measured={p['grain_std']:.2f}, ring_amp_measured={p['ring_amp']:.2f}")
    print("ARTIFACTS SELF-TEST:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
