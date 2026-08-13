"""GEOMETRIC normalisation: bring a block's wrap period to the one the network was trained at.

This is the exact counterpart of `normalize.py`. That module harmonises INTENSITY to the training
distribution; this one harmonises GEOMETRY. We were doing the first and not the second, and the
geometric mismatch is larger than any intensity mismatch we ever corrected:

    per-cube median wrap period lambda (vox @ 7.91 um)
      REAL Scroll-1 (training)  16.69      min over 30 cubes: 14.50
      SYNTH        (training)   15.69
      Scroll-3 core (target)    10.50      max over 30 cubes: 13.25
      Scroll-4 core (target)    10.56

The training and target distributions DO NOT OVERLAP. And the network is a single-scale filter bank
-- 18 stacked 3^3 convs, no pooling, no dilation, no multi-scale branch -- with a measured tuning
curve (512 teacher-placed FOVs/point, 8 labelled val cubes):

    apparent lambda   8.3    10.0   13.3   14.9   16.6   21.6
    AUC self-vs-neigh 0.722  0.811  0.925  0.938  0.931  0.888
    mean p on NEIGH   0.772  0.725  0.424  0.342  0.313  0.325

At the cores' native lambda ~10 the model assigns the NEIGHBOURING wrap p = 0.73 -- it is actively
claiming it. That is the merge mechanism, measured rather than inferred.

Controlled test on 5 labelled Scroll-1 cubes / 41 adjacency pairs: compacting to the core period
raised merge 0.073 -> 0.317 (Fisher p = 0.0105); upsampling that SAME image back -- adding zero
information -- dropped it to 0.024 (p = 0.00069 vs compact), statistically indistinguishable from
native. Rescaling adds no information and changes no dimensionless quantity; what it changes is which
dimensionless quantity the network's FIXED-VOXEL machinery corresponds to. All of that machinery is
in voxels: measured ERF r50 = 7 / r90 = 13, delta 6, face half-width 6, seed_spacing 6,
seed_reject_radius 3, min_instance_size 2000.

Why not fix it in training instead: `aug_scale = 0.12` is drawn PER AXIS (volumes.py:242), so its
most extreme single draw is 0.880 -- reaching the cores needs 0.62. It structurally cannot.
(That remains the durable fix for a future run; this is the one available today, with no retrain.)

TWO SCALARS MUST TRAVEL WITH THE RESAMPLE or the result is silently wrong:
  fill_max_steps *= s**2   an FFN fill walks a SURFACE, so step count scales with area, not length
  min_instance_size *= s**3  the commit floor counts VOLUME
Everything else (delta, seed spacing, face width) is correct as-is once the image is at training
scale -- that is the entire point of doing it this way rather than retuning each constant.

Estimator provenance: structure-tensor normal + run-length along it, identical to the census that
produced the table above. Deliberately NOT ffn.ctstats.wrap_period, which research/09 measured
locking onto a sub-harmonic (77 where truth was 21) on radial profiles.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

STEP = 0.25          # ray sampling step, voxels
REACH = 44.0         # +/- ray half-length, voxels
LAMBDA_STAR = 15.5   # target period: the tuning-curve peak (14.9-16.6 plateau)


def _air_pap_threshold(vol: np.ndarray) -> float:
    """Air/papyrus boundary -- the codebase's own EM fit, not a reimplementation.

    A first version of this file re-derived the 2-Gaussian fit with a HARD-CODED component variance
    of 400. That threshold sat too high, so it split single bright bands into multiple runs and the
    measured period came out ~20% short on every domain (13.94 where the census reads 16.69 on the
    training corpus). Reuse the audited estimator; do not re-derive physics that already has one.
    """
    import sys
    if "/root/surface_detection/FFN" not in sys.path:
        sys.path.insert(0, "/root/surface_detection/FFN")
    from ffn.ctstats import air_papyrus_threshold
    return float(air_papyrus_threshold(vol))


def _sheet_normals(vol: np.ndarray, pts: np.ndarray, sigma=1.0, rho=3.0) -> np.ndarray:
    """Local sheet normal = principal eigenvector of the structure tensor of the smoothed CT."""
    v = ndi.gaussian_filter(vol.astype(np.float32), sigma)
    gz, gy, gx = np.gradient(v)
    comps = {}
    for a, ga in (("z", gz), ("y", gy), ("x", gx)):
        for b, gb in (("z", gz), ("y", gy), ("x", gx)):
            k = "".join(sorted(a + b))
            if k not in comps:
                comps[k] = ndi.gaussian_filter(ga * gb, rho)
    zz, yy, xx = pts[:, 0], pts[:, 1], pts[:, 2]
    J = np.empty((len(pts), 3, 3), np.float64)
    idx = {"z": 0, "y": 1, "x": 2}
    for a in "zyx":
        for b in "zyx":
            J[:, idx[a], idx[b]] = comps["".join(sorted(a + b))][zz, yy, xx]
    _, V = np.linalg.eigh(J)
    n = V[:, :, 2]
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)


def measure_lambda(vol: np.ndarray, n_rays: int = 1500, seed: int = 0) -> float:
    """Median band-centre-to-band-centre spacing along the local sheet normal, in voxels.

    Returns 0.0 when the block has too little material to support an estimate; callers must treat
    that as "do not rescale" rather than as a small lambda.
    """
    if vol.dtype != np.uint8:
        vol = np.clip(vol, 0, 255).astype(np.uint8)
    thr = _air_pap_threshold(vol)
    pap = vol >= thr
    Z, Y, X = vol.shape
    inner = np.zeros_like(pap)
    inner[12:Z - 12, 12:Y - 12, 12:X - 12] = True
    cand = np.flatnonzero((pap & inner).ravel())
    if cand.size < 200:
        return 0.0
    rng = np.random.default_rng(seed)
    take = rng.choice(cand, size=min(n_rays, cand.size), replace=False)
    pts = np.stack(np.unravel_index(take, vol.shape), 1).astype(np.int64)
    nrm = _sheet_normals(vol, pts)
    K = int(2 * REACH / STEP) + 1
    s = (np.arange(K) * STEP - REACH).astype(np.float32)
    coords = pts[:, None, :].astype(np.float32) + s[None, :, None] * nrm[:, None, :]
    c = np.rint(coords).astype(np.int32)
    for a in range(3):
        np.clip(c[..., a], 0, vol.shape[a] - 1, out=c[..., a])
    P = pap[c[..., 0], c[..., 1], c[..., 2]]
    c0 = K // 2
    lams = []
    for i in range(len(pts)):
        p = P[i]
        if not p[c0]:
            continue
        d = np.diff(np.concatenate(([0], p.view(np.int8), [0])))
        st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        keep = (st > 0) & (en < K)                 # drop the two clipped end runs
        st, en = st[keep], en[keep]
        if len(st) < 2:
            continue
        ctr = 0.5 * (st + en - 1) * STEP
        lams.extend(np.diff(ctr).tolist())
    return float(np.median(lams)) if len(lams) >= 50 else 0.0


def scale_for(lam: float, lam_star: float = LAMBDA_STAR,
              lo: float = 1.0, hi: float = 2.0) -> float:
    """Resample factor bringing `lam` to `lam_star`, clamped.

    Lower clamp 1.0: we never DOWN-sample. The tuning curve is asymmetric -- over-zooming is cheap
    (AUC 0.938 -> 0.888 from lambda 14.9 to 21.6) while under-zooming is expensive (0.722 at 8.3) --
    and downsampling additionally destroys information, which upsampling does not.
    """
    if not np.isfinite(lam) or lam <= 0:
        return 1.0
    return float(np.clip(lam_star / lam, lo, hi))


def resample(vol: np.ndarray, s: float) -> np.ndarray:
    """Trilinear upsample by `s`. uint8 in, uint8 out."""
    if abs(s - 1.0) < 1e-3:
        return vol
    out = ndi.zoom(vol.astype(np.float32), s, order=1, mode="nearest")
    return np.clip(out, 0, 255).astype(np.uint8)


def labels_back(lab: np.ndarray, s: float, shape) -> np.ndarray:
    """Map labels from the rescaled grid back to the native grid (nearest neighbour)."""
    if abs(s - 1.0) < 1e-3:
        return lab[:shape[0], :shape[1], :shape[2]]
    idx = [np.clip(np.rint(np.arange(n) * s).astype(int), 0, lab.shape[a] - 1)
           for a, n in enumerate(shape)]
    return lab[np.ix_(idx[0], idx[1], idx[2])]


def lambda_relative_cfg(cfg, lam: float, lam_star: float = LAMBDA_STAR):
    """FALLBACK for blocks decoded at NATIVE scale despite a measured off-target lambda
    (compute-capped runs, or s at the clamp): make the voxel-unit decode constants
    lambda-RELATIVE instead of absolute.

    This fixes only the DECODE-GEOMETRY half of the mismatch -- the model's perception stays
    off-peak -- so it is strictly weaker than rescaling. Ratios, with r = lam / lam_star (< 1 on
    compact material):

      delta            <= lam/2 (the half-period design rule; at core lambda 10.3 the shipped
                       delta=6 is 0.58*lambda and 47.3% of the (2*delta+1)^2 gate face lies beyond
                       lambda/2 transversely -- the gate stops braking). Floor 3: below that the
                       walk cannot clear its own FOV overlap.
      seed_spacing     ~ r     (seeds per wrap held constant)
      min_instance     ~ r^3   (objects occupy r^3 fewer voxels)
      fill_max_steps   ~ r^2   (a fill walks a surface)

    NOTE delta is a TRAINED quantity (the movement policy saw delta=6 walks); changing it moves the
    model off its training distribution, which this repo has measured to be dangerous elsewhere.
    That is why this is the fallback and rescaling is the primary: rescaling reaches the same
    ratios without touching any trained constant.
    """
    import dataclasses
    if not np.isfinite(lam) or lam <= 0:
        return cfg
    r = lam / lam_star
    delta = int(max(3, min(cfg.delta, np.floor(lam / 2.0))))
    return dataclasses.replace(
        cfg,
        delta=delta,
        seed_spacing=max(2, int(round(cfg.seed_spacing * r))),
        min_instance_size=max(200, int(round(cfg.min_instance_size * r ** 3))),
        fill_max_steps=int(round(cfg.fill_max_steps * r * r)),
    )


def scaled_cfg(cfg, s: float):
    """cfg with the two GRID-DEPENDENT scalars corrected for the resample.

    A fill walks a SURFACE, so its step count scales with area (s^2), not length. The commit floor
    counts volume (s^3). Leaving either unscaled is silently wrong: at s=2 an unscaled
    min_instance_size makes the commit floor 8x more permissive, and an unscaled fill budget
    truncates every object -- the two biases point in opposite directions, so they do not cancel.
    """
    import dataclasses
    return dataclasses.replace(
        cfg,
        fill_max_steps=int(round(cfg.fill_max_steps * s * s)),
        min_instance_size=int(round(cfg.min_instance_size * s ** 3)),
    )
