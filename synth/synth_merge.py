#!/usr/bin/env python3
"""SOTA synthetic DENSE/FUSED-region generator (Approach A) for Vesuvius surface detection.

Goal: manufacture training cubes where papyrus sheets are packed far DENSER than the original -- up to fully
fused with NO visible gap (the failure mode current models can't separate) -- with ground-truth boundaries by
construction, so the model learns to separate sheets the pixels alone can't.

Approach A pipeline (validated against the 31-agent SOTA workflow + the beta0 de-risking test):
  1. load real instance cube (CT 0..255 + per-sheet integer IDs; mask is SPARSE).
  2. rigid diversity (flip + k*90 rotation).
  3. GAP-COLLAPSE densify: along the cross-sheet axis, shrink AIR runs (air = CT < Otsu, NOT the sparse mask)
     down to ZERO where a smooth random field is low -> sheets pack/fuse; spatial variation gives a touch->fused
     curriculum in one cube. Per-column cumsum warp (diffeomorphic in 1D; the SOTA workflow REFUTED the fancier
     SDF-gradient warp -- it folds). Real voxels are MOVED, so texture stays real.
  4. densest_crop to the most-papyrus out_size^3 window.
  5. LOCAL CONTACT RE-SCAN (not full-volume FBP -- that double-blurs already-real CT): only at the NEW contact
     band (where different sheets now abut) apply a small Gaussian PSF (partial-volume) + NPS/std-matched
     correlated grain, so the fused contact looks really scanned while the rest stays native real texture.
  6. EMIT (Dataset059-compatible):
       *_volume.nrrd  : CT uint8
       *_mask.nrrd    : binary {bg:0, fiber:1}, with inter-sheet boundaries CARVED (thin non-fiber) so a fused
                        pair is TWO fiber regions -> the model can express two surfaces.
       *_weight.npy   : per-voxel loss weight, high on the carved inter-sheet boundary (the AUXILIARY signal of
                        Approach A) so the thin separation is actually learned despite class imbalance.
Pure numpy/scipy/skimage. Determinism via --seed (vary per index for diversity)."""
import os, glob, argparse, numpy as np
from scipy import ndimage as ndi


def load_cube(d):
    import nrrd
    vol = np.asarray(nrrd.read(glob.glob(os.path.join(d, "*_volume.nrrd"))[0])[0]).astype(np.float32)
    msk = np.asarray(nrrd.read(glob.glob(os.path.join(d, "*_mask.nrrd"))[0])[0])
    # Source cubes are MIXED dtype (some uint8 0-255, some uint16 0-65535). Percentile-normalize every source to
    # a consistent 0-255 domain (matching Dataset059's CT) so the final uint8 cast doesn't saturate uint16 cubes
    # to uniform-255 garbage, and so the synthetic CT lands in medial_059's expected intensity range.
    lo, hi = np.percentile(vol, 0.5), np.percentile(vol, 99.5)
    vol = np.clip((vol - lo) / (hi - lo + 1e-6) * 255.0, 0, 255)
    return vol, msk


def cross_sheet_axis(mask):
    return int(np.argmax([(np.diff(mask, axis=ax) != 0).mean() for ax in range(3)]))


def rigid_diversity(vol, msk, rng):
    if rng.random() < 0.5:
        ax = int(rng.integers(0, 3)); vol = np.flip(vol, ax).copy(); msk = np.flip(msk, ax).copy()
    k = int(rng.integers(0, 4)); axes = tuple(int(x) for x in rng.choice(3, 2, replace=False))
    return np.rot90(vol, k, axes).copy(), np.rot90(msk, k, axes).copy()


def gap_collapse(vol, msk, axis, keep_air_max, rng, air_thresh, spatial=True, keep_air_min=0.0, smooth_plane=8.0):
    """Collapse AIR runs (vol<air_thresh) along `axis` so sheets pack denser -- to ZERO gap (fused) where the
    field is low. Per-column cumsum warp; tail beyond the warped content -> real background (no edge streaks).

    `keep_air_min` is the EASY-FLOOR FIX. The spatial field is rescaled to [keep_air_min, keep_air_max], not
    [0, keep_air_max]. Why it matters: min-max normalizing to a 0 floor guarantees that EVERY cube contains a
    fully-collapsed (fused) region -- the field's own minimum is 0 by construction, whatever keep_air_max is. Then
    densest_crop, which selects the most-papyrus window, lands on exactly those fused columns. Net effect (measured):
    the whole corpus was ALL-HARD -- median synth gap 1-2 vox vs real 11 (real p25/50/90 = 6/11/35), 0% of cubes
    reaching median >=4 -- so the sampler's 15% "easy floor" stratum never actually existed and split protection came
    only from the real cubes. With keep_air_min>0 no column can fully collapse, so an easy stratum is real.
    Keep keep_air_min=0 for the fused/tight strata, where full fusion is the POINT.

    `smooth_plane` is the LAMINARITY FIX (visual audit 2026-07-16). Each in-plane column was an INDEPENDENT 1D warp
    driven by a noisy binary air mask, so adjacent columns shifted by different amounts and originally-smooth sheets
    came out with pixel-scale sawtooth edges + deep 'icicles' at the content/fill boundary -- nothing like real merged
    sheets (smooth, laminar), and a global synthetic fingerprint a long training run can key on instead of the
    physical cue. Two established principles say how to fix it: (a) diffeomorphic registration REGULARIZES the
    displacement field (Gaussian smoothing, e.g. diffeomorphic demons); (b) thin-sheet mechanics: bending stiffness
    forbids deformation below a wavelength of a few sheet thicknesses (~8-10 vox), so a physical collapse field
    CANNOT vary at 1-voxel scale. Implementation: smooth the per-column CUMULATIVE displacement in-plane with
    sigma=smooth_plane (approx. one sheet thickness) and detect air on a sigma-1-smoothed volume so CT grain does not
    inject run-length noise. Monotonicity along the column survives (in-plane averaging of per-column monotone maps
    stays monotone) -> still fold-free. Fused regions stay fused (nearby columns now collapse coherently -- smooth
    contact patches, which is also the physical behaviour); the fill boundary becomes a smooth rolling line.
    smooth_plane=0 restores the legacy jagged behaviour."""
    v = np.moveaxis(vol, axis, -1).astype(np.float32)
    m = np.moveaxis(msk, axis, -1)
    shp = v.shape; N = shp[-1]
    air = (ndi.gaussian_filter(v, 1.0) if smooth_plane > 0 else v) < air_thresh
    if spatial and keep_air_max > 0:
        f = ndi.gaussian_filter(rng.random(shp, dtype=np.float32), sigma=10)
        f = (f - f.min()) / (f.max() - f.min() + 1e-6)                       # -> [0,1]
        f = keep_air_min + f * (keep_air_max - keep_air_min)                 # -> [keep_air_min, keep_air_max]
        step = np.where(air, f, 1.0).astype(np.float32)
    else:
        step = np.where(air, float(keep_air_max), 1.0).astype(np.float32)
    outpos = np.cumsum(step, axis=-1)
    if smooth_plane > 0:
        base = np.arange(1, N + 1, dtype=np.float32)
        u = outpos - base[None, None, :]                     # per-column cumulative displacement (<=0)
        u = ndi.gaussian_filter(u, sigma=(float(smooth_plane), float(smooth_plane), 0.0))
        outpos = u + base[None, None, :]
    fv = v.reshape(-1, N); fm = m.reshape(-1, N); fo = outpos.reshape(-1, N)
    src = np.arange(N, dtype=np.float32)
    bg = float(np.median(v[air])) if air.any() else 0.0
    ov = np.full_like(fv, bg); om = np.zeros_like(fm)
    grid = np.arange(N, dtype=np.float32)
    for c in range(fv.shape[0]):
        maxo = fo[c][-1]
        valid = grid <= maxo
        inp = np.interp(grid, fo[c], src)
        ov[c] = np.where(valid, np.interp(inp, src, fv[c]), bg)
        om[c] = np.where(valid, fm[c][np.clip(np.rint(inp).astype(np.int64), 0, N - 1)], 0)
    return np.moveaxis(ov.reshape(shp), -1, axis), np.moveaxis(om.reshape(shp), -1, axis)


def shear_warp(vol, msk, axis, rng, max_shift=6.0, smooth=40.0):
    """Diffeomorphic SHEAR/BEND for the 3+-sheet pile-up arm: smooth in-plane displacement that VARIES along the
    cross-sheet axis, so initially-parallel sheets tilt and CONVERGE -> acute 3-sheet junctions / triple points the
    pairwise-collapse arm rarely makes. Same warp on CT (linear) and instance IDs (nearest) so labels stay exact;
    the carve (inter_sheet_boundary) then separates the new junctions automatically. Jacobian>0 by construction:
    the displacement is a low-frequency field (sigma=`smooth`) with bounded amplitude (max_shift << smooth), so
    |grad u| << 1 -> strictly monotonic, no folding (same diffeomorphic guarantee as the gap-collapse warp)."""
    v = np.moveaxis(vol, axis, 0).astype(np.float32)
    m = np.moveaxis(msk, axis, 0)
    Z, Y, X = v.shape

    def field():
        f = ndi.gaussian_filter(rng.standard_normal((Z, Y, X)).astype(np.float32), smooth)
        return f / (np.abs(f).max() + 1e-6) * max_shift           # bounded amplitude -> Jacobian>0

    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    coords = np.stack([zz.astype(np.float32), yy + field(), xx + field()])   # shift only IN-PLANE (axes 1,2)
    vw = ndi.map_coordinates(v, coords, order=1, mode="nearest")
    mw = ndi.map_coordinates(m, coords, order=0, mode="nearest")
    return np.moveaxis(vw, 0, axis), np.moveaxis(mw, 0, axis)


def svf_warp(vol, msk, axis, rng, amp=10.0, octaves=((40.0, 1.0), (10.0, 0.5), (5.0, 0.20)), n_sq=6):
    """Rank-2 deformation: in-plane diffeomorphism via a MULTI-OCTAVE stationary velocity field, exponentiated by
    scaling-and-squaring. Adds the finer spatial scale the pipeline lacks (gap-collapse sigma=10, shear sigma=40 only),
    giving sharper/finer convergent creases + junctions than shear. Same warp on CT (order1) and instance IDs (order0).
    In-plane only (axis 0 = cross-sheet has zero velocity) so it reshapes/converges sheets WITHOUT undoing fusion.
    JACOBIAN>0 verified: octaves+n_sq tuned so the discretized warp folds 0% up to amp~15 (min det +0.08; the
    workflow's sigma=3/n_sq=5 folded ~0.001% from discretization -- this sigma=5,w0.2,n_sq=6 is the folding-free fix).
    Keep amp <= ~15 (max displacement ~14.6 vox, within real normal-curvature p90 ~30deg)."""
    v = np.moveaxis(vol, axis, 0).astype(np.float32)
    m = np.moveaxis(msk, axis, 0)
    Z, Y, X = v.shape

    def field():
        f = sum(ndi.gaussian_filter(rng.standard_normal((Z, Y, X)).astype(np.float32), s) * w for s, w in octaves)
        return f / (np.abs(f).max() + 1e-6) * amp

    uy = field() / (2 ** n_sq); ux = field() / (2 ** n_sq)          # small initial velocity step (|grad|<<1)
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    zz = zz.astype(np.float32); yy = yy.astype(np.float32); xx = xx.astype(np.float32)
    for _ in range(n_sq):                                          # scaling-and-squaring: phi <- phi o phi
        cy = yy + uy; cx = xx + ux
        uy = uy + ndi.map_coordinates(uy, [zz, cy, cx], order=1, mode="nearest")
        ux = ux + ndi.map_coordinates(ux, [zz, cy, cx], order=1, mode="nearest")
    coords = [zz, yy + uy, xx + ux]
    vw = ndi.map_coordinates(v, coords, order=1, mode="nearest")
    mw = ndi.map_coordinates(m, coords, order=0, mode="nearest")
    return np.moveaxis(vw, 0, axis), np.moveaxis(mw, 0, axis)


def restore_grain_at_strain(ct, disp, rng):
    """Order-1 map_coordinates LOW-PASSES grain proportionally to local strain -> a deformation-correlated TELL (the
    high-strain leakage probe caught AUC~0.97 on the raw crease). Restore it: at high-strain papyrus voxels, ADD grain
    whose radial spectrum matches the cube's own, scaled by the LOCAL high-frequency deficit (so untouched regions are
    not over-textured). Collapses the strain<->grain correlation back toward chance."""
    ct = ct.astype(np.float32)
    pap = ct > np.percentile(ct, 50)
    if not pap.any():
        return ct
    strain = np.zeros(ct.shape, np.float32)
    for i in range(3):
        for gj in np.gradient(disp[i]):
            strain += gj * gj
    strain = np.sqrt(strain)
    hi = ct - ndi.gaussian_filter(ct, 0.8)                                     # the cube's high-freq grain
    lo = pap & (strain <= np.percentile(strain[pap], 40))                      # low-strain papyrus = intact grain
    nstd = float(np.std(hi[lo if lo.sum() > 100 else pap]))
    freq, nps = radial_nps(hi)                                                 # cube's own grain spectrum
    local = np.sqrt(np.maximum(ndi.gaussian_filter(hi * hi, 3.0), 0.0))        # local HF energy (std-like)
    deficit = np.sqrt(np.clip(nstd ** 2 - local ** 2, 0.0, None))             # how much grain was lost locally
    hs = np.percentile(strain[pap], 60); hi_s = np.percentile(strain[pap], 95)
    w = np.clip((strain - hs) / (hi_s - hs + 1e-6), 0.0, 1.0) * pap            # weight: only high-strain papyrus
    grain = colored_noise(ct.shape, freq, nps, rng)
    return np.clip(ct + grain * deficit * w, 0, 255)


def crease_warp(vol, msk, axis, rng, theta_deg=18.0, beta_deg=33.0, band_w=18.0, feather=1.5,
                conjugate=False, shear_axis=None, return_disp=False, restore_grain=False, **_):
    # restore_grain default OFF: the high-strain leakage AUC (~0.95) is LEGITIMATE kink-hinge geometry, NOT a resampling
    # tell -- proven by order-1 vs order-3 resampling giving the SAME AUC (0.950 vs 0.943). So no synthetic grain needed
    # (it would add its own tell); the real moved voxels are the most honest. (ICLR 2024 "How I Warped Your Noise" notes
    # the dissipation phenomenon in general, but it is negligible at our strains.)
    """Rank-3 (SHEAR formulation -- replaces the rotation-SVF that folded at theta>15deg). Sharp brittle kink-band as a
    LOCALIZED IN-PLANE SHEAR: displacement u = A*rho(s) with A PARALLEL to the band (A perp n_hat) and rho a smooth bump
    across the band-normal coord s. Then grad u = A (x) (rho'*n_hat) is rank-1 with A perp n_hat, so
    det(I+grad u) == 1 EXACTLY for ANY angle -> no lever-arm folding, no scaling-and-squaring needed (it's a
    diffeomorphism by construction). Layers crossing the band kink by ~atan(tan theta). Carbonized papyrus kinks
    (band inclination beta~32-35deg), not bends. Same warp on CT (order1) + instance IDs (order0). conjugate sums two
    rank-1 terms (cross-term can perturb det) -> default False (single kink = guaranteed det==1); GATE if enabled."""
    theta = np.deg2rad(theta_deg)
    shape = vol.shape
    if shear_axis is None:
        shear_axis = (axis + 1) % 3
    grids = [g.astype(np.float32) for g in np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")]
    ctr = [s / 2.0 for s in shape]

    def kink(beta_signed, s0):
        b = np.deg2rad(beta_signed)
        n = np.zeros(3, np.float32); n[axis] = np.cos(b); n[shear_axis] = np.sin(b)          # band normal
        adir = np.zeros(3, np.float32); adir[axis] = -np.sin(b); adir[shear_axis] = np.cos(b)  # PARALLEL to band (perp n)
        s = (grids[axis] - ctr[axis]) * n[axis] + (grids[shear_axis] - ctr[shear_axis]) * n[shear_axis] - s0
        rho = np.exp(-0.5 * (s / (band_w * 0.5)) ** 2)                                        # smooth bump -> chevron kink
        amag = band_w * np.tan(theta)
        return [adir[d] * amag * rho for d in range(3)]

    disp = kink(beta_deg, float(rng.uniform(-shape[axis] * 0.2, shape[axis] * 0.2)))
    if conjugate:
        d2 = kink(-beta_deg, float(rng.uniform(-shape[axis] * 0.2, shape[axis] * 0.2)))
        disp = [disp[d] + d2[d] for d in range(3)]
    fc = np.stack([grids[d] + disp[d] for d in range(3)])
    vw = ndi.map_coordinates(vol.astype(np.float32), fc, order=1, mode="nearest")
    mw = ndi.map_coordinates(msk, fc, order=0, mode="nearest").astype(msk.dtype)
    if restore_grain:                                            # kill the strain<->grain low-pass tell (leakage gate)
        vw = restore_grain_at_strain(vw, disp, rng)
    if return_disp:
        return vw, mw, disp
    return vw, mw


def jacobian_folding(disp):
    """Folding diagnostic for a displacement field (list [dz,dy,dx]): fraction of voxels with det(I+grad u)<=0 and the
    min det. Gate for crease_warp/svf_warp = 0% folding (discrete scaling-and-squaring is only approximately diffeo)."""
    g = [np.gradient(disp[i]) for i in range(3)]                  # g[i][j] = d(disp_i)/d(axis_j), central differences
    J = np.empty(disp[0].shape + (3, 3), np.float32)
    for i in range(3):
        for j in range(3):
            J[..., i, j] = g[i][j] + (1.0 if i == j else 0.0)
    det = np.linalg.det(J)
    return float((det <= 0).mean()), float(det.min())


def densest_crop(vol, msk, size, air_thresh):
    out_v = np.zeros((size, size, size), vol.dtype); out_m = np.zeros((size, size, size), msk.dtype)
    sl_s, sl_d = [], []; pap = (vol >= air_thresh)
    for a in range(3):
        n = msk.shape[a]
        if n <= size:
            d0 = (size - n) // 2; sl_s.append(slice(0, n)); sl_d.append(slice(d0, d0 + n)); continue
        prof = pap.sum(axis=tuple(i for i in range(3) if i != a)).astype(np.float64)
        csum = np.concatenate([[0], np.cumsum(prof)]); win = csum[size:] - csum[:-size]
        s0 = int(np.argmax(win)); sl_s.append(slice(s0, s0 + size)); sl_d.append(slice(0, size))
    out_v[tuple(sl_d)] = vol[tuple(sl_s)]; out_m[tuple(sl_d)] = msk[tuple(sl_s)]
    return out_v, out_m


def random_crop(vol, msk, size, rng):
    """Uniform-random out_size^3 window (the easy-floor companion to densest_crop).

    densest_crop maximizes papyrus fraction, i.e. it deliberately seeks the TIGHTEST region of the cube. That is
    exactly right for the fused/tight strata, and exactly wrong for the easy floor, whose whole job is to show the
    model REAL open gaps (6/11/35 vox) for split protection -- picking the densest window there biases the gap
    distribution back down and re-defeats the floor even after gap_collapse's keep_air_min fix."""
    out_v = np.zeros((size, size, size), vol.dtype); out_m = np.zeros((size, size, size), msk.dtype)
    sl_s, sl_d = [], []
    for a in range(3):
        n = msk.shape[a]
        if n <= size:
            d0 = (size - n) // 2; sl_s.append(slice(0, n)); sl_d.append(slice(d0, d0 + n)); continue
        s0 = int(rng.integers(0, n - size + 1)); sl_s.append(slice(s0, s0 + size)); sl_d.append(slice(0, size))
    out_v[tuple(sl_d)] = vol[tuple(sl_s)]; out_m[tuple(sl_d)] = msk[tuple(sl_s)]
    return out_v, out_m


def inter_sheet_boundary(inst):
    """Voxels where a papyrus voxel neighbours a DIFFERENT non-zero instance id (the fused contact surface)."""
    inst = inst.astype(np.int32); bnd = np.zeros(inst.shape, bool)
    for ax in range(3):
        for sh in (1, -1):
            nb = np.roll(inst, sh, axis=ax)
            bnd |= (inst > 0) & (nb > 0) & (inst != nb)
    return bnd


def radial_nps(residual, n_bins=48):
    """3D radially-averaged power spectrum (noise power spectrum) of a high-pass residual volume.
    Returns (freq in cycles/voxel over [0,0.5], nps). Used to COLOR the synthetic contact grain so it carries the
    cube's OWN spatial-frequency signature -- not a fixed Gaussian blob (the old, detectable tell)."""
    r = (residual - residual.mean()).astype(np.float32)
    P = np.abs(np.fft.rfftn(r)) ** 2
    fz = np.fft.fftfreq(r.shape[0]); fy = np.fft.fftfreq(r.shape[1]); fx = np.fft.rfftfreq(r.shape[2])
    kr = np.sqrt(fz[:, None, None] ** 2 + fy[None, :, None] ** 2 + fx[None, None, :] ** 2)
    bins = np.linspace(0.0, 0.5, n_bins + 1)
    idx = np.clip(np.digitize(kr.ravel(), bins) - 1, 0, n_bins - 1)
    nps = np.bincount(idx, weights=P.ravel(), minlength=n_bins)
    cnt = np.bincount(idx, minlength=n_bins)
    nps = nps / np.maximum(cnt, 1)
    return 0.5 * (bins[:-1] + bins[1:]), nps


def colored_noise(shape, freq, nps, rng):
    """White Gaussian noise COLORED to the radial NPS profile (isotropic), returned with unit variance. Filtering
    in the Fourier domain decouples the output shape from the patch the NPS was measured on."""
    fz = np.fft.fftfreq(shape[0]); fy = np.fft.fftfreq(shape[1]); fx = np.fft.rfftfreq(shape[2])
    kr = np.sqrt(fz[:, None, None] ** 2 + fy[None, :, None] ** 2 + fx[None, None, :] ** 2)
    amp = np.interp(kr, freq, np.sqrt(np.maximum(nps, 0.0)), left=float(np.sqrt(max(nps[0], 0.0))), right=0.0)
    F = np.fft.rfftn(rng.standard_normal(shape).astype(np.float32)) * amp
    g = np.fft.irfftn(F, s=shape, axes=tuple(range(len(shape)))).astype(np.float32)
    g -= g.mean()
    return g / (g.std() + 1e-8)


def flat_grain_model(ct, contact_band, rng, n_boxes=6, box=32):
    """Estimate the cube's OWN grain (std + radial NPS) from FLAT papyrus INTERIOR boxes -- eroded away from sheet
    edges, air and the new contact -- so the model captures real scan noise, NOT edge/structure power (which would
    make the synthetic 'grain' carry fake edges). Falls back to whole-volume residual if no clean interior exists."""
    hi = ct - ndi.gaussian_filter(ct, 0.8)
    pap = ct > np.percentile(ct, 50)
    interior = ndi.binary_erosion(pap & ~ndi.binary_dilation(contact_band, iterations=3), iterations=2)
    half = box // 2
    pts = np.argwhere(interior)
    ok = pts[np.all((pts >= half) & (pts < (np.array(ct.shape) - half)), axis=1)] if len(pts) else pts
    npss, stds = [], []
    if len(ok):
        sel = ok[rng.choice(len(ok), size=min(n_boxes, len(ok)), replace=False)]
        for z, y, x in sel:
            p = hi[z - half:z + half, y - half:y + half, x - half:x + half]
            f, nps = radial_nps(p); npss.append(nps); stds.append(float(p.std()))
    if npss:
        return float(np.mean(stds)), f, np.mean(npss, axis=0)
    far = pap & ~ndi.binary_dilation(contact_band, iterations=4)
    f, nps = radial_nps(hi)
    return float(np.std(hi[far if far.any() else pap])), f, nps


def local_contact_rescan(ct, contact_band, rng, psf_sigma=0.9, noise_scale=1.0, feather=1.5):
    """Make the NEW contact look really scanned WITHOUT leaving a detectable texture patch (the honesty fix).
    Lessons from the leakage probe: (1) the warped voxels already carry the cube's REAL grain -- destroy as little
    of it as possible; (2) at full fusion the inter-sheet band is a large fraction of the volume, so a wide blur
    ruins real texture (the old bug, and worse if dilated). So:
      - partial-volume Gaussian PSF only on a THIN, FEATHERED seam (the exact contact +/- ~`feather` vox);
      - the blur removes local grain -> restore it with additive grain whose STD *and RADIAL SPECTRUM* match the
        cube's own FLAT-INTERIOR grain (flat_grain_model), so noise level+spectrum stay native by construction;
      - everything outside the thin feathered seam is untouched real texture.
    Validate with leakage_probe.py: NEW should add ~0 over the no-rescan geometric floor. OPT: bbox-local filters."""
    ct = ct.astype(np.float32)
    if not contact_band.any():
        return np.clip(ct, 0, 255)
    nstd, freq, nps = flat_grain_model(ct, contact_band, rng)
    nstd *= noise_scale
    out = ct.copy()
    fpad = int(np.ceil(3 * max(psf_sigma, feather))) + 2
    idx = np.argwhere(ndi.binary_dilation(contact_band, iterations=fpad))
    lo = np.maximum(idx.min(0), 0); hg = np.minimum(idx.max(0) + 1, np.array(ct.shape))
    sl = tuple(slice(int(lo[i]), int(hg[i])) for i in range(3))
    sub = ct[sl]
    d = ndi.distance_transform_edt(~contact_band[sl])                  # 0 ON the seam, grows outward
    alpha = np.clip(1.0 - d / max(feather, 1e-3), 0.0, 1.0)            # 1 on seam, feathered ramp to 0 in ~`feather`
    structure = ndi.gaussian_filter(sub, psf_sigma)                    # partial-volume low-pass at the seam
    grain = colored_noise(sub.shape, freq, nps, rng) * nstd           # native-spectrum, native-std grain
    rescanned = structure + grain                                     # restores the native noise the blur removed
    out[sl] = (1.0 - alpha) * sub + alpha * rescanned
    return np.clip(out, 0, 255)


def self_affine_surface(shape2d, hurst, rms, q_lo, q_hi, rng):
    """Rank-2 primitive: isotropic self-affine rough surface via Fourier filtering. 2D PSD C(q) ~ q^(-2(H+1)) ->
    amplitude ~ q^(-(H+1)) (the 1D-vs-2D off-by-one: a 1D trace falls as q^(-1-2H)). Band-limited [q_lo,q_hi] with a
    plateau below q_lo; empirical RMS normalization (robust). Fibrous/paper H~0.7-0.8. q in rad/voxel, q_hi<=pi."""
    ny, nx = shape2d; H = float(hurst)
    qx = 2 * np.pi * np.fft.rfftfreq(nx); qy = 2 * np.pi * np.fft.fftfreq(ny)
    QX, QY = np.meshgrid(qx, qy); q = np.hypot(QX, QY)
    amp = np.zeros_like(q); band = (q >= q_lo) & (q <= q_hi); amp[band] = q[band] ** (-(H + 1.0))
    roll = (q < q_lo) & (q > 0); amp[roll] = q_lo ** (-(H + 1.0)); amp[0, 0] = 0.0
    spec = amp * (rng.normal(size=amp.shape) + 1j * rng.normal(size=amp.shape))
    h = np.fft.irfft2(spec, s=(ny, nx)); h -= h.mean(); h *= (rms / (h.std() + 1e-8))
    return h.astype(np.float32)


def gap_collapse_contact(vol, msk, axis, load, rng, air_thresh, hurst=0.75, rms=9.0, q_lo=None, q_hi=None):
    """Rank-2: replace the SCALAR keep_air with a self-affine PATCHY-CONTACT gap field (Persson/GW). Per in-plane
    column, collapse its air run to a residual gap = max(u_bar - h, 0) voxels (0 = contact/fuse), where h is a 2D
    self-affine height field and u_bar is set by `load` (= target contact-area fraction). Gives few kiss-points ->
    coalescing patches -> conformal contact as load rises, with open asperity valleys PRESERVED as real gap (split
    protection). Per-column cumsum warp keeps it diffeomorphic. Calibrate rms+load so residual-gap pcts hit 6/11/35."""
    v = np.moveaxis(vol, axis, -1).astype(np.float32); m = np.moveaxis(msk, axis, -1)
    shp = v.shape; N = shp[-1]
    if q_lo is None: q_lo = 2 * np.pi / max(shp[0], shp[1])
    if q_hi is None: q_hi = np.pi
    air = v < air_thresh
    pap = ~air                                                                 # restrict the gap budget to INTERIOR
    seen_pre = np.cumsum(pap, axis=-1) > 0                                     # air between first & last papyrus voxel
    seen_post = np.cumsum(pap[..., ::-1], axis=-1)[..., ::-1] > 0             # (the real inter-sheet gaps), NOT cube-end air
    interior_air = air & seen_pre & seen_post
    h = self_affine_surface((shp[0], shp[1]), hurst, rms, q_lo, q_hi, rng)     # 2D self-affine in-plane contact field
    u_bar = float(np.percentile(h, 100 * (1 - load)))                          # contact where field>u_bar; P~load
    # PER-INTERFACE bearing-area contact: label each interior air RUN (connect only along the cross-sheet/last axis) and
    # add a per-run offset to the column field, so WHICH interface in a column closes is DECORRELATED (not all-or-nothing
    # per column -> avoids the unrealistic accordion tell). Contacting runs FUSE; open runs keep their REAL gap (6/11/35).
    runstruct = np.zeros((3, 3, 3), int); runstruct[1, 1, :] = 1
    run_lab, nruns = ndi.label(interior_air, structure=runstruct)
    off = rng.normal(0.0, 0.7 * float(h.std() + 1e-6), nruns + 1).astype(np.float32); off[0] = 0.0
    contact_vox = interior_air & ((h[..., None] + off[run_lab]) > u_bar)       # per-interface patchy contact decision
    step = np.ones(shp, np.float32); step[contact_vox] = 0.0                   # end-air & papyrus & open runs untouched
    outpos = np.cumsum(step, axis=-1)
    fv = v.reshape(-1, N); fm = m.reshape(-1, N); fo = outpos.reshape(-1, N)
    src = np.arange(N, dtype=np.float32); bg = float(np.median(v[air])) if air.any() else 0.0
    ov = np.full_like(fv, bg); om = np.zeros_like(fm); grid = np.arange(N, dtype=np.float32)
    for c in range(fv.shape[0]):
        maxo = fo[c][-1]; valid = grid <= maxo
        inp = np.interp(grid, fo[c], src)
        ov[c] = np.where(valid, np.interp(inp, src, fv[c]), bg)
        om[c] = np.where(valid, fm[c][np.clip(np.rint(inp).astype(np.int64), 0, N - 1)], 0)
    return np.moveaxis(ov.reshape(shp), -1, axis), np.moveaxis(om.reshape(shp), -1, axis)


def paganin_lowpass(vol3d, L_vox, Lz_vox=None, pad=True, gpm=True):
    """Rank-1: the CAUSAL merge operator. Real scroll recons are single-distance Paganin phase-retrieved -> a
    Lorentzian low-pass H(k)=1/(1+(2*pi*L*kr)^2) that smears a thin (~6vox) inter-sheet air trough below the air
    threshold so two sheets read as one blob. Applying the SAME filter (L randomized + CLAMPED to the real edge-spread
    band) to synthetic cubes teaches de-merging that transfers by construction. L_vox = sqrt(R*lambda*(delta/beta)/(4*pi))/px
    (plausible 2-12 vox). DC gain = 1 (mean preserved, no ringing -- the Lorentzian kernel is a non-negative double
    exponential). Apply to RECONSTRUCTED intensity (valid: Paganin is LSI, FBP linear). Lz_vox<L_vox (or 0) -> the
    physically in-plane-only blur; isotropic is fine for augmentation robustness."""
    v = vol3d.astype(np.float32)
    Lz = L_vox if Lz_vox is None else Lz_vox
    p = int(np.ceil(4 * max(L_vox, Lz))) if pad else 0          # 4*L tail (e^-4~1.8%) -> enough; ~30% less FFT than 5*L
    if p:
        v = np.pad(v, p, mode="reflect")
    nz, ny, nx = v.shape
    kz = np.fft.fftfreq(nz)[:, None, None]; ky = np.fft.fftfreq(ny)[None, :, None]
    kx = np.fft.rfftfreq(nx)[None, None, :]                         # last axis -> rfftfreq
    if gpm:                                                         # Generalised Paganin Method: discrete-Laplacian symbol
        # 2(1-cos(2*pi*k)) ~ (2*pi*k)^2 at low freq but stays recon-faithful near Nyquist (Paganin 2020 J.Opt 22:115607)
        arg2 = (Lz ** 2 * (2 - 2 * np.cos(2 * np.pi * kz)) + L_vox ** 2 * (2 - 2 * np.cos(2 * np.pi * ky))
                + L_vox ** 2 * (2 - 2 * np.cos(2 * np.pi * kx)))
    else:                                                          # pure-Lorentzian (continuous k^2)
        arg2 = (2 * np.pi) ** 2 * ((Lz * kz) ** 2 + (L_vox * ky) ** 2 + (L_vox * kx) ** 2)
    H = (1.0 / (1.0 + arg2)).astype(np.float32)                    # H(0)=1 exactly -> mean preserved, no ring
    out = np.fft.irfftn(np.fft.rfftn(v) * H, s=v.shape, axes=(0, 1, 2)).astype(np.float32)
    if p:
        out = out[p:-p, p:-p, p:-p]
    return out


def partition_labels(fiber, inst):
    """Assign EVERY fibre voxel to its NEAREST instance.

    Why this is mandatory (and not cosmetic): the source instance masks are SPARSE, and the per-column cumsum warp
    in gap_collapse leaves a 1-voxel 0-label seam at every fully-collapsed junction (verified: at keep_air=0 a
    column reads `.5555555.4444444.` — the CT sheets fuse but the LABELS never touch). Consequently any test of the
    form `inst_a adjacent to inst_b` (inter_sheet_boundary, svpv.interface_sheet) finds ZERO contacts on raw `inst`
    and silently degrades to a no-op. Densifying the labels first is what makes contacts exist at all."""
    idx = ndi.distance_transform_edt(inst == 0, return_indices=True)[1]
    return (inst[tuple(idx)] * (fiber > 0)).astype(np.int32)


def to_outputs(ct, inst, air_thresh, sep=2, weight_boundary=8.0, partition=False):
    """Dataset059-format binary fiber (boundary carved) + per-voxel loss-weight map (high on the carved boundary).

    partition=False (legacy): carve only where two LABELED instances directly abut, dilated by sep-1. VERIFIED to
      leave the fiber as ONE 26-connected blob spanning all sheets -- unlabeled fiber + thin/holey carve bridge it,
      so the foreground target does NOT encode separated sheets (the separation is taught only by the up-weighted CE).
    partition=True (C1 fix): assign EVERY fiber voxel to its nearest instance (so unlabeled fiber can't bridge),
      carve the inter-instance ridges dilated by `sep`, giving a target whose connected components actually match the
      sheets. Strictly stronger separation signal; verify per cube with assert_separates()."""
    fiber = (ct >= air_thresh).astype(np.uint8)
    if partition:
        labels = partition_labels(fiber, inst)                   # nearest-instance label for ALL fiber voxels
        bnd = inter_sheet_boundary(labels)
        carve = ndi.binary_dilation(bnd, iterations=sep) & (fiber > 0)   # thicker -> disconnects under 26-conn
    else:
        bnd = inter_sheet_boundary(inst)
        carve = ndi.binary_dilation(bnd, iterations=max(0, sep - 1)) & (fiber > 0)
    fiber[carve] = 0
    weight = np.ones(ct.shape, np.float32)
    weight[ndi.binary_dilation(bnd, iterations=sep + 1)] = weight_boundary   # up-weight the separation
    return fiber, bnd, weight


def synth(cube_dir, seed, keep_air=None, out_size=192, sep=2, rescan_fn=None, shear=0.0, svf=0.0,
          partition=False, paganin=0.0, crease=0.0, contact_load=0.0, crumple=0.0, artifact=0.0,
          svpv=0.0, weld=0.0, keep_air_min=0.0, crop="densest", smooth_plane=8.0):
    rng = np.random.default_rng(seed)
    vol, msk = load_cube(cube_dir)
    vol, msk = rigid_diversity(vol, msk, rng)
    from skimage.filters import threshold_otsu
    air_thresh = float(threshold_otsu(vol))
    axis = cross_sheet_axis(msk)
    ka = keep_air if keep_air is not None else float(rng.uniform(0.0, 0.3))   # curriculum incl. 0 (fully fused)
    kmin = float(min(keep_air_min, ka))                                       # guard: band must be non-inverted
    if contact_load > 0:                                                      # Rank-2: self-affine patchy-contact gap field
        dv, dm = gap_collapse_contact(vol, msk, axis, float(contact_load), rng, air_thresh)
    else:
        dv, dm = gap_collapse(vol, msk, axis, ka, rng, air_thresh, spatial=True, keep_air_min=kmin,
                              smooth_plane=float(smooth_plane))
    if svf > 0:                                                              # Rank-2 (queued): multi-octave diffeo
        dv, dm = svf_warp(dv, dm, axis, rng, amp=float(svf))
    elif shear > 0:                                                          # 3-sheet arm: tilt/converge sheets
        dv, dm = shear_warp(dv, dm, axis, rng, max_shift=float(shear))
    if crease > 0:                                                           # Rank-3: sharp shear-kink (det==1, composes on top)
        dv, dm = crease_warp(dv, dm, axis, rng, theta_deg=float(crease), conjugate=False)
    if crumple > 0:                                                          # Rank-4: d-cone/ridge crumple-fold (curved crescent contacts)
        from crumple import crumple_warp
        dv, dm = crumple_warp(dv, dm, axis, rng, n_cones=int(rng.integers(1, 4)), amp=float(crumple))
    dv, dm = (random_crop(dv, dm, out_size, rng) if crop == "random"
              else densest_crop(dv, dm, out_size, air_thresh))
    # Rank-6: patchy pyrolysis WELD -- glue a self-affine subset of the REAL open gaps shut (CT only, additive
    # PSF forward model). Must run BEFORE to_outputs so the welded gap becomes fibre and `partition` assigns it to
    # the nearest instance -> the label keeps two sheets with the boundary through the weld's middle.
    if weld > 0:
        from weld import weld_gaps
        dv = weld_gaps(dv, axis, rng, weld_frac=float(weld), air_thresh=air_thresh)
    # Rank-5: SVPV -- re-render the collapse's QUANTIZED (>=1 vox, unphysically sharp) seams as CONTINUOUS
    # sub-voxel gaps. Also BEFORE to_outputs: it fuses the seam, so `partition` must then assign the now-continuous
    # fibre to the nearest instance and carve the boundary through it (that carve IS the supervision).
    if svpv > 0:
        from svpv import svpv_film
        dv = svpv_film(dv, axis, rng, g_max=float(svpv), air_thresh=air_thresh)
    fiber, bnd, weight = to_outputs(dv, dm, air_thresh, sep=sep, partition=partition)
    ct = (rescan_fn or local_contact_rescan)(dv, bnd, rng)               # rescan_fn lets the leakage probe A/B old vs new
    if paganin > 0:                                                      # Rank-1: causal Paganin merge operator (global)
        ct = np.clip(paganin_lowpass(ct, L_vox=float(paganin)), 0, 255)  # GT (fiber/carve) stays = true separation
    if artifact > 0:                                                     # Rank-A: structured rings/streaks (intensity-only, GT untouched)
        from artifacts import inject_artifacts
        ct = inject_artifacts(ct, rng, axis=axis, ring_amp=float(artifact), streak_amp=float(artifact) * 0.8)
    meta = dict(cube=os.path.basename(cube_dir.rstrip("/")), seed=int(seed), axis=axis, keep_air=round(ka, 3),
                keep_air_min=round(kmin, 3), crop=crop,
                air_thresh=round(air_thresh, 1), dens_before=round(100 * (msk != 0).mean(), 1),
                dens_after=round(100 * (fiber != 0).mean(), 1), boundary_vox=int(bnd.sum()),
                crumple=round(float(crumple), 2), artifact=round(float(artifact), 2),
                svpv=round(float(svpv), 3), weld=round(float(weld), 3), crease=round(float(crease), 1),
                shear=round(float(shear), 1), paganin=round(float(paganin), 2))
    return ct.astype(np.uint8), fiber, weight, dm, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", required=True); ap.add_argument("--out_dir", default="/root/surf/data/synth")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--keep_air", type=float, default=None)
    ap.add_argument("--out_size", type=int, default=192); ap.add_argument("--sep", type=int, default=2)
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    import nrrd
    os.makedirs(a.out_dir, exist_ok=True)
    ct, fiber, weight, inst, meta = synth(a.cube, a.seed, a.keep_air, a.out_size, a.sep)
    tag = f"{meta['cube']}_s{a.seed}"
    nrrd.write(os.path.join(a.out_dir, f"{tag}_volume.nrrd"), ct)
    nrrd.write(os.path.join(a.out_dir, f"{tag}_mask.nrrd"), fiber)        # binary {bg,fiber}, carved
    np.save(os.path.join(a.out_dir, f"{tag}_weight.npy"), weight)
    nrrd.write(os.path.join(a.out_dir, f"{tag}_inst.nrrd"), inst.astype(np.uint16))  # kept for verification
    print(f"{tag}: {ct.shape} keep_air {meta['keep_air']} density {meta['dens_before']}%->{meta['dens_after']}% "
          f"boundary_vox {meta['boundary_vox']} -> {a.out_dir}")
    if a.render:
        import subprocess, sys
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "render_layers.py"),
                        "--vol", os.path.join(a.out_dir, f"{tag}_volume.nrrd"),
                        "--lbl", os.path.join(a.out_dir, f"{tag}_inst.nrrd"),
                        "--out", os.path.join(a.out_dir, f"{tag}_panel.png")])


if __name__ == "__main__":
    main()
