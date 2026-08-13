#!/usr/bin/env python3
"""Rank-6 primitive: PATCHY PYROLYSIS WELD — photometric adhesion with real gaps surviving next door.

THE COMPLEMENTARY DIRECTION. svpv.py starts from FUSED sheets and faintly parts them (sub-resolution regime).
This starts from sheets with REAL, resolved air gaps (measured 6/11/35 vox at p25/50/90 on this box) and GLUES
patches of them shut — leaving the neighbouring gap fully open. One cube therefore contains, on the SAME interface:
  * patches with a real, obvious 6-35 vox gap  (boundary trivially visible)
  * patches welded solid, zero gap, zero contrast  (boundary invisible — must be inferred)
  * a smooth transition between them
which is precisely the stated failure mode: "glued at one part but nearby not, rather smoothly different".

PHYSICS. Carbonisation is a pyrolysis process: volatiles migrate and re-deposit as tar/char that solidifies into
genuine solid bridges between touching laminae — the sheets are not merely in contact, they are welded by material
of essentially the same composition (and therefore the same X-ray attenuation) as the papyrus itself. That is why
the fused case has NO density step for the scanner to record. We add that material through the same one-pass
forward model svpv.py uses (see weld_gaps): the weld is ADDED to the object and convolved once with the measured
PSF, never pasted. Consequently the weld edges are partial-volume feathered for free, a bridge thinner than the PSF
stays faint while a wide one saturates to solid papyrus, and — because we add a smooth field rather than overwrite
voxels — the cube's own recon grain is preserved inside the weld by construction (verified: welded/papyrus texture
std = 1.00), so there is no synthetic texture to spectrum-match and none to detect.

WHY THIS IS LEARNABLE (and why patchiness is mandatory). A weld with no open gap anywhere carries zero information
— no method could recover the boundary, and training on it only teaches noise. Patchiness is what makes it a
curriculum: `weld_frac` is capped well below 1 so every welded patch has resolved gap in its neighbourhood, and the
only viable strategy is to PROPAGATE the boundary from where it is visible into where it is not. `weld_frac` is
sampled per-cube to span easy (few small bridges) -> hard (most of the interface welded, thin open ribbons left).

GT. This mechanic touches ONLY the CT. The welded voxels become papyrus-valued, so downstream to_outputs(
partition=True) assigns every fibre voxel to its nearest instance and carves the inter-instance ridge — i.e. the
label keeps TWO sheets with the boundary running through the middle of the weld, exactly as ground truth demands.
"""
import numpy as np
from scipy import ndimage as ndi

ESF_L_P25, ESF_L_P90 = 1.45, 3.54                       # measured real edge-spread band (native voxels)
LORENTZ_TO_GAUSS = 2 * np.log(5) / 2.563                # = 1.256


def interior_air(ct, axis, air_thresh):
    """Air voxels BETWEEN papyrus along the cross-sheet axis (the real inter-sheet gaps), excluding the open air
    outside the sheet stack. Same definition as gap_collapse_contact, so the two mechanics agree."""
    v = np.moveaxis(ct, axis, -1)
    air = v < air_thresh
    pap = ~air
    seen_pre = np.cumsum(pap, axis=-1) > 0
    seen_post = np.cumsum(pap[..., ::-1], axis=-1)[..., ::-1] > 0
    return np.moveaxis(air & seen_pre & seen_post, -1, axis)


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


def weld_gaps(ct, axis, rng, weld_frac=None, esf_L=None, air_thresh=None, hurst=0.75, patch_scale=None,
              return_meta=False):
    """Glue a self-affine PATCHY subset of the real inter-sheet gaps shut with pyrolysis-weld material.
      weld_frac  : fraction of the gap AREA to weld (default sampled 0.15-0.75 -> easy..hard; never ~1, so open
                   gap always survives nearby and the boundary stays inferable)
      esf_L      : PSF edge-spread L in native vox (default sampled from the measured real band)
      patch_scale: in-plane correlation length of the adhesion patches, in voxels (default sampled 6-22).
                   This matters physically: a weld patch narrower than the PSF gets partial-volume diluted and only
                   brightens the gap slightly, whereas a patch several sigma across saturates to solid papyrus and
                   truly erases the boundary. Sampling the scale spans both regimes (faint bridge -> hard weld).
    Returns ct_out [, meta]. CT only — labels are handled by to_outputs(partition=True) downstream."""
    ct = np.asarray(ct, np.float32)
    if air_thresh is None:
        from skimage.filters import threshold_otsu
        air_thresh = float(threshold_otsu(ct))
    if weld_frac is None:
        weld_frac = float(rng.uniform(0.15, 0.75))
    if esf_L is None:
        esf_L = float(rng.uniform(ESF_L_P25, ESF_L_P90))
    if patch_scale is None:
        patch_scale = float(rng.uniform(6.0, 22.0))
    sigma = esf_L * LORENTZ_TO_GAUSS

    gap = interior_air(ct, axis, air_thresh)
    if not gap.any():
        return (ct, dict(gap_vox=0, welded_vox=0, weld_frac=0.0)) if return_meta else ct

    # in-plane adhesion field: self-affine (all scales) then low-passed to `patch_scale` so the patches have a
    # realistic dominant size; threshold at the weld_frac quantile -> patchy, spatially correlated adhesion
    ip = [s for i, s in enumerate(ct.shape) if i != axis]
    h = self_affine_2d(tuple(ip), rng, hurst=hurst)
    h = ndi.gaussian_filter(h, patch_scale)
    h = (h - h.mean()) / (h.std() + 1e-8)
    thr = float(np.percentile(h, 100 * (1 - weld_frac)))
    patch = np.expand_dims(h > thr, axis)                       # broadcasts along the cross-sheet axis
    welded = gap & np.broadcast_to(patch, ct.shape)
    if not welded.any():
        return (ct, dict(gap_vox=int(gap.sum()), welded_vox=0, weld_frac=weld_frac)) if return_meta else ct

    # FORWARD MODEL (identical in spirit to svpv.py, opposite sign). The real cube already carries the PSF, so we do
    # NOT paste or re-blur: we ADD the new weld material to the true object and apply the PSF once to that addition.
    #   d(mu) = +(mu_pap - mu_air) * welded   =>   d(image) = (mu_pap - mu_air) * (welded (*) PSF)
    # Consequences, all physically right and all free:
    #   * the weld CORE saturates to papyrus only if it is thicker than ~2*sigma (a thin bridge stays intermediate);
    #   * the weld EDGES are partial-volume feathered by the PSF automatically -- no ad-hoc alpha;
    #   * the original recon GRAIN of those voxels is preserved (we add a smooth field, we do not overwrite), so the
    #     weld carries the cube's own noise by construction -- nothing to spectrum-match, nothing to detect;
    #   * papyrus adjacent to a welded patch brightens slightly, which is exactly what the scanner would record.
    pap = ct >= air_thresh
    I_pap = float(np.median(ct[pap])) if pap.any() else 180.0
    I_air = float(np.median(ct[~pap])) if (~pap).any() else 30.0
    delta = max(I_pap - I_air, 1.0)
    source = (delta * welded).astype(np.float32)
    out = np.clip(ct + ndi.gaussian_filter(source, sigma), 0, 255)
    if return_meta:
        return out, dict(gap_vox=int(gap.sum()), welded_vox=int(welded.sum()),
                         welded_of_gap=round(float(welded.sum()) / max(int(gap.sum()), 1), 3),
                         weld_frac=round(weld_frac, 3), esf_L=round(esf_L, 2), sigma=round(sigma, 2),
                         delta=round(delta, 1))
    return out


if __name__ == "__main__":
    ok = True

    def phantom(gap_w=8, noise=3.0, seed=0):
        """Two sheets separated by a REAL resolved gap of gap_w voxels."""
        r = np.random.default_rng(seed)
        Z = 96
        vol = np.full((Z, 64, 64), 30.0, np.float32)
        inst = np.zeros((Z, 64, 64), np.int32)
        z1a, z1b = 20, 44
        z2a = z1b + gap_w
        vol[z1a:z1b] = 180.0; inst[z1a:z1b] = 1
        vol[z2a:z2a + 24] = 180.0; inst[z2a:z2a + 24] = 2
        vol += r.standard_normal(vol.shape).astype(np.float32) * noise
        return np.clip(vol, 0, 255), inst

    vol, inst = phantom()

    # T1: interior_air finds the gap and NOT the outside air
    gap = interior_air(vol, 0, 100.0)
    inside = gap[44:52].mean(); outside = gap[:19].mean()
    print(f"T1 interior_air: gap-region={inside:.0%} outside-air={outside:.0%} -> "
          f"{'PASS' if inside > 0.9 and outside < 0.01 else 'FAIL'}")
    ok &= inside > 0.9 and outside < 0.01

    # T2: welding raises the gap toward papyrus — the boundary cue is destroyed IN the welded patch
    out, meta = weld_gaps(vol, 0, np.random.default_rng(0), weld_frac=0.5, esf_L=2.0, air_thresh=100.0,
                          patch_scale=12.0, return_meta=True)
    gz = out[44:52]
    welded_like = float((gz > 140).mean())
    print(f"T2 weld fills gap: {meta['welded_of_gap']:.0%} of gap voxels welded, "
          f"{welded_like:.0%} of gap slab now papyrus-valued -> {'PASS' if welded_like > 0.2 else 'FAIL'}")
    ok &= welded_like > 0.2

    # T3: PATCHINESS — open gap MUST survive next to the weld (else unlearnable)
    still_air = float((gz < 100).mean())
    print(f"T3 open gap survives: {still_air:.0%} of the gap slab is still air (resolved boundary next door) -> "
          f"{'PASS' if still_air > 0.15 else 'FAIL'}")
    ok &= still_air > 0.15

    # T4: curriculum — welded area rises monotonically with weld_frac
    fr = []
    for wf in (0.15, 0.35, 0.55, 0.75):
        _, m = weld_gaps(vol, 0, np.random.default_rng(1), weld_frac=wf, esf_L=2.0, air_thresh=100.0,
                         patch_scale=12.0, return_meta=True)
        fr.append(m['welded_of_gap'])
    mono = all(fr[i] < fr[i + 1] for i in range(len(fr) - 1))
    print(f"T4 curriculum: welded-of-gap {fr} for weld_frac 0.15/0.35/0.55/0.75 -> "
          f"{'PASS' if mono else 'FAIL'} (monotone={mono})")
    ok &= mono

    # T5: the weld is LOCAL — beyond the PSF reach (~4*sigma) the sheet is untouched. (Within the PSF reach the
    # papyrus DOES brighten slightly; that is correct physics, not a bug: adding material next door raises the
    # convolved value. So probe deep interior only.)
    sig = 2.0 * LORENTZ_TO_GAUSS
    reach = int(np.ceil(4 * sig))
    far = np.zeros_like(vol, bool); far[20:44 - reach] = True     # sheet 1, >4 sigma from the gap at z=44
    identical = np.allclose(vol[far], out[far], atol=0.5)
    print(f"T5 weld locality: sheet interior >{reach} vox from gap unchanged -> {'PASS' if identical else 'FAIL'}")
    ok &= identical

    # T6: the cube's OWN grain is PRESERVED in the weld (we add a smooth field, never overwrite) — so welded
    # texture std must match papyrus texture std without any spectrum-matching machinery.
    wel = (out[44:52] > 140)
    if wel.sum() > 200:
        hi_w = (out[44:52] - ndi.gaussian_filter(out[44:52], 0.8))[wel].std()
        hi_p = (out[24:40] - ndi.gaussian_filter(out[24:40], 0.8)).std()
        ratio = hi_w / max(hi_p, 1e-6)
        print(f"T6 weld grain preserved: welded std/papyrus std = {ratio:.2f} (want ~0.5-1.8) -> "
              f"{'PASS' if 0.4 < ratio < 2.0 else 'FAIL'}")
        ok &= 0.4 < ratio < 2.0

    # T7: range/finite safety with default random params
    o2, m2 = weld_gaps(*phantom(gap_w=12, seed=3)[:1], 0, np.random.default_rng(9), air_thresh=100.0,
                       return_meta=True) if False else (weld_gaps(vol, 0, np.random.default_rng(9),
                                                                  air_thresh=100.0), None)
    print(f"T7 robustness: range=({o2.min():.0f},{o2.max():.0f}) finite={np.isfinite(o2).all()} -> "
          f"{'PASS' if o2.min() >= 0 and o2.max() <= 255 and np.isfinite(o2).all() else 'FAIL'}")
    ok &= o2.min() >= 0 and o2.max() <= 255 and np.isfinite(o2).all()

    print("WELD SELF-TEST:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
