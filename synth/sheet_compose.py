#!/usr/bin/env python3
"""SHEET COMPOSER v6 -- deposit real sheets from unrelated cubes with rigid-approach + patchy conforming contact,
rendered as CONTINUOUS surfaces (the v5 family failed visually: integer per-column rasterization + label-noise
thickness made every sheet a jagged Minecraft stack, and each sheet settled onto the previous one's ragged top so
the jaggedness COMPOUNDED. Real sheets are smooth flowing ribbons; nothing in a real slice varies at 1-voxel scale).

v6 representation -- everything is a smooth float field, voxels only appear at the very end:
  bottom surface b(x,y)   float, smooth by construction (flow + smoothed detail; conform is smooth-clipped)
  thickness    t(x,y)     float: label thickness SMOOTHED in-plane (label jitter is annotation noise, not geometry),
                          tapered to 0 over ~8 px at the support boundary (real sheet ends thin out, not cliff),
                          x squeeze ~ U(0.55,0.95) (real compact wraps are compressed thinner than our sources)
  rasterize: per voxel COVERAGE = overlap([z,z+1), [b,b+t)) -> alpha-composited texture (anti-aliased everywhere,
             no terraces), texture sampled with linear interpolation through the sheet's own real layers.

Gate: test_compose.py. Calibration targets: analyze_compact.py on slice z=10192 (fg .63-.66, seams 5/10/34)."""
import os, glob, sys
import numpy as np
from scipy import ndimage as ndi
try:
    from scipy import fft as _sfft
except ImportError:                                       # pragma: no cover
    _sfft = None


def _rfftn(a):
    # workers=1: multi-worker scipy.fft spawns a pthread pool inside FORKED pool children, which
    # intermittently deadlocks (observed: 22/3000 jobs hung idle at the end of a bulk run)
    return _sfft.rfftn(a, workers=1) if _sfft is not None else np.fft.rfftn(a)


def _irfftn(A, shape):
    return (_sfft.irfftn(A, s=shape, workers=1) if _sfft is not None
            else np.fft.irfftn(A, s=shape))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_merge import load_cube, cross_sheet_axis, radial_nps, colored_noise


# ----------------------------------------------------------------------------- harvest
def extract_sheets(vol, msk, axis, min_support=11000, max_sheets=12):
    min_support = int(os.environ.get("SC_MINSUP", min_support))
    v = np.moveaxis(vol, axis, -1).astype(np.float32)
    m = np.moveaxis(msk, axis, -1).astype(np.int32)
    H, W, N = v.shape
    out = []
    ids, counts = np.unique(m[m > 0], return_counts=True)
    zidx = np.arange(N)
    for iid in ids[np.argsort(-counts)][:max_sheets]:
        sel = m == iid
        cols = sel.any(axis=-1)
        if cols.sum() < min_support:
            continue
        cols = ndi.binary_closing(cols, np.ones((7, 7)))
        lab, nl = ndi.label(cols)
        if nl > 1:
            cols = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
        cols = ndi.binary_fill_holes(cols)
        if cols.sum() < min_support:
            continue
        # FULL 3D OBJECT harvest -- the sheet's ENTIRE column run from its lowest to highest voxel, INCLUDING
        # internal air (delaminated plies, holes, wisps). The earlier heightfield representation (top+thickness+
        # texture stack) FLATTENED every real bifurcation into a slab and we then re-synthesized a fake version --
        # exactly backwards. Real structure must be PRESERVED and only pose/contact augmented (user requirement).
        first = np.argmax(sel, axis=-1).astype(np.int32)                 # lowest sheet voxel per column
        # BUNDLE, don't span: harmonized ids connect MULTIPLE wraps, so [first..last] can swallow several wraps
        # plus the real air between them (thick blobs + unsupervised same-id contacts). Keep runs within 5 vox of
        # the first run -- that is a genuinely delaminated ply of THIS sheet (real bifurcation, preserved) -- and
        # cut before farther runs (those are other wraps; they get their own deposit).
        ext = np.zeros((H, W), np.float32)
        flat = sel.reshape(-1, N)
        # BUNDLING 5 -> 12 vox, gated on gap PURITY. The real ply-gap width MODE is 7 vox (~55 um,
        # lamination census 2026-07): a 5-vox cutoff discarded the single most common real
        # delamination as "another wrap", which is why synth had 0.00% strongly-delaminated
        # instances against real 47.7%. 12 covers the real distribution's bulk; the purity check
        # (no OTHER instance inside the gap) is what actually separates "my delaminated ply"
        # from "the neighbouring wrap" -- measured contamination without it: 0.26%.
        BUNDLE = int(os.environ.get("SC_BUNDLE", "12"))
        ofl = ((m > 0) & ~sel).reshape(-1, N)   # OTHER instances (m, not the 2D CC label `lab`)
        efl = ext.reshape(-1)
        ffl = first.reshape(-1)
        n_split = 0
        n_cols = 0
        for c in np.flatnonzero(flat.any(axis=1)):
            col = flat[c]
            d = np.diff(np.concatenate(([0], col.view(np.int8), [0])))
            st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
            e = en[0]
            kept = 1
            for r in range(1, len(st)):
                if st[r] - e <= BUNDLE and not ofl[c, e:st[r]].any():
                    e = en[r]
                    kept += 1
                else:
                    break
            efl[c] = e - ffl[c]
            n_cols += 1
            n_split += int(kept > 1)
        delam_frac = float(n_split) / max(n_cols, 1)
        lab_cols = sel.any(axis=-1)
        holes = cols & ~lab_cols
        nn = ndi.distance_transform_edt(~lab_cols, return_indices=True)[1]
        if holes.any():
            jr = np.random.default_rng(int(iid) * 977 + 13)
            for ax_ in (0, 1):                              # JITTER the inpaint source: exact-nearest copies whole
                j_ = jr.integers(-2, 3, size=nn[ax_].shape)  # columns -> along-sheet smear bands
                nn[ax_] = np.clip(nn[ax_] + np.where(holes, j_, 0), 0, nn[ax_].shape[ax_] - 1)
            bad = ~lab_cols[tuple(nn)] & holes               # jitter may land on unlabeled -> re-take exact nearest
            if bad.any():
                nn0 = ndi.distance_transform_edt(~lab_cols, return_indices=True)[1]
                nn[0] = np.where(bad, nn0[0], nn[0]); nn[1] = np.where(bad, nn0[1], nn[1])
            first = first[tuple(nn)]; ext = ext[tuple(nn)]
        # SMOOTH-ENVELOPE box (v13): anchor the box at the SMOOTHED first-surface, padded 3 vox into real air.
        # v12 anchored at raw per-column `first`, so box contents jumped column-to-column = jagged. With a smooth
        # anchor the box edge lies in AIR and the VISIBLE surface is the sheet's genuine papyrus/air transition
        # copied from the source -- placement smooth (v9), material real (v12).
        PAD = 2
        first_s = ndi.gaussian_filter(first.astype(np.float32), 3.0)
        emax = int(np.clip(np.percentile(ext[cols], 99), 4, 30)) + 2 * PAD
        ctbox = np.zeros((H, W, emax), np.float32)
        papbox = np.zeros((H, W, emax), bool)
        resid = np.zeros((H, W), np.float32)
        src_i = np.where(holes, nn[0], np.arange(H)[:, None])
        src_j = np.where(holes, nn[1], np.arange(W)[None, :])
        for (i, j) in np.argwhere(cols):
            si_, sj_ = int(src_i[i, j]), int(src_j[i, j])
            z0 = int(round(float(first_s[i, j]))) - PAD
            z0 = max(0, min(z0, N - emax))
            ctbox[i, j] = v[si_, sj_, z0:z0 + emax]
            papbox[i, j] = sel[si_, sj_, z0:z0 + emax]
            resid[i, j] = float(first_s[i, j]) - PAD - z0        # subvoxel anchor residual (kills content steps)
        fthr = float(np.percentile(v, 55))
        papbox |= ndi.binary_dilation(papbox, np.ones((3, 3, 3), bool)) & (ctbox > fthr)
        papbox |= ndi.binary_dilation(papbox, np.ones((3, 3, 3), bool)) & (ctbox > fthr)
        papbox = ndi.binary_opening(papbox, np.ones((3, 3, 1), bool))   # kill 1-vox threshold spurs
        papbox = ndi.binary_closing(papbox, np.ones((3, 3, 1), bool))   # IN-PLANE only: the old (1,3,3)
        #          closed DEPTH gaps <=2 vox, blunt-ending every delamination lens taper (advisor D2)
        # CORE-FILL box (Fable-5 audit issue 1: every synthetic contact carried a dim line, so true zero-evidence
        # fusion -- the regime where mergers are catastrophic -- was unreachable). The harvested run keeps each
        # sheet's air-facing DIM skin: partial-volume ramp frozen from the SOURCE's air gap. At a real invisible
        # fusion two dense cores press together with no air, so the boundary reads bright, no ramp. ctcore is the
        # run with skin/pad voxels replaced by the NEAREST real interior-core voxel (papyrus eroded 1 off the air
        # boundary along depth) -- real material relocated to close the gap, exactly as pose augmentation relocates
        # it. Blended in only at high-fusion patches at deposit; touching-but-not-fused seams keep the real skin.
        core_m = ndi.binary_erosion(papbox, np.ones((1, 1, 3), bool))    # interior: drop the 1-vox depth skin
        if core_m.any():
            ci_ = ndi.distance_transform_edt(~core_m, return_indices=True)[1]
            # JITTER the core-fill source (leakage probe: contact-zone AUC +0.079 over real): an
            # exact-nearest map is a Voronoi copy, so fused seams carried column-replicated
            # texture a patch classifier can spot. Jitter in-plane, re-take exact nearest where
            # the jittered index lands off-core.
            # exact-nearest core fill: BOTH jitter variants (per-voxel and smooth-field) measured
            # WORSE on the v2.1 probe -- the copy's spatial rearrangement is itself the tell, and
            # exact-nearest minimizes it (0.771 old-bank vs 0.87+ jittered banks).
            ctcore = ctbox[ci_[0], ci_[1], ci_[2]].astype(np.float32)
        else:
            ctcore = ctbox.copy()
        # detail of the REAL bottom surface (preserved; only the low-frequency trend gets replaced at re-pose)
        bot_det = np.clip(first_s - ndi.gaussian_filter(first_s, 25), -3, 3)
        ext_det = np.clip(ext - ndi.gaussian_filter(ext, 15.0), -4.0, 4.0)   # real short-scale thickness wander
        # the ribbon masks are harvested SLICE-WISE, so their surfaces carry single-row stair-step;
        # at full ribbon scale it rendered as saw-tooth striping on every band edge. A 3x3 median
        # kills the single-row outliers while keeping the genuine thickness wander.
        bot_det = ndi.median_filter(bot_det, 5); ext_det = ndi.median_filter(ext_det, 3)
        resid = ndi.median_filter(resid, 3)
        ext_env = ndi.gaussian_filter(ext, 3.0) + 2 * PAD
        out.append(dict(bot_det=bot_det, resid=resid, ext_det=ext_det, ext=np.clip(ext_env, 1.0, float(emax)),
                        ctbox=ctbox, ctcore=ctcore, papbox=papbox, sup=cols, emax=emax, id=int(iid),
                        delam=delam_frac, papf=papf_from_box(papbox)))
    air = v[m == 0]
    air = air[air > 0]        # ribbon exports zero the out-of-mask region; zeros are padding,
    #                           not air -- median of the filtered-empty set was NaN and one NaN
    #                           air_med NaN'd the whole background (observed: black cubes)
    air = air[air < np.percentile(air, 60)] if air.size else np.array([60.0])
    block = None
    airmask = (m == 0) & (v < np.percentile(v, 55))
    score = ndi.uniform_filter(airmask.astype(np.float32), 64)
    yx = np.unravel_index(np.argmax(score), score.shape)
    y, x, z = (int(np.clip(c - 32, 0, dim - 64)) for c, dim in zip(yx, v.shape))
    block = v[y:y + 64, x:x + 64, z:z + 64].copy()                 # REAL air texture (best window, always)
    if air.size < 50:
        air = np.array([60.0], np.float32)
    return out, dict(air_med=float(np.median(air)), air_std=float(air.std()), block=block)


def papf_from_box(papbox):
    """Sub-voxel depth occupancy from 1-D signed distance (vectorized two-pass)."""
    E1 = papbox.shape[-1]
    ar = np.arange(E1, dtype=np.float32)
    INF = 1e6

    def _d_to(maskT):
        idx = np.where(maskT, ar, -INF)
        fwd = np.maximum.accumulate(idx, axis=-1)
        idxb = np.where(maskT, ar, INF)
        bwd = np.minimum.accumulate(idxb[..., ::-1], axis=-1)[..., ::-1]
        return np.minimum(ar - fwd, bwd - ar)

    return np.clip(0.5 + np.where(papbox, _d_to(~papbox), -_d_to(papbox)),
                   0.0, 1.0).astype(np.float32)


def lerp512(x, xq, yq):
    """np.interp on monotone knots without the float64 upcast (2.3 s/cube profiled): float32
    searchsorted + lerp, with np.interp end-clamping semantics."""
    xq = np.asarray(xq, np.float32)
    yq = np.asarray(yq, np.float32)
    xr = np.asarray(x, np.float32).ravel()
    idx = np.clip(np.searchsorted(xq, xr), 1, xq.size - 1)
    x0 = xq[idx - 1]
    w = np.clip((xr - x0) / np.maximum(xq[idx] - x0, 1e-6), 0.0, 1.0)
    return (yq[idx - 1] * (1 - w) + yq[idx] * w).reshape(np.shape(x)).astype(np.float32)


def nanfill(f):
    if not np.isnan(f).any():
        return f
    idx = ndi.distance_transform_edt(np.isnan(f), return_indices=True)[1]
    return f[tuple(idx)]


# ----------------------------------------------------------------------------- re-pose
def repose(sheet, hw, rng, flow, squeeze=(0.48, 0.78), winding=False, supmin=0, crease=None,
           spanmin=None):
    """Pose-only augmentation of a REAL 3D sheet object: rotation/flip/crop + replace the LOW-frequency trend of
    its bottom surface with a fresh profile. Everything real rides along untouched: internal delaminated plies,
    holes, wisps, real thickness variation, the real bottom-surface detail (bot_det), the real texture -- no
    physical property is synthesized, only WHERE the sheet sits."""
    H, W = hw
    # GATES-FIRST restructure (profile: the 3D edge-pads + nanfills ran for all 371 attempts while
    # only ~140 were accepted). Only sup (2D bool) is transformed/padded/cropped for the gates;
    # accepted poses gather every other field directly through clipped index maps -- identical to
    # pad(edge)+crop without ever materializing the padded arrays.
    sup0 = sheet["sup"]
    k = int(rng.integers(0, 4))
    sup_t = np.rot90(sup0, k, (0, 1)) if k else sup0
    do_flip = rng.random() < 0.5
    if do_flip:
        sup_t = np.flip(sup_t, 0)
    h0, w0 = sup_t.shape
    ph = max(0, H - h0); pw = max(0, W - w0)
    ph0, pw0 = ph // 2, pw // 2
    hp, wp = h0 + ph, w0 + pw
    if ph or pw:
        sup_p = np.zeros((hp, wp), bool)
        sup_p[ph0:ph0 + h0, pw0:pw0 + w0] = sup_t
    else:
        sup_p = sup_t
    best, best_cov = None, -1.0
    for _ in range(10):
        oy = int(rng.integers(0, max(hp - H, 0) + 1)); ox = int(rng.integers(0, max(wp - W, 0) + 1))
        c = float(sup_p[oy:oy + H, ox:ox + W].mean()) if (hp >= H and wp >= W) else 0.0
        if c > best_cov:
            best, best_cov = (oy, ox), c
        if c > 0.97:
            break
    oy, ox = best if best else (0, 0)
    sup = np.ascontiguousarray(sup_p[oy:oy + H, ox:ox + W])
    if supmin and int(sup.sum()) < int(supmin):
        return dict(sup=sup)
    if spanmin is not None and float(sup.mean()) < 0.92 and             (float(sup.mean()) < spanmin or rng.random() < 0.75):
        return dict(sup=sup)
    # accepted: row/col gather maps composing pad-offset + crop in the TRANSFORMED frame, then one
    # gather per field. Edge-mode = clipped indices; constant-mode = zeroed out-of-range rows.
    ri = np.arange(H) + oy - ph0
    ci = np.arange(W) + ox - pw0
    inb = ((ri >= 0) & (ri < h0))[:, None] & ((ci >= 0) & (ci < w0))[None, :]
    ric = np.clip(ri, 0, h0 - 1)
    cic = np.clip(ci, 0, w0 - 1)

    def _gat(a, const=False):
        a = np.rot90(a, k, (0, 1)) if k else a
        if do_flip:
            a = np.flip(a, 0)
        out = a[np.ix_(ric, cic)]
        if const and not inb.all():
            out[~inb] = 0
        return out

    bot_det = nanfill(_gat(sheet["bot_det"]))
    ext = nanfill(_gat(sheet["ext"]))
    resid = nanfill(_gat(sheet["resid"]))
    ext_det = nanfill(_gat(sheet["ext_det"]))
    ctbox = _gat(sheet["ctbox"])
    ctcore = _gat(sheet["ctcore"])
    papbox = _gat(sheet["papbox"], const=True)
    papf = _gat(sheet["papf"], const=True) if sheet.get("papf") is not None else None
    # NO content inpaint (advisor R1-b): a nearest-neighbour index map is a Voronoi diagram, and
    # copying whole depth stacks into its cells paints piecewise-constant LABELED slabs with
    # straight polygon edges (up to 4% of 36864 = 1475 columns slipped through the old >=0.96
    # gate -- the texture-poor mid-grey rectangles in gate15). A hole in the donor support is a
    # real torn end: deposit it as genuine air, exactly what the partial-sheet path already does.
    # s_top/s_mat stay sane across holes via the fill-holes + nearest-top spanning at deposit.
    # POSE profile: real neighbours undulate QUASI-INDEPENDENTLY with phase drift (subagent finding) -- weak
    # shared flow, stronger own long-wave component. bot_det damped (annotation noise rode in at full gain).
    # sigma 50-90 read as "short-wavelength ropey meanders, nowhere in calibration" (Fable-5 judge); the
    # panel that fooled it at 75% had long-wavelength undulation -- shift the family longer
    # ANISOTROPIC undulation (audit issue 3): real wraps run near-straight along the scroll axis (canvas H)
    # and undulate in the winding plane (W) -- isotropic fields gave the network a conflicting geometry prior
    # and under-taught axis-wise seam continuation. sigma_H stretched / sigma_W shrunk by sqrt(A).
    A = float(os.environ.get("SC_ANISO", "1.14"))    # MEASURED on slab truth: wander 4.76 vs 5.7 deg/4vox
    so_ = float(rng.uniform(75, 140))
    # sigma 75-140 smooth field synthesized at 1/4 grid + trilinear upsample: identical spectrum
    # for a field with no content above the coarse grid scale, ~16x cheaper (profiled: the full-res
    # per-repose gaussians were 38% of compose time).
    Hc, Wc = max(H // 8, 8), max(W // 8, 8)
    own_c = ndi.gaussian_filter(rng.standard_normal((Hc, Wc)).astype(np.float32),
                                (so_ * np.sqrt(A) / 8.0, so_ / np.sqrt(A) / 8.0))
    own = ndi.zoom(own_c, (H / Hc, W / Wc), order=1)[:H, :W]
    own = own / (np.abs(own).max() + 1e-6) * float(rng.uniform(2.0, 4.5))
    yy = (np.arange(H, dtype=np.float32)[:, None] - H / 2) * float(rng.uniform(-0.05, 0.05))
    xx = (np.arange(W, dtype=np.float32)[None, :] - W / 2) * float(rng.uniform(-0.03, 0.03))
    # own amp up / flow gain down (was 1.5-3.5 / 0.5-1.1): consecutive open sheets riding the shared flow at
    # near-equal offsets drew evenly-spaced parallel arc trains (v32 panel tell)
    if winding:
        # per-sheet CONFORMITY draw: real neighbouring wraps share low-frequency curvature but
        # differ at mid frequencies -- a fixed 0.45 own-weight made every cube one coherent wave
        # (the "zebra" tell). Sheets in tight contact conform; sheets with room buckle alone.
        prof = flow * float(rng.uniform(0.7, 1.2)) + own * float(rng.uniform(0.35, 0.85))             + yy + xx + 0.3 * bot_det + (resid - resid.mean())
    else:
        prof = flow * float(rng.uniform(0.25, 1.05)) + own * float(rng.uniform(0.6, 1.25))             + yy + xx + 0.3 * bot_det + (resid - resid.mean())
    if crease is not None:
        # fold apexes stay ALIGNED across the pack (a fold folds the whole stack) but their
        # amplitude varies smoothly sheet-to-sheet -- the fixed full-gain shared ridge repeated
        # an identical chevron on every sheet (the zebra/jitter tell in gate17 09207/09208)
        prof = prof + crease * float(rng.uniform(0.35, 1.15))
    # PHYSICAL z-compression: compact regions ARE squeezed -- resample the real object thinner (real texture,
    # physically transformed; not a fabricated property). Papyrus mask resampled with the same map.
    # 0.32-0.6 over-squeezed: sheet pitch landed ~1.6x finer than real compact windows, so the sigma-1.2 base
    # lost 28% contrast (real loses 3%) and the quantile map re-stretched fine energy by slope^2 (lv_pap 172
    # vs 111). Real compact wraps are squeezed, but not that hard.
    cf = float(rng.uniform(*squeeze))
    E = ctbox.shape[-1]
    # AIR-PRESERVING squeeze. The old nearest-neighbour decimation dropped layers uniformly, which
    # collapsed the 1-3 vox INTERNAL air gaps (delaminated plies) the harvest had just preserved --
    # measured: synth ply-gap mode 1-2 vox vs real 7. Physics note above stands: papyrus is
    # near-incompressible, compaction closes AIR; but the harvested internal gaps are the
    # carbonisation record and the composer's job is to keep them. So: compress PAPYRUS-dominated
    # layer runs by cf, keep air-dominated layers verbatim. Extent bookkeeping uses the achieved
    # cf_eff (>= cf when internal air exists).
    frac_l = papbox.mean(axis=(0, 1))
    is_air = frac_l < float(os.environ.get("SC_AIRTHR", "0.35"))
    src_list = []
    i0 = 0
    while i0 < E:
        if is_air[i0]:
            src_list.append(i0)
            i0 += 1
        else:
            j0 = i0
            while j0 < E and not is_air[j0]:
                j0 += 1
            n = j0 - i0
            En_run = max(int(np.ceil(n * cf)), 1)
            idx = np.clip(np.rint(np.arange(En_run, dtype=np.float32) / cf), 0, n - 1).astype(np.int32) + i0
            src_list.extend(idx.tolist())
            i0 = j0
    src = np.asarray(src_list, np.int32)
    cf_eff = len(src) / max(E, 1)
    ctbox = ctbox[..., src]
    ctcore = ctcore[..., src]
    papbox = papbox[..., src]
    papf = papf[..., src] if papf is not None else None
    ext = np.clip(ext * cf_eff + ext_det * cf_eff * float(rng.uniform(0.7, 1.3)), 1.5, None)
    # OUTER-SKIN mask: layers within SC_SKIN of the column's outer papyrus envelope (or outside it).
    # The fusion core-swap below may only act here: swapping INTERIOR layers overwrote internal
    # delamination seams with bright core (measured: deep stratum split_frac 0.61%, worst of all).
    En2 = ctbox.shape[-1]
    SKIN = int(os.environ.get("SC_SKIN", "2"))
    occ_any = papbox.any(-1)
    firstl = np.argmax(papbox, axis=-1)
    lastl = En2 - 1 - np.argmax(papbox[..., ::-1], axis=-1)
    # skin is consumed only as 1-D gathers at deposit: ship the column bounds instead of the
    # (H,W,E) mask (skin[i,j,l] == l <= firstl+SKIN | l >= lastl-SKIN | ~occ_any)
    # sub-voxel occupancy along depth (advisor J1): exact for a locally-planar interface, and unlike
    # a blurred binary it cannot erode a 1-voxel ply.
    # papf: use the bank-precomputed base occupancy when present (per-sheet, rot/flip/crop applied
    # above like every other field; the squeeze resampled it with the same src map). Fallback keeps
    # the vectorized 1-D construction for live harvests.
    if papf is None:
        papf = papf_from_box(papbox)
    # firstm/lastm are the same argmaxes as firstl/lastl (previously computed twice over (H,W,E))
    firstm = np.where(occ_any, firstl, 0).astype(np.float32)
    lastm = np.where(occ_any, lastl, -1).astype(np.float32)
    return dict(prof=prof, ext=ext.astype(np.float32), ctbox=ctbox, ctcore=ctcore, papbox=papbox, sup=sup,
                emax=ctbox.shape[-1], skinp=(firstl, lastl, occ_any, SKIN), papf=papf,
                firstm=firstm, lastm=lastm)


# ----------------------------------------------------------------------------- settle + composite
def fold_warp_crop(ct, inst, papv, base_air, rng, out, p_warp, p_rot):
    """Geometry diversification AFTER compositing, BEFORE texture harmonization (grain and PSF are
    applied post-warp, so there is no strain-correlated grain tell). Fixes the advisor's top gaps:

    (1) FOLD WARP: the deposit is a single-valued height field, so hairpins / M-W-Z folds /
        collapse eddies / SELF-contact are unrepresentable -- yet self-contact is the only
        geometry where a tight contact means CONTINUE, the exact decision the FFN must learn.
        Bend the whole composed stack with a divergence-free vortex/saddle velocity field
        (integrated backward analytically, RK steps): level sets of the stacking field become
        U/M/hairpin shapes, gaps close geometrically at fold interiors (self-contact), label
        topology is preserved by construction (diffeomorphism), delam gaps stay pure. Amplitude
        distribution includes 0 -- today's corpus is the zero-vorticity member of the family.
    (2) small-angle SO(3) rotation about a random axis (stack normal was pinned within ~12 deg
        of one array axis) -- voids at cube corners fill from the stationary air field.
    (3) center crop from the margin canvas: no cube face coincides with a stack boundary (kills
        the flat-obstacle corrugation band and the fill_frac truncation edge)."""
    C = ct.shape[0]
    o = int(out)
    gz, gy, gx = np.mgrid[0:o, 0:o, 0:o].astype(np.float32)
    ctr = (o - 1) / 2.0
    X = np.stack([gz - ctr, gy - ctr, gx - ctr])
    if rng.random() < p_rot:
        ax_ = rng.standard_normal(3).astype(np.float32)
        ax_ /= float(np.linalg.norm(ax_)) + 1e-9
        th = float(rng.uniform(0.04, 0.30))
        K = np.array([[0, -ax_[2], ax_[1]], [ax_[2], 0, -ax_[0]], [-ax_[1], ax_[0], 0]], np.float32)
        R = np.eye(3, dtype=np.float32) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
        X = np.tensordot(R, X.reshape(3, -1), 1).reshape(3, o, o, o)
    coords = X + (C - 1) / 2.0
    if rng.random() < p_warp:
        pl = int(rng.integers(0, 2))              # fold in the (0,2) or (1,2) plane
        a_i, b_i = (0, 2) if pl == 0 else (1, 2)
        m_i = 1 - pl
        vort = [(float(rng.uniform(0.15, 0.85)) * C, float(rng.uniform(0.15, 0.85)) * C,
                 float(rng.uniform(45, 110)),
                 float(np.sign(rng.standard_normal())) * float(rng.uniform(0.5, 1.9)))
                for _ in range(int(rng.integers(1, 3)))]
        mfreq = float(rng.uniform(0.5, 1.5)) * np.pi / C
        mphase = float(rng.uniform(0, np.pi))
        # displacement integrated on a HALF grid then upsampled: the vortex field is smooth by
        # construction (sigma >= 45), so the displacement is band-limited far below the half-grid
        # Nyquist -- order-1 upsample is visually exact at ~8x less arithmetic (profiled 4.8s).
        h2 = np.s_[::2, ::2, ::2]
        pa, pb = coords[a_i][h2].copy(), coords[b_i][h2].copy()
        pa0, pb0 = pa.copy(), pb.copy()
        mod = 0.6 + 0.4 * np.cos(coords[m_i][h2] * mfreq + mphase)   # folds vary along 3rd axis
        NSTEP = 10
        for _ in range(NSTEP):
            ua = np.zeros_like(pa)
            ub = np.zeros_like(pb)
            for (ca, cb, sig, om) in vort:
                da, db = pa - ca, pb - cb
                w = om * np.exp(-0.5 * (da * da + db * db) / (sig * sig))
                ua += -db * w
                ub += da * w
            pa += ua * (mod / NSTEP)
            pb += ub * (mod / NSTEP)
        sh_c = pa.shape
        za = ndi.zoom(pa - pa0, [o / c for c in sh_c], order=1, grid_mode=True, mode="nearest")
        zb = ndi.zoom(pb - pb0, [o / c for c in sh_c], order=1, grid_mode=True, mode="nearest")
        coords[a_i] = coords[a_i] + za[:o, :o, :o]
        coords[b_i] = coords[b_i] + zb[:o, :o, :o]
    ctw = ndi.map_coordinates(ct, coords, order=1, mode="constant", cval=np.nan)
    instw = ndi.map_coordinates(inst, coords, order=0, mode="constant", cval=0)
    papw = ndi.map_coordinates(papv, coords, order=1, mode="constant", cval=0.0)
    void = ~np.isfinite(ctw)
    if void.any():
        ctw[void] = base_air[:o, :o, :o][void]    # stationary field: any window is seamless
        papw[void] = 0.0
        instw[void] = 0
    return ctw.astype(np.float32), instw.astype(np.int32), np.clip(papw, 0.0, 1.0)


def load_field(hw, coverage, sigma, rng):
    # anisotropic contact patches: fused regions are axis-elongated ribbons (straight along the scroll axis),
    # not isotropic blobs (audit issue 3)
    A = float(os.environ.get("SC_ANISO", "1.14"))    # MEASURED on slab truth: wander 4.76 vs 5.7 deg/4vox
    Hf, Wf = max(hw[0] // 4, 8), max(hw[1] // 4, 8)
    f = ndi.zoom(ndi.gaussian_filter(rng.random((Hf, Wf), dtype=np.float32),
                                     (sigma * np.sqrt(A) / 4, sigma / np.sqrt(A) / 4)),
                 (hw[0] / Hf, hw[1] / Wf), order=1)[:hw[0], :hw[1]]
    # second octave (advisor: single-scale fields give 1-2 long contact patches per pair; real
    # pairs open and close repeatedly along the same seam at several scales)
    s2 = float(rng.uniform(8, 20))
    H2, W2 = max(hw[0] // 2, 8), max(hw[1] // 2, 8)
    f2 = ndi.zoom(ndi.gaussian_filter(rng.random((H2, W2), dtype=np.float32),
                                      (s2 * np.sqrt(A) / 2, s2 / np.sqrt(A) / 2)),
                  (hw[0] / H2, hw[1] / W2), order=1)[:hw[0], :hw[1]]
    if os.environ.get("SC_LOAD_OCT2", "1") == "1":
        f = f + 0.35 * (f2 - float(f2.mean())) * (float(f.std()) / (float(f2.std()) + 1e-6))
    f = (f - f.min()) / (f.max() - f.min() + 1e-6)
    thr = np.quantile(f, 1.0 - coverage)
    width = 1.0 * (np.quantile(f, 0.95) - np.quantile(f, 0.05)) + 1e-6   # WIDE ramps: sharp contact/open flips made sawtooth
    return np.clip((f - thr) / width + 0.5, 0.0, 1.0)


def compose(cubes_dir, seed, hw=(192, 192), depth=192, coverage=(0.45, 0.98), p_spacer=0.06,
            spacer=(1.5, 4.0), n_source_cubes=3, fill_frac=0.95, hist_ref=None,
            fuse_onset=0.45, squeeze=(0.48, 0.78), winding=None, fg_target=None):
    rng = np.random.default_rng(seed)
    # GEO-DIVERSITY margin canvas: compose larger, then fold-warp + rotate + center-crop to the
    # requested size (SC_GEODIV=0 restores the axis-pure path for gate measurement).
    if os.environ.get("SC_GEODIV", "1") == "1":
        _C = int(os.environ.get("SC_CANVAS", "232"))
        if _C > hw[0]:
            hw = (_C, _C)
            depth = _C
    # FULL-3D RIBBON BANK (SC_SHEETS): harvest from the bank's full per-sheet ribbons instead of
    # window-cropped cubes. Three wins, all measured: (1) full spans -- the ribbons are the sheet's
    # whole extent, so the deposit span gate stops rejecting donors; (2) ~360 donors instead of ~8
    # single-sheet windows per composition; (3) the manifest carries the CORRECTED cross-sheet axis
    # per sheet -- the cross_sheet_axis heuristic mis-picks on these masks (recorded: 25%->96%
    # composer-gate pass when fixed), and a wrong axis slices plies obliquely, which is one reason
    # measured ply gaps came out 1-2 vox instead of the real ~7.
    sheets_dir = os.environ.get("SC_SHEETS")
    if sheets_dir:
        import json as _json
        import tifffile as _tf
        man = _json.load(open(os.path.join(sheets_dir, "sheets_manifest.json")))
        man = [r for r in (man if isinstance(man, list) else man.get("sheets", []))
               if r.get("passes_composer_gate", True)]
        nd = int(os.environ.get("SC_NDONORS", "22"))
        picks_r = rng.choice(len(man), size=min(nd, len(man)), replace=False)
        bank, airs = [], []
        for ri in picks_r:
            r = man[int(ri)]
            base_p = os.path.join(sheets_dir, "sheets", r["name"])
            vol = _tf.imread(base_p + "_0000.tif").astype(np.float32)
            msk = (_tf.imread(base_p + "_mask.tif") > 0).astype(np.int32)
            lo_, hi_ = np.percentile(vol, 0.5), np.percentile(vol, 99.5)
            vol = np.clip((vol - lo_) / (hi_ - lo_ + 1e-6) * 255.0, 0, 255)
            try:
                sh, air = extract_sheets(vol, msk, int(r["axis"]))
            except Exception:
                continue
            for s_ in sh:
                s_["src"] = r["name"]
            bank += sh
            airs.append(air)
        if len(bank) < 3:
            raise RuntimeError("not enough sheets harvested")
        air_dir = os.environ.get("SC_AIRCUBES")
        if air_dir:
            # ribbons are masked exports (no real air); harvest air blocks from window cubes so the
            # background keeps real gap texture instead of the flat fallback
            wc = sorted(glob.glob(os.path.join(air_dir, "*/")))
            for ci in rng.choice(len(wc), size=min(3, len(wc)), replace=False):
                try:
                    vol_a, msk_a = load_cube(wc[int(ci)])
                    _, air_a = extract_sheets(vol_a, msk_a, cross_sheet_axis(msk_a), max_sheets=0)
                    airs.append(air_a)
                except Exception:
                    pass
        return _compose_body(bank, airs, rng, seed, hw, depth, coverage, p_spacer, spacer,
                             fill_frac, hist_ref, fuse_onset, squeeze, winding, fg_target)
    cubes = sorted(glob.glob(os.path.join(cubes_dir, "*/")))
    picks = rng.choice(len(cubes), size=min(n_source_cubes, len(cubes)), replace=False)
    bank, airs = [], []
    bank_dir = os.environ.get("SC_BANK")
    for ci in picks:
        cd = cubes[int(ci)]
        cname = os.path.basename(cd.rstrip("/"))
        pk = os.path.join(bank_dir, cname + ".pkl") if bank_dir else None
        if pk and os.path.exists(pk):
            # PRE-HARVESTED bank: extract_sheets is deterministic per source cube (its only rng is seeded by
            # the sheet id), so the cache is bit-identical to a live harvest -- pure speed, zero output change.
            # Per-PROCESS memo on top: pool workers survive across jobs, so the same donor pickle
            # is otherwise re-read for every composition.
            global _BANK_MEMO
            try:
                _BANK_MEMO
            except NameError:
                _BANK_MEMO = {}
            if pk in _BANK_MEMO:
                sh, air = _BANK_MEMO[pk]
            else:
                import pickle
                sh, air = pickle.load(open(pk, "rb"))
                if len(_BANK_MEMO) < 40:
                    _BANK_MEMO[pk] = (sh, air)
        else:
            vol, msk = load_cube(cd)
            ax = cross_sheet_axis(msk)
            sh, air = extract_sheets(vol, msk, ax)
        for s in sh:
            s["src"] = cname
        # DONOR FUSED-LAMINA SCREEN: donor masks are the one channel that can put TWO real
        # laminae under ONE label (measured 3.89% suspect single-columns bank-wide); an exclusion
        # list (built by donor_screen.py from the census) drops the worst offenders at load.
        _excl = getattr(compose, "_excl", None)
        if _excl is None:
            _ep = os.environ.get("SC_EXCL", "")
            if _ep and os.path.exists(_ep):
                import json as _json2
                _excl = set(_json2.load(open(_ep)))
            else:
                _excl = set()
            compose._excl = _excl
        sh = [s for s in sh if f"{cname}:{s['id']}" not in _excl]
        bank += sh; airs.append(air)
    if len(bank) < 3:
        raise RuntimeError("not enough sheets harvested")
    return _compose_body(bank, airs, rng, seed, hw, depth, coverage, p_spacer, spacer,
                         fill_frac, hist_ref, fuse_onset, squeeze, winding, fg_target)


def _compose_body(bank, airs, rng, seed, hw, depth, coverage, p_spacer, spacer,
                  fill_frac, hist_ref, fuse_onset, squeeze, winding, fg_target=None):
    # PER-CUBE spacer probability: a fixed 6% contact-free rate made every composition equally
    # fused. Real windows range from tight packs to airy ones; draw the regime per cube.
    if os.environ.get("SC_VARSPACER", "1") == "1":
        if fg_target is not None:
            # STEERED density (advisor 1.0/3.1): fg is a deterministic monotone function of
            # fill_frac/p_spacer drawn from the SAME seed as the accept gate's band, so the band
            # could only reject compositions, never produce one -- the airy stratum was empty and
            # every miss cost a full compose. Set the knobs FROM the target (calibration: fill
            # 0.95 -> fg 0.40-0.57, roughly fg ~ 0.5*fill_frac) with a little jitter kept.
            fill_frac = float(np.clip(fg_target * 1.9 + rng.uniform(-0.05, 0.05), 0.40, 0.97))
            p_spacer = float(np.clip(0.04 + 0.41 * (0.50 - fg_target) / 0.35
                                     + rng.uniform(-0.03, 0.03), 0.03, 0.5))
        else:
            p_spacer = float(rng.uniform(0.04, 0.45))
            # true AIRY cubes need the STACK to stop early, not just wider spacers: fill_frac=0.95
            # packs every canvas to 95% depth by construction. Real windows span sparse to packed.
            fill_frac = float(rng.uniform(0.55, 0.97))
        if rng.random() < 0.2:
            spacer = (spacer[0] * 2.5, spacer[1] * 2.2)   # occasional genuinely wide gaps
    H, W = hw
    A_ = float(os.environ.get("SC_ANISO", "1.14"))
    sf_ = float(rng.uniform(40, 80))
    # sigma 40-80 field at 1/4 grid + bilinear upsample (band-limited far below coarse Nyquist)
    Hc4, Wc4 = max(H // 4, 8), max(W // 4, 8)
    flow = ndi.zoom(ndi.gaussian_filter(rng.standard_normal((Hc4, Wc4)).astype(np.float32),
                                        (sf_ * np.sqrt(A_) / 4, sf_ / np.sqrt(A_) / 4)),
                    (H / Hc4, W / Wc4), order=1)[:H, :W]
    flow = flow / (np.abs(flow).max() + 1e-6) * float(rng.uniform(4, 9))
    gy = (np.arange(H, dtype=np.float32)[:, None] - H / 2) * float(rng.uniform(-0.22, 0.22))
    gx = (np.arange(W, dtype=np.float32)[None, :] - W / 2) * float(rng.uniform(-0.12, 0.12))
    flow = flow + gy + gx                       # real sheets cross a window at 10-20 deg obliques
    if rng.random() < 0.30:
        # FOLD-CURVATURE seeds: real windows include fold apexes and arcs; a pure oblique-plane flow made
        # every composition the same compact-pack regime (the last blind tell was layout family, not texture)
        yc = (np.arange(H, dtype=np.float32)[:, None] - H * float(rng.uniform(0.2, 0.8))) / H
        xc = (np.arange(W, dtype=np.float32)[None, :] - W * float(rng.uniform(0.2, 0.8))) / W
        bowl = (yc * yc * float(rng.uniform(-1, 1)) + xc * xc * float(rng.uniform(-1, 1))
                + yc * xc * float(rng.uniform(-1.2, 1.2)))
        flow = flow + bowl * float(rng.uniform(18, 42))
    crease = None
    if rng.random() < 0.35:
        # CREASE / kink band: real compact regions contain fold apexes where curvature changes
        # discontinuously across a line -- a pure smooth flow field cannot produce one, and its
        # absence is exactly the "too simplistic single wave" look. Ridge profile along a random
        # line; every sheet inherits it through the shared flow (a fold folds the whole pack).
        th = float(rng.uniform(0, np.pi))
        yc2 = np.arange(H, dtype=np.float32)[:, None] - H * float(rng.uniform(0.25, 0.75))
        xc2 = np.arange(W, dtype=np.float32)[None, :] - W * float(rng.uniform(0.25, 0.75))
        dline = yc2 * np.cos(th) + xc2 * np.sin(th)
        wid = float(rng.uniform(18, 60))
        crease = (float(rng.uniform(8, 26)) * np.sign(rng.standard_normal())
                  * np.maximum(0.0, 1.0 - np.abs(dline) / wid))
    # first obstacle carries MID-STACK roughness, not a flat wall: the flat start packed the
    # first deposits into a tight uniform-pitch corrugation band at the low-x edge (a learnable
    # positional tell, advisor #4); real windows are crops of an ongoing stack.
    _bi = 1.0 if os.environ.get("SC_BURNIN", "1") == "1" else 0.0
    Hc2, Wc2 = max(H // 2, 8), max(W // 2, 8)
    s_top = (np.full((H, W), 2.0, np.float32)
             + ndi.zoom(ndi.gaussian_filter(
                 rng.standard_normal((Hc4, Wc4)).astype(np.float32), 15),
                 (H / Hc4, W / Wc4), order=1)[:H, :W] * 4
             + ndi.zoom(ndi.gaussian_filter(
                 rng.standard_normal((Hc2, Wc2)).astype(np.float32), 13),
                 (H / Hc2, W / Wc2), order=1)[:H, :W]
             * float(rng.uniform(1.0, 3.0)) * _bi
             + ndi.gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), 8)
             * float(rng.uniform(1.0, 2.5)) * _bi)
    # WINDING layout family (unbiased audit item 3): real compact wraps are order-preserving windings --
    # straighter, shared curvature, full-span. Half the compositions use it: profiles lean on the shared
    # flow with damped idiosyncratic wander, teaching the continuation prior that resolves invisible contacts.
    wind_ = bool(rng.random() < 0.5) if winding is None else bool(winding)
    # DELAM-RICHNESS diversity: per composition, bias donor choice toward delaminated sheets by a
    # random amount (k=0 -> uniform, k=8 -> strongly delam-rich). Gives cubes ranging from solid
    # compact packs to heavily-delaminated compact packs -- the combination the cores actually show.
    dk = float(rng.uniform(0.0, float(os.environ.get("SC_DELAM_K", "8"))))
    dvals = np.asarray([sh.get("delam", 0.0) for sh in bank], np.float64)
    dw = (1.0 + dk * dvals)
    dw = dw / dw.sum() if np.isfinite(dw).all() and dw.sum() > 0 else None
    ct_acc = np.zeros((H, W, depth), np.float32)          # premultiplied texture
    pap_acc = np.zeros((H, W, depth), np.float32)         # PAPYRUS-only coverage (phase field)
    best_occ = np.zeros((H, W, depth), np.float32)        # winner-take-all papyrus-volume score
    s_mat = np.zeros((H, W), np.float32)                  # top of MATERIAL per column
    wrmark = np.zeros((H, W, depth), np.int8) if os.environ.get("SC_WRMARK") else None
    cov_acc = np.zeros((H, W, depth), np.float32)         # coverage
    inst = np.zeros((H, W, depth), np.int32)
    meta_if, viol = [], 0
    tries = 0
    pend_tear = None                                      # (b, far-side mask) of an open tear
    # INTER-DONOR CORE-LEVEL HARMONIZATION (probe v2.1: per-cube contact AUC 0.82-0.94 vs real
    # 0.652 in EVERY stratum): donors harvested from different source cubes carry different core
    # brightness, so every fused abutment of two donors has a first-order statistical step that a
    # real contact -- one continuous medium imaged once -- never has. Pull each donor's core level
    # 60% toward the composition median (adjacent real wraps share material and beam; the excess
    # inter-sheet contrast is a harvest normalization artifact). Mean shift only, texture intact.
    _cms = []
    for _sh in bank:
        if _sh.get("cmed") is None:
            _pb0 = _sh["papbox"]
            _sh["cmed"] = float(np.median(_sh["ctbox"][_pb0])) if _pb0.any() else 128.0
        _cms.append(_sh["cmed"])
    cmed_T = float(np.median(_cms)) if _cms else 128.0
    recent = []                                           # last few bank indices: back-to-back reuse of one
    # count DEPOSITS against the budget, not attempts (v6 burned its budget on skips -> fg 0.16 outliers)
    while float((s_top < fill_frac * depth).mean()) > 0.015 and len(meta_if) < 140 and tries < 420:
        tries += 1
        bi = int(rng.integers(0, len(bank)))              # object stacks identical thicknesses = even arc trains
        if dw is not None:
            bi = int(rng.choice(len(bank), p=dw))
        if len(bank) > 4 and bi in recent:
            continue
        recent = (recent + [bi])[-3:]
        sheet = bank[bi]
        _supmin = int(os.environ.get("SC_SUPMIN", "9000"))
        sp = repose(sheet, hw, rng, flow, squeeze=squeeze, winding=wind_, supmin=_supmin,
                    crease=crease, spanmin=float(os.environ.get("SC_SPANMIN", "0.5")))
        if "ext" not in sp:
            continue                                      # hoisted support/span reject (cheap path)
        sup, ext_f, ctbox, ctcore, papbox = sp["sup"], sp["ext"], sp["ctbox"], sp["ctcore"], sp["papbox"]
        papf = sp["papf"]
        firstm_B, lastm_B = sp["firstm"], sp["lastm"]
        base = s_top + 0.6                                # first admissible bottom (the hard obstacle)
        # BRIDGE RELAXATION with DECAY: a sheet spanning an air gap follows its OWN smooth shape (bending
        # stiffness), not the substrate's bumps -- referencing raw s_top made every wiggle echo upward in
        # phase. And each successive sheet filters the shape AGAIN (real kinks fade with stacking distance;
        # v33 tell was one S-kink echoed in-phase through 6+ strands): blend toward a wider trend so
        # inherited detail decays ~25% per deposit instead of riding up forever.
        g26 = ndi.zoom(ndi.gaussian_filter(base[::2, ::2], 13.0), 2, order=1)[:H, :W]
        base_relax = np.maximum(base, 0.75 * ndi.gaussian_filter(base, 8.0) + 0.25 * g26)
        want = nanfill(np.where(sup, sp["prof"], np.nan))
        delta = float(np.nanmin(want[sup])) if sup.any() else 0.0
        b_rigid = base_relax + (want - delta)             # rigid approach against the relaxed reference
        gap = b_rigid - base
        fuse = None                                       # load-driven fusion (conform branch only)
        if rng.random() < p_spacer:
            # per-column modulation: constant-offset spacers drew constant-width gaps; real gaps pinch and swell
            spf = ndi.zoom(ndi.gaussian_filter(
                rng.standard_normal((max(hw[0] // 4, 8), max(hw[1] // 4, 8))).astype(np.float32),
                10), (hw[0] / max(hw[0] // 4, 8), hw[1] / max(hw[1] // 4, 8)),
                order=1)[:hw[0], :hw[1]]
            spf = spf / (np.abs(spf).max() + 1e-6)
            b = b_rigid + float(rng.uniform(*spacer)) * (1.0 + 0.5 * spf)
            mode = "spacer"
        else:
            cov = float(rng.uniform(*coverage))
            # field scale 40-100 (was 24-60): real fused contacts are LONG CONTINUOUS seams, not a leopard print
            # of islands. Measured on 12 test cubes: sigma 24-60 produced ~260 contact patches per 192^3 cube vs
            # the real slab's ~18 per equal volume -- each small island is dominated by its transition rim, which
            # is where the visible dips live, so fragmentation alone kept patch-level visibility ~12x real.
            M = load_field(hw, cov, float(rng.uniform(40, 100)), rng)
            b = b_rigid - np.minimum(M * gap, gap)        # conform; obstacle-clipped
            b = b - np.clip(M - 0.55, 0.0, 0.45) * 9.5    # pad-overlap up to ~4.3 vox where M~1: the air pads fully
                                                          # interleave -> REAL surfaces meet, zero-gap fusion
            # ZERO-EVIDENCE FUSION (issue 1): where the contact load is highest, drive the cores fully together
            # and switch the deposit to CORE material so the two real dense cores meet with no dim skin between --
            # a truly invisible boundary the model must learn to split from long-range context, not a local dip.
            # broad trigger: real compact windows are ~74% invisible contacts, so fusion must cover most of the
            # high-load area, not just the M~1 tail. Onset at M~0.5 (where pad-overlap begins).
            #
            # RAMP WIDTH (2026-07-20, contact_visibility measurement): the original /0.35 ramp spread the
            # partial-fusion transition over 10-20 px of the load field's 24-60 px scale, leaving bands of
            # blended half-skin INSIDE every contact -- so synth contact patches measured ~5.6x more locally
            # visible than real ones (fully-blind patches 13.2% vs real 73.3%; median visible frac 0.083 vs
            # 0.003). Real fused contacts are blind END-TO-END: the gap pinches closed and the intensity dip
            # disappears as soon as the residual gap is below the PSF (~1-2 vox), so the fused->open transition
            # in-plane is PSF-narrow, not 15 px wide. A model trained on the wide ramp can solve most synth
            # contacts from the local half-dips and never learn the propagate-from-anchor skill that real
            # contacts require. Narrow ramp = the fused stretch is uniformly blind, the anchors live OUTSIDE
            # the fused area (where sheets genuinely separate) -- matching how real pairs are 98% anchored yet
            # 73% of their contact patches are blind. Env SC_FUSE_RAMP to re-tune.
            # Round-2 measurement (ramp 0.10 alone): patch median 0.039, fully-blind 16.9% -- better than the
            # 0.35 ramp but still 12x more visible than real. Residual dips traced to the smoothing pipeline,
            # not the ramp: G(M,4) re-widens the transition to ~8 px regardless of ramp width, and the fuse
            # drop was applied BEFORE the sigma-3 bending smooth, which re-spread the sharp core-butt into a
            # wide grazing-touch ring (1-3 vox residual gaps -> dips all around every fused patch). Fixes:
            # tighter field smoothing (sigma 2), and the drop is applied AFTER the bending smooth (see below).
            _ramp = float(os.environ.get("SC_FUSE_RAMP", "0.06"))
            fuse = np.clip((ndi.gaussian_filter(M, 2.0) - fuse_onset) / _ramp, 0.0, 1.0)
            if os.environ.get("SC_NOFUSE"):
                fuse = np.zeros_like(fuse)               # A/B toggle: reproduces the pre-issue1 dim-line contacts
            ext_f = np.clip(ext_f * (1.0 - 0.30 * ndi.gaussian_filter(M, 4.0)), 1.5, None)
            mode = f"conform(cov={cov:.2f})"
        b = ndi.gaussian_filter(b, 3.0)                   # bending stiffness of the settled sheet itself
        # GEOMETRY-DRIVEN FUSION (round 3, the structural fix): fuse wherever the RESIDUAL gap after settling is
        # below the PSF scale, regardless of the load field. Rationale: a sub-PSF air gap physically images as
        # fused -- the scanner cannot represent it -- so "grazing contact with a crisp dim skin" is a synthesis
        # artifact, not a physical state. The conform geometry produces exactly that: wherever partial conforming
        # brings surfaces within 0-2 vox, the crisp rasterizer drew a skin/dim line that the later PSF blur only
        # softens into a HALF-visible dip. Those speckle contacts were the residual visibility source after the
        # ramp fixes (round-2 measurement: patch median still 0.041 vs real 0.003): hundreds of small
        # touching-with-skin patches per cube that let the model solve contacts locally. Full fusion below 1.2
        # vox, open above 2.0; REAL thin gaps >= 2 vox keep their genuine dim seam (real compact windows
        # measurably have those -- the 5/10/34 seam calibration -- and spacer-mode gaps start at 1.5).
        # Applied AFTER the bending smooth so nothing re-widens the transition; real sheets do kink where they
        # snap together. Load-driven fusion (the deliberate zero-evidence stretches) still applies on top.
        base_s = ndi.gaussian_filter(base, 2.0)       # advisor J2: s_top is a running max, never
        gap_res = b - base_s                          # smoothed; subtracting it re-injected 1-voxel
        fuse_g = np.clip((2.0 - gap_res) / 0.8, 0.0, 1.0)   # roughness AFTER the bending smooth
        # SEPARATE drops (thickness-gate regression fix): the first cut applied the full 4.5-vox load-fusion
        # plunge to every grazing contact, and in the deep stratum that is most of every interface -- the
        # overwrite then ate the lower sheet's top voxels across the whole contact and deep p50 thickness
        # collapsed to 2.0 vox (the exact bug the squeeze recalibration had just fixed). Geometry fusion only
        # needs to CLOSE the sub-PSF gap plus ~2 vox of skin interleave so the cores meet; the deliberate
        # zero-evidence stretches keep their calibrated 4.5 (thickness-validated pre-round-3).
        drop_geom = ndi.gaussian_filter(fuse_g * np.clip(gap_res + 2.0, 0.0, 3.0), 2.0)
        drop_load = ndi.gaussian_filter(fuse * 4.5, 2.0) if fuse is not None else 0.0
        b = b - np.maximum(drop_load, drop_geom)
        fuse = fuse_g if fuse is None else np.maximum(fuse, fuse_g)   # core-swap/overwrite strength downstream
        # penetration clamp is a TUNED tradeoff once the fuse ramp is sharp: too deep and every fused sheet's
        # bottom voxels overlap the sheet below across the WHOLE interface -- they lose the first-claim race and
        # the stratum's label thickness collapses (deep p50 4.0 -> 2.0 at 4.2); too shallow and the air pads
        # survive between cores and the interface reads dim again. SC_FUSE_CLAMP to sweep.
        # Swept 4.2/3.4/2.8/2.2 (2026-07-20): deep label thickness 2.0/2.0/3.5/3.5, blind patches 42/38/33/26%.
        # 2.8 is the knee -- deeper eats claims across whole interfaces, shallower loses blindness for nothing.
        # MATERIAL-REFERENCED penetration bound (root cause of user defects 1-3). The constant
        # clamp let an arriving sheet drive its material THROUGH the previous sheet's envelope
        # whenever the top ply was thinner than the clamp: its material+label landed inside the
        # host's delamination gap (foreign-in-gap, measured 20.2% at compose vs real 0.45%), its
        # own bottom voxels lost the claim race (thin-column films, 40.5% vs real 1.7%), and its
        # unclaimed bright matter smeared into gaps (traces, 4x old synth). Physics: fusion may
        # close AIR until material touches material -- interpenetration does not exist. Bound b so
        # B's FIRST material layer never descends below A's material top (s_mat), with 0.25 vox
        # PSF interleave. Pads and envelope air close for free; blindness is unchanged because a
        # material-on-material contact is exactly the blind case.
        b = np.maximum(b, s_mat - firstm_B - 0.25)
        # TEAR JUNCTIONS (advisor #3): the corpus had the hard POSITIVE (one sheet across a hole)
        # but never the hard NEGATIVE -- two DIFFERENT sheets collinear across a tear gap, the
        # second-most-common real false-merge. Consume an open tear: this sheet becomes the
        # far-side partner at the SAME settled height (collinear by construction), cut by the
        # stored ragged mask; the material bound is re-applied after the override.
        if pend_tear is not None:
            _tb, _tm = pend_tear
            pend_tear = None
            if int((sup & _tm).sum()) > 4000:
                sup = sup & _tm
                b = np.maximum(_tb + float(rng.uniform(-1.5, 1.5)), s_mat - firstm_B - 0.25)
        elif rng.random() < float(os.environ.get("SC_TEAR_P", "0.10")) and float(sup.mean()) > 0.8:
            _thc = float(rng.uniform(0, np.pi))
            _d0 = ((np.arange(H, dtype=np.float32)[:, None] - H * float(rng.uniform(0.3, 0.7)))
                   * np.cos(_thc)
                   + (np.arange(W, dtype=np.float32)[None, :] - W * float(rng.uniform(0.3, 0.7)))
                   * np.sin(_thc))
            # ragged cut: real tear edges wander; a straight cut would be a learnable tell
            _d0 = _d0 + ndi.gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), 18)                 * float(rng.uniform(3.0, 10.0))
            _gapw = float(rng.uniform(1.0, 6.0))
            pend_tear = (b.copy(), _d0 < -_gapw / 2)
            sup = sup & (_d0 > _gapw / 2)
        viol += 0
        viol += int((b < base - 6.5)[sup].sum())
        # ---- composite the REAL 3D object run: [b, b+ext) carries the sheet's actual voxels, including its
        # internal REAL air (delaminated plies, holes) -- nothing about the material is synthesized. Coverage AA
        # applies only at the run's outer boundaries; papyrus voxels get the instance id, internal air stays 0.
        iid = len(meta_if) + 1
        kmax = int(np.ceil(float(np.nanmax(ext_f)))) + 2
        zb = np.floor(b).astype(np.int32)
        # 1-D deposit (profile: the per-k full-canvas planes were 4.2 s/cube). The candidate column
        # set is FIXED across k -- pack it once and run the k-loop entirely on 1-D vectors.
        candm = sup & (ext_f > 0.4)
        ic, jc = np.nonzero(candm)
        b_c = b[candm].astype(np.float32)
        ext_c = ext_f[candm].astype(np.float32)
        zb_c = zb[candm]
        fuse_c = fuse[candm].astype(np.float32) if fuse is not None             else np.zeros(ic.shape, np.float32)
        _fl, _ll, _oc, _SK = sp["skinp"]
        fl_c = _fl[ic, jc]
        ll_c = _ll[ic, jc]
        oc_c = _oc[ic, jc]
        fm_c = firstm_B[ic, jc].astype(np.float32)
        _sqmode = os.environ.get("SC_FUSE_MODE", "coreswap") == "squeeze"
        E1m = ctbox.shape[-1]
        for k in range(kmax):
            zc = zb_c + k
            zfc = zc.astype(np.float32)
            cov_c = np.clip(np.minimum(b_c + ext_c, zfc + 1.0) - np.maximum(b_c, zfc), 0.0, 1.0)
            m = (zc >= 0) & (zc < depth) & (cov_c > 1e-3)
            if not m.any():
                continue
            # SUB-VOXEL deposit (advisor J1): linear content + signed-distance occupancy; the
            # material surface moves continuously with b (integer snap caused the 1-voxel comb).
            ii = ic[m]
            jj = jc[m]
            zi = zc[m]
            covm = cov_c[m]
            uf = np.clip(zfc[m] + 0.5 - b_c[m], 0.0, E1m - 1.0001)
            fv = fuse_c[m]
            # SKIN-ONLY fusion: the fusion path acts through fv, so one gate protects internal
            # delamination seams from being filled with bright core.
            lay0 = np.clip(np.rint(uf).astype(np.int32), 0, E1m - 1)
            fv = fv * ((lay0 <= fl_c[m] + _SK) | (lay0 >= ll_c[m] - _SK)
                       | ~oc_c[m]).astype(np.float32)
            if _sqmode:
                # COMPRESSION FUSION (probe v2.1: real through-material contacts read as interior,
                # AUC 0.652, while core-swap's copied texture reads 0.77-0.84). Physical model:
                # invisible fusion does NOT replace skins with core -- the skins survive, the air
                # goes to zero, and the PSF averages skin+skin+core into brightness. Compress the
                # donor's own depth coordinate at fused patches (anchored at its first material
                # layer, the contact side): the genuine skin renders sub-voxel, partial-volume
                # blends it into the neighbour's core, and the material sequence stays real.
                c_loc = 1.0 + 1.5 * fv
                fm = fm_c[m]
                uf = np.clip(fm + (uf - fm) * c_loc, 0.0, E1m - 1.0001)
            l0 = np.floor(uf).astype(np.int32)
            l1 = np.minimum(l0 + 1, E1m - 1)
            wgt = (uf - l0).astype(np.float32)
            val = ctbox[ii, jj, l0] * (1 - wgt) + ctbox[ii, jj, l1] * wgt
            occ = papf[ii, jj, l0] * (1 - wgt) + papf[ii, jj, l1] * wgt
            pap = occ >= 0.5
            lay = np.where(wgt < 0.5, l0, l1)             # nearest, for skin/core lookups
            # CORE-SWAP at fused patches: the harvested run's air-facing DIM skin is replaced by the sheet's own
            # nearest interior-core voxel (real material, ctcore) blended by fusion strength -- so a fused contact
            # reads as continuous bright dense papyrus, not a dim ramp. Touching-but-not-fused (fv=0) keeps the
            # real skin and its faint seam verbatim.
            if _sqmode:
                # compressed material is denser: modest attenuation lift, gated by fv
                val = val * (1.0 + 0.35 * (c_loc - 1.0))
            else:
                val = val * (1.0 - fv) + ctcore[ii, jj, lay] * fv
            val = val + 0.6 * (cmed_T - sheet["cmed"]) * occ
            ip, jp, zp = ii[pap], jj[pap], zi[pap]
            fvp = fv[pap]; covp = covm[pap]
            # FUSION OVERWRITE: where the contact load is high and the arriving core fully covers the voxel, the
            # compression has physically pushed this bright dense core into the space the lower sheet's air-facing
            # DIM skin occupied -- so the CT reads THIS core's density, not a blend. Reset the accumulator there
            # (drop the lower sheet's dim skin) before the add, so the voxel ends up core-bright with NO dip. This
            # is what closes the invisible contact; capacity-lift/first-claim alone left the dim average (median
            # dip 2.5 vs real 0). All values are real core voxels -- material relocated by compression, not faked.
            # covp gate at 0.3, not 0.6: interface voxels are precisely the anti-aliased PARTIAL-coverage ones
            # (b is non-integer, so the meeting voxel is split between the runs), and the 0.6 gate left ~half of
            # them un-overwritten -- the lower sheet's dim skin survived in the blend as a scattered 1-vox dip
            # line inside "fused" contacts (the residual visibility contact_visibility kept measuring).
            # inst==0 gate (thickness regression fix, round 7): the reset may clear SKIN/PAD/AIR -- material the
            # compression physically displaced, which is always UNCLAIMED (inst 0) -- but must never delete a
            # voxel another sheet has already claimed as core. With the old soft fuse ramp full overwrite only
            # occurred at small load peaks and the eating was invisible; the sharp ramp (blindness fix) put
            # fv~1 across entire interfaces and the roughness-overlap voxels it deleted halved the deep-stratum
            # label thickness (p50 4.0 -> 2.0 vs the production baseline). Where cores genuinely overlap the
            # first-claim add now BLENDS them instead (both are bright core -- reads as compressed material).
            ow = (fvp > 0.6) & (covp > 0.3) & (inst[ip, jp, zp] == 0)
            if ow.any():
                ct_acc[ip[ow], jp[ow], zp[ow]] = 0.0
                cov_acc[ip[ow], jp[ow], zp[ow]] = 0.0
                # overwrite writes THIS sheet's relocated core CT at full brightness: label it and
                # give it a winning score so the claim survives the winner-take-all pass below.
                inst[ip[ow], jp[ow], zp[ow]] = iid
                best_occ[ip[ow], jp[ow], zp[ow]] = 1.0
                pap_acc[ip[ow], jp[ow], zp[ow]] = np.maximum(pap_acc[ip[ow], jp[ow], zp[ow]], 0.51)
            # FIRST-CLAIM compositing elsewhere: real fused bundles keep each wrap's bright-core/dim-skin profile
            # with faint dim seams; the first material into a voxel keeps it verbatim (fv=0 -> pure first-claim).
            cur = cov_acc[ip, jp, zp]
            add = np.minimum(covp, np.clip(1.0 - cur * (1.0 - fvp), 0.0, 1.0))
            ct_acc[ip, jp, zp] += add * val[pap]
            cov_acc[ip, jp, zp] += add
            if wrmark is not None:
                wrmark[ip, jp, zp] = 1
            # WINNER-TAKE-ALL labels (advisor 2.2c): a voxel belongs to whichever sheet contributes
            # the most papyrus volume to it (occ * covg). Replaces the first-claim race + covg>0.5
            # gate, which together left ~half the deposited material unlabeled: fused interfaces
            # lose the claim (pap_acc saturates before the arriving core lands), and the fractional
            # surface b phases coverage <=0.5/<=0.5 across two voxels so whole squeezed plies never
            # cross the gate -- the ghost films and the 55%-unlabelled-fg collapse. Labels become a
            # partition of the material by construction and deposit order stops mattering. The 0.15
            # floor keeps genuinely sub-visible slivers (occ*covg below the PSF) as background.
            occ_eff = (occ * covm).astype(np.float32)
            win = pap.copy()
            win[pap] = occ_eff[pap] > np.maximum(best_occ[ip, jp, zp], 0.15)
            iw, jw, zw = ii[win], jj[win], zi[win]
            inst[iw, jw, zw] = iid
            best_occ[iw, jw, zw] = occ_eff[win]
            pap_acc[ip, jp, zp] += add
            # the box's REAL air (inter-ply gaps, surface margins): keep its true texture, but only where nothing
            # else has claimed the voxel -- pads may overlap other sheets' pads or papyrus. At fused patches the
            # inter-core air is physically squeezed out, so suppress the pad-air injection by fusion strength.
            an = ~pap
            if an.any():
                empty = cov_acc[ii[an], jj[an], zi[an]] < 0.05
                ia, ja, za = ii[an][empty], jj[an][empty], zi[an][empty]
                aw = 0.6 * (1.0 - fv[an][empty])
                ct_acc[ia, ja, za] += aw * val[an][empty]
                cov_acc[ia, ja, za] += aw
                if wrmark is not None:
                    wrmark[ia, ja, za] = 2
        contact = float((b - base < 0.75)[sup].mean()) if sup.any() else 0.0
        top_new = b + ext_f
        sup_solid = ndi.binary_fill_holes(sup)
        holes_ = sup_solid & ~sup
        if holes_.any():
            # nearest-support lookup at HALF resolution: the filled tops are smooth fields, the
            # sampled even-coordinate positions are genuine sup columns, and the full-res EDT with
            # indices cost 1.9 s/cube across deposits.
            s2_ = sup[::2, ::2]
            if s2_.any():
                nn2 = ndi.distance_transform_edt(~s2_, return_indices=True)[1]
                iiF = np.clip(np.repeat(np.repeat(nn2[0] * 2, 2, 0), 2, 1)[:H, :W], 0, H - 1)
                jjF = np.clip(np.repeat(np.repeat(nn2[1] * 2, 2, 0), 2, 1)[:H, :W], 0, W - 1)
                nn_ = (iiF, jjF)
            else:
                nn_ = ndi.distance_transform_edt(~sup, return_indices=True)[1]
                nn_ = (nn_[0], nn_[1])
            top_fill = top_new[nn_]
            top_new = np.where(holes_, top_fill, top_new)
        s_top = np.where(sup_solid, np.maximum(s_top, top_new), s_top)
        mat_new = b + lastm_B + 1.0
        if holes_.any():
            mat_new = np.where(holes_, mat_new[nn_], mat_new)
        s_mat = np.where(sup_solid & (lastm_B >= 0), np.maximum(s_mat, mat_new), s_mat)
        meta_if.append(dict(i=iid, src=sheet["src"], mode=mode, contact_frac=round(contact, 3),
                            sup=int(sup.sum())))
    # ---- background = REAL AIR tiled from the source cubes (real gaps are mid-gray and textured, never
    # black iid noise); then composite; then histogram-match everything to a REAL compact window ----
    blocks = [a["block"] for a in airs if a.get("block") is not None]
    if blocks:
        # ribbon donors yield air blocks of varying size -- crop all to the common cube
        bs = min(min(b.shape) for b in blocks)
        blocks = [b[:bs, :bs, :bs] for b in blocks] if bs >= 16 else []
    if blocks:
        # STATIONARY PSD-MATCHED AIR FIELD (advisor R1-a). The old Hann overlap-add of real windows
        # had three defects: variance swung sqrt(8)x on a 32-voxel lattice (Hann kills MEAN seams,
        # not variance seams), the first 32 rows/cols/slices were a verbatim single tile (the
        # rectangular corner blocks in renders), and the "air" windows are merely the AIRIEST 64^3
        # of a real cube -- sheet fragments tiled into the background (measured: 37% of bright
        # unlabeled trace voxels were background-written). Real inter-wrap air is recon noise +
        # sub-resolution debris + a slow level trend: synthesize ONE seamless field carrying the
        # blocks' measured spectrum + air-only marginal, plus a smooth macro-level term.
        from skimage.filters import threshold_otsu as _otsu
        marg, meds = [], []
        for t in blocks:
            t32 = t.astype(np.float32)
            try:
                th_ = float(_otsu(t32))
            except Exception:
                th_ = float(np.percentile(t32, 70))
            a_ = t32[t32 < th_]
            if a_.size > 500:
                marg.append(a_); meds.append(float(np.median(a_)))
        if marg:
            margv = np.concatenate(marg)
            t0 = blocks[int(rng.integers(0, len(blocks)))].astype(np.float32)
            # flatten sheet fragments before measuring the spectrum: clip to the air range so the
            # PSD is the AIR texture's, then drop the block's own low-frequency trend
            bc = np.clip(t0, None, float(np.percentile(margv, 99.5)))
            bc = bc - ndi.gaussian_filter(bc, 8.0)
            nbA = 40
            FA = np.abs(_sfft.fftn(bc, workers=1) if _sfft is not None
                        else np.fft.fftn(bc)) ** 2 / bc.size
            fzb = np.fft.fftfreq(bc.shape[0])[:, None, None]
            fyb = np.fft.fftfreq(bc.shape[1])[None, :, None]
            fxb = np.fft.fftfreq(bc.shape[2])[None, None, :]
            rbb = np.minimum((np.sqrt(fzb**2 + fyb**2 + fxb**2) * 2 * nbA).astype(int), nbA - 1)
            Pa = (np.bincount(rbb.ravel(), FA.ravel(), minlength=nbA)
                  / np.maximum(np.bincount(rbb.ravel(), minlength=nbA), 1))
            Pa[0] = 0.0                                       # trend handled by the macro term
            fz3 = np.fft.fftfreq(H)[:, None, None]
            fy3 = np.fft.fftfreq(W)[None, :, None]
            fx3 = np.fft.rfftfreq(depth)[None, None, :]
            rb3a = np.minimum(np.sqrt(fz3**2 + fy3**2 + fx3**2), 0.5).ravel() * 2 * nbA - 0.5
            ampA = np.interp(rb3a, np.arange(nbA), np.sqrt(np.maximum(Pa, 0.0)))
            wnA = _rfftn(rng.standard_normal((H, W, depth)).astype(np.float32))
            gA = _irfftn(wnA * ampA.reshape(wnA.shape).astype(np.float32),
                         (H, W, depth)).astype(np.float32)
            # rank-map onto the measured air-only marginal: texture becomes real-air distributed
            gq_ = np.quantile(margv, np.linspace(0, 1, 512))
            sq_ = np.quantile(gA.ravel()[::8], np.linspace(0, 1, 512)) + np.arange(512) * 1e-6
            base_air = np.interp(gA, sq_, gq_).astype(np.float32)
            # gap-to-gap macro level variation, now an EXPLICIT smooth term with the measured spread
            lv_sd = float(np.std(meds)) if len(meds) > 1 else 2.0
            if lv_sd > 0.1:
                mac = ndi.gaussian_filter(rng.standard_normal(
                    (max(H // 8, 4), max(W // 8, 4), max(depth // 8, 4))).astype(np.float32), 4.0)
                mac = ndi.zoom(mac, (H / mac.shape[0], W / mac.shape[1], depth / mac.shape[2]),
                               order=1)[:H, :W, :depth]
                base_air = base_air + mac / (mac.std() + 1e-6) * lv_sd
        else:
            blocks = []
    if not blocks:
        am = float(np.nanmean([a["air_med"] for a in airs]))
        base_air = np.full((H, W, depth), am, np.float32)
    cov = np.clip(cov_acc, 0.0, 1.0)
    ct = ct_acc / np.maximum(cov_acc, 1e-3) * cov + base_air * (1.0 - cov)
    # PHASE FIELD = papyrus-only coverage (advisor 2.2a). cov_acc contains the 0.6-weight pad-air
    # writes, so an unfused pad-air voxel reads cov=0.6 and was classified as PAPYRUS by every
    # downstream phase decision -- mean-shifted and quantile-mapped onto the reference's papyrus
    # distribution, i.e. dim sheet-shaped films (R2), and the papyrus phase diluted with remapped
    # air (inside-sheet median 128 vs real 153). cov stays as the compositing alpha ONLY.
    papv = np.clip(pap_acc, 0.0, 1.0)
    if os.environ.get("SC_GEODIV", "1") == "1":
        _out = int(os.environ.get("SC_OUT", "192"))
        if _out < H or float(os.environ.get("SC_WARP_P", "0.55")) > 0:
            ct, inst, papv = fold_warp_crop(ct, inst, papv, base_air, rng, min(_out, H),
                                            float(os.environ.get("SC_WARP_P", "0.55")),
                                            float(os.environ.get("SC_ROT_P", "0.8")))
    if wrmark is not None:
        np.save("/root/wr.npy", wrmark)
    if np.isnan(ct).any():
        ct = nanfill(ct)                    # belt-and-suspenders: never let a NaN reach the blur
    # DIP-CONDITIONAL width-1 closing. Deposits leave 1-voxel same-id label holes where a partial-
    # coverage voxel failed the papyrus gate while the CT is bright (anti-aliasing) -- measured as a
    # width-1 flood the DONOR masks do not have. A real pinhole ply shows a CT dip (donor width-1
    # dips: median 16 grey); an AA hole does not. Close only the bright ones; labels elsewhere
    # untouched.
    mid = inst[..., 1:-1]
    same = (inst[..., :-2] == inst[..., 2:]) & (inst[..., :-2] > 0) & (mid == 0)
    flank = np.minimum(ct[..., :-2], ct[..., 2:])
    bright = ct[..., 1:-1] > flank - 8.0          # no dip deeper than 8 grey => artifact
    fill = same & bright
    mid[fill] = inst[..., :-2][fill]
    # frame-edge debris: deposits cropped by the canvas leave thin partial-label slivers along the
    # borders (visible in every render as rim confetti). Real corpora min-size instances anyway.
    # NOTE deliberately NO label nulling here. A dropped label leaves its CT behind as bright
    # matter belonging to nothing ("white traces", user-reported); labels stay truthful to the
    # material and small instances are filtered at TRAINING time, not erased from the image.
    # INTENSITY + RESOLUTION HARMONIZATION, in physical order:
    #   1. split quantile map on the CRISP composite, blended by the AA coverage field itself (cov is the true
    #      papyrus fraction per voxel: interiors of even 1px sheets get their pure phase map -- no halo
    #      dilution -- and the transition is confined to the genuine 1-2px partial-volume band);
    #   2. THEN the PSF blur, as the last spatial op: a real edge is PSF(bright|dark), and mapping before
    #      blurring means nothing can re-steepen what the blur softened (post-blur mapping put lv_bnd at
    #      240-610 vs real 169 across three variants);
    #   3. THEN grain with the reference's measured spectrum -- real grain lives at recon resolution, post-PSF.
    if hist_ref is not None:
        if isinstance(hist_ref, (list, tuple)):
            # FG-MATCHED reference selection (user defect: bright traces in AIRY cubes, D2 1.6-3.0%
            # there vs ~0% in dense ones). The window sampler is fg-gated toward DENSE references;
            # quantile-mapping an airy composition onto a dense reference stretches the air phase
            # into the reference's brighter range -- a bright air tail real scans do not have.
            # Match the reference to the composition's own density instead.
            from skimage.filters import threshold_otsu
            fg_syn = float((papv >= 0.5).mean())
            best_w, best_d = hist_ref[0], 1e9
            for wnd in hist_ref:
                w32 = np.asarray(wnd, np.float32)
                try:
                    fgw = float((w32 >= threshold_otsu(w32)).mean())
                except Exception:
                    continue
                if abs(fgw - fg_syn) < best_d:
                    best_d, best_w = abs(fgw - fg_syn), wnd
            hist_ref = best_w
        hr = np.asarray(hist_ref, np.float32)
        hp = hr - ndi.gaussian_filter(hr, 1.2)                    # the real window's own grain
        gstd = float(hp.std())
        from skimage.filters import threshold_otsu
        # BASE/DETAIL split on both sides: the quantile map must redistribute STRUCTURAL variation, not
        # amplify pixel-scale residue (mapping raw voxels stretched post-blur micro-noise to fill the target
        # width: lv_pap 173-211 vs real 111). Map the sigma-1.2 base to the reference BASE's quantiles;
        # fine detail passes through unscaled; the grain below is the pixel-scale term -- the same
        # decomposition hp itself comes from, so base<->base, detail<->detail, grain<->grain.
        hb = ndi.gaussian_filter(hr, 1.2)
        ref = hb.ravel()
        thr = float(threshold_otsu(ref))                     # otsu phases: bright=papyrus-like, dark=gap-like
        import os as _os
        _dbg = _os.environ.get("SC_DEBUG")

        def _st(tag, a):
            if _dbg:
                cs_ = a[a.shape[0] // 2]
                lv_ = float(ndi.laplace(ndi.gaussian_filter(cs_, 0.5)).var())
                print(f"  [{tag}] vol std={a.std():.1f} cs std={cs_.std():.1f} lv={lv_:.0f} "
                      f"cs b215={(cs_ >= 215).mean():.4f} cs b230={(cs_ >= 230).mean():.4f}", flush=True)
        if _dbg:
            lvhp = float(ndi.laplace(ndi.gaussian_filter(hp, 0.5)).var())
            lvhr = float(ndi.laplace(ndi.gaussian_filter(hr, 0.5)).var())
            print(f"  [ref] std={hr.std():.1f} lv(hr)={lvhr:.0f} lv(hp grain)={lvhp:.0f} gstd={gstd:.1f} "
                  f"b215={(hr >= 215).mean():.4f} b230={(hr >= 230).mean():.4f} "
                  f"fg={(hr >= thr).mean():.3f} | synth fg={(papv >= 0.5).mean():.3f}", flush=True)
        _st("composite", ct)
        # PER-PHASE AFFINE first, on the crisp composite: the sources' global normalization gives a smaller
        # sheet-vs-gap separation than the reference, and any later quantile map would have to stretch
        # (slope>1 = fine-structure amplification, the recurring lv_pap excess). A linear map has constant
        # slope: separation and levels land on the reference with ZERO relative crisping.
        thr_r = float(threshold_otsu(hr.ravel()))
        aff = []
        for m_, tpart in ((papv >= 0.5, hr[hr >= thr_r]), (papv < 0.5, hr[hr < thr_r])):
            v = ct[m_]
            # MEAN SHIFT ONLY: scaling std to the reference amplified the pap texture (lv_pap 157) --
            # the macro deficit was phase separation, which offsets alone fix with zero texture change
            aff.append(float(tpart.mean()) - float(v.mean()) if v.size > 10 and tpart.size > 10 else 0.0)
        ct = papv * (ct + aff[0]) + (1.0 - papv) * (ct + aff[1])
        _st("affine", ct)
        # BASE EXTRACTION directly from the crisp composite -- and the detail is DISCARDED, not blurred:
        # the sub-1.2px content of the sources is their own recon grain, which the target grain g3 replaces
        # wholesale. The old strong global blur over-crushed mid-scale contrast (std 35-40 vs target ~55),
        # forcing the quantile map to stretch ~1.5x and re-amplify fine energy by slope^2 (stage trace:
        # blur lv 107 -> map lv 214 vs real 123-150). Base-to-base mapping at slope ~1.1 amplifies nothing.
        # sig_d = small residual PSF difference between the two recons at base scale.
        sig_d = float(np.random.default_rng(seed ^ 0xA5).uniform(0.25, 0.45))
        bs_ = ndi.gaussian_filter(ct, float(np.hypot(1.2, sig_d)))
        # EDGE-BAND widening: real papyrus|air transitions are wider than the PSF -- the sheet face
        # undulates sub-voxel, so partial volume smears the ramp (real lv_bnd 169 at separation ~65 vs our
        # 349: edges 2.6x sharper per unit contrast). Our settled bottoms are smooth surfaces, so the
        # roughness ramp is added by widening the base only inside the boundary band; interiors keep their
        # texture scale.
        w_e = 4.0 * ndi.gaussian_filter(papv, 0.9) * (1.0 - ndi.gaussian_filter(papv, 0.9))
        bs_ = bs_ * (1.0 - w_e) + ndi.gaussian_filter(bs_, 1.7) * w_e
        # detail band KEPT at calibrated amplitude, not discarded: it carries the real along-ridge mottle,
        # gap wisps and fleck cores (Fable-5 judge: synth gaps "uniform mid-gray, weak grain, fleck deficit";
        # discarding det and substituting stationary grain airbrushed the sheets). Normalized on the PAPYRUS
        # phase (det concentrates in sheets; a global norm under-damped it: lv_pap 197 vs band max 152), and
        # the grain below is phase-weighted so pap fine = sqrt(0.55^2+0.83^2) ~ 1.0 gstd, air fine = 1.0 gstd.
        det = ct - bs_
        cov_b = ndi.gaussian_filter(papv, 1.25)              # wide blend: the tight sigma-0.9 variant raised
        pm_ = cov_b >= 0.5                                   # the map slope (lv_pap); with the affine fixing
                                                             # separation, the blend's std shrink is minor
        # per-phase det scales: det is crisper per unit std than the reference grain (~2.6x Laplacian), so
        # pap takes 0.35 gstd of it; real AIR interiors carry substantial micro-structure beyond grain
        # (real lv_air 121 vs grain-only 25 -- wisps/debris), so air keeps its own real detail at 0.7 gstd.
        pp_ = cov_b > 0.95; aa_ = cov_b < 0.05                # PURE regions: the AA band's big residuals
        sp_ = float(det[pp_].std()) + 1e-6 if pp_.sum() > 100 else 1.0    # inflated the norms
        sa_ = float(det[aa_].std()) + 1e-6 if aa_.sum() > 100 else 1.0
        # FLECK SPLIT: bulk-normalizing det crushed the genuine mineral flecks 6x (they live in the tail,
        # not the bulk: real b215 0.051 vs our 0.026). Exceedances beyond 3 local std are real dense
        # inclusions that survive the real PSF at near-full brightness -- keep them at 0.8.
        sfield = cov_b * sp_ + (1.0 - cov_b) * sa_
        # ASYMMETRIC clip (ITERATION_PLAN C): dark lacunae cores pass at 3 sigma (the symmetric 4.5
        # damped them -- "creamy troughs that never go truly dark"); bright stays 4.5 because the
        # bright-tail population is injected explicitly below, not smuggled through a sigma gate.
        fleck = det - np.clip(det, -3.0 * sfield, 4.5 * sfield)
        bulk = det - fleck
        # DETAIL AMPLITUDE SOLVED from the reference's own within-phase fine energy (advisor 2.3):
        # per phase, det^2 + grain^2 must equal the reference high-pass band's variance there. The
        # fixed 0.28/0.85 constants compensated a quantile-map slope the mean-shift-only affine has
        # since removed, so they now under-shoot by construction (lv_pap 420 vs real 628).
        sp_ref = float(hp[hr >= thr_r].std()) if int((hr >= thr_r).sum()) > 100 else gstd
        sa_ref = float(hp[hr < thr_r].std()) if int((hr < thr_r).sum()) > 100 else gstd
        fpap = float(np.sqrt(max(sp_ref**2 - (0.87 * gstd)**2, (0.15 * gstd)**2))) / sp_
        fair = float(np.sqrt(max(sa_ref**2 - (1.0 * gstd)**2, (0.30 * gstd)**2))) / sa_
        det = (bulk * (cov_b * fpap + (1.0 - cov_b) * fair)
               + fleck * 0.8) * (1.0 - 0.75 * w_e)            # damped in-band: det would otherwise
        # re-insert the roughness ramp the widening just removed
        _st("base", bs_)
        # per-phase QUANTILE map at FULL amplitude (a 2-moment match cannot create the reference's heavy
        # bright tail: the mineral flecks live in that tail), base-to-base, blended by the coverage field.
        qs = np.linspace(0, 1, 512)

        def _sub(a, cap=300000):
            # deterministic stride subsample for KNOT estimation only (512 knots from 300k samples
            # are within ~0.2%); the per-voxel interp application stays exact and full-resolution
            return a if a.size <= cap else a.ravel()[:: max(a.size // cap, 1)]

        def _qmap(vals, refpart, qlo=0.0, qhi=1.0):
            if vals.size <= 10 or refpart.size <= 10:
                return None
            rq = np.quantile(_sub(refpart), qlo + (qhi - qlo) * qs)
            vq = np.quantile(_sub(vals), qs) + np.arange(512) * 1e-4    # strictly increasing knots
            return vq, rq

        # AIR SUB-PHASING (user defect: bright traces in airy cubes; measured air p99 78 vs real 52).
        # A single air<->air quantile map sends synth's brightest FAR air to the reference air-part's
        # TOP quantiles -- which are the reference's partial-volume EDGE voxels, at near-sheet
        # brightness. Far air must inherit the reference's DARK range, band air its PV range. Split
        # the air phase by coverage and give each half the matching quantile SEGMENT of the
        # reference air distribution; monotone overall, so ranks are preserved.
        pap_m = cov_b >= 0.5
        a_far = (cov_b < 0.05)
        a_band = (cov_b >= 0.05) & (cov_b < 0.5)
        n_air = max(int(a_far.sum() + a_band.sum()), 1)
        f_far = float(a_far.sum()) / n_air
        refair = ref[ref < thr]
        m_pap = _qmap(bs_[pap_m], ref[ref >= thr])
        m_far = _qmap(bs_[a_far], refair, 0.0, max(f_far, 0.05))
        m_band = _qmap(bs_[a_band], refair, max(f_far, 0.05), 1.0)
        if m_pap is not None and (m_far is not None or m_band is not None):
            pmapped = np.interp(bs_, m_pap[0], m_pap[1]).astype(np.float32)
            amapped = bs_.copy()
            if m_far is not None:
                amapped = np.where(a_far, np.interp(bs_, m_far[0], m_far[1]).astype(np.float32), amapped)
            if m_band is not None:
                amapped = np.where(a_band, np.interp(bs_, m_band[0], m_band[1]).astype(np.float32), amapped)
            ct = cov_b * pmapped + (1.0 - cov_b) * amapped + det
        else:
            ct = bs_ + det
        # EXPLICIT MINERAL-FLECK POINT PROCESS (advisor 2.3, tell #3: b>=210 deficit 10-57x vs real).
        # Dense inclusions are a physical population -- a sigma threshold on a residual cannot set a
        # population's density. Measure count density, size and contrast of the reference's own
        # flecks (CCs above p99.5) and inject the same population into the papyrus phase as
        # PSF-shaped bright blobs (impulses blurred at two size classes, peak-normalized).
        fl_thr = float(np.quantile(hr, 0.995))
        labf, nf = ndi.label(hr > fl_thr)
        if nf >= 3:
            szs = np.bincount(labf.ravel())[1:].astype(np.float32)
            amps = ndi.labeled_comprehension(hr, labf, np.arange(1, nf + 1),
                                             np.max, np.float32, fl_thr) - fl_thr
            amps = amps[amps > 0]
            d_m = float(np.sqrt(4.0 * szs.mean() / np.pi))
            n3 = int(round(nf / hr.size * H * W * depth / max(d_m, 1.0)))
            pi_, pj_, pk_ = np.nonzero(papv > 0.7)
            if amps.size >= 3 and n3 > 4 and pi_.size > n3:
                for sig_f, frac_f in ((0.8, 0.6), (1.3, 0.4)):
                    nn_f = int(n3 * frac_f)
                    if nn_f < 1:
                        continue
                    idxs = rng.integers(0, pi_.size, size=nn_f)
                    imp = np.zeros_like(ct)
                    imp[pi_[idxs], pj_[idxs], pk_[idxs]] = amps[rng.integers(0, amps.size, size=nn_f)]
                    ct = ct + ndi.gaussian_filter(imp, sig_f) * float((2 * np.pi) ** 1.5 * sig_f ** 3)
        _st("map", ct)
    if hist_ref is not None:
        # 3D grain carrying the reference grain's MEASURED radial power spectrum at gstd amplitude. The previous
        # per-slice roll/flip tiling had zero z-correlation, and the displayed cross-section is the (z,x) plane --
        # decorrelated rows doubled the fine-scale energy (synth lapvar 330-534 vs real 106-185 even after strong
        # blur). Recon grain is a stochastic field: its spectrum and amplitude are the physical properties to
        # preserve; the phases are random in the real scan too.
        nb = 48

        def _rpow(img):
            F = np.abs(np.fft.fft2(img - img.mean())) ** 2 / img.size   # per-pixel PSD, size-comparable
            fy2 = np.fft.fftfreq(img.shape[0])[:, None]; fx2 = np.fft.fftfreq(img.shape[1])[None, :]
            bi2 = np.minimum((np.hypot(fy2, fx2) * 2 * nb).astype(int), nb - 1)
            return (np.bincount(bi2.ravel(), F.ravel(), minlength=nb)
                    / np.maximum(np.bincount(bi2.ravel(), minlength=nb), 1))

        tgtP = _rpow(hp)
        zf = np.fft.fftfreq(ct.shape[0])[:, None, None]
        yf = np.fft.fftfreq(ct.shape[1])[None, :, None]
        xf = np.fft.rfftfreq(ct.shape[2])[None, None, :]
        rb3 = np.minimum(np.sqrt(zf * zf + yf * yf + xf * xf), 0.5).ravel() * 2 * nb - 0.5
        rg = np.random.default_rng(seed ^ 0x77)
        wn = rg.standard_normal(ct.shape).astype(np.float32)
        WN = _rfftn(wn)
        # fixed-point filter calibration on the SLICE spectrum: setting the 3D radial PSD to the 2D target
        # over-weights high frequency in any 2D cross-section (projection-slice integrates all fz shells),
        # which inflated pixel-scale grain energy inside sheets. Two rounds land within a few percent.
        Hr = np.sqrt(np.maximum(tgtP, 1e-8))
        for _ in range(2):
            ampH = np.interp(rb3, np.arange(nb), Hr).reshape(WN.shape).astype(np.float32)
            g3 = _irfftn(WN * ampH, ct.shape).astype(np.float32)
            mid = ct.shape[0] // 2
            curP = np.mean([_rpow(g3[z]) for z in range(mid - 2, mid + 3)], axis=0)
            Hr *= np.sqrt(np.clip(tgtP / np.maximum(curP, 1e-8), 0.0625, 16.0))
        g3 *= gstd / (g3.std() + 1e-6)
        # marginal remap to the reference grain's own quantiles: a gaussian-bodied field at the right spectrum
        # still reads "creamy" -- real recon grain is leptokurtic (sharp pinpoint pepper). Rank-preserving, so
        # the spatial correlation (spectrum) survives; amplitude distribution becomes the measured one.
        gq = np.quantile(hp.ravel()[::4], np.linspace(0, 1, 512))
        sq = np.quantile(g3.ravel()[::8], np.linspace(0, 1, 512)) + np.arange(512) * 1e-6
        g3 = np.interp(g3, sq, gq).astype(np.float32)
        ct = ct + g3 * (1.0 - 0.13 * cov_b)                  # pap 0.87 (det carries the rest), air 1.0
        if _os.environ.get("SC_DEBUG"):
            cs_ = ct[ct.shape[0] // 2]
            print(f"  [grain] vol std={ct.std():.1f} cs std={cs_.std():.1f} "
                  f"cs b215={(cs_ >= 215).mean():.4f} cs b230={(cs_ >= 230).mean():.4f} gstd={gstd:.1f}", flush=True)
    ct = np.clip(ct, 0, 255).astype(np.uint8)
    srcs = sorted({m["src"] for m in meta_if})
    return ct, inst, dict(seed=int(seed), sheets=len(meta_if), sources=srcs, violations=int(viol),
                          interfaces=meta_if)


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubes", required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ct, inst, meta = compose(a.cubes, a.seed)
    print(json.dumps({k: v for k, v in meta.items() if k != "interfaces"}, indent=1))
    for m in meta["interfaces"][:8]:
        print("  ", m)
