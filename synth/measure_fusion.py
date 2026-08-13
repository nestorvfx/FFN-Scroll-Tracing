#!/usr/bin/env python3
"""Invisible-contact-fraction metric (Fable-5 audit issue 1).

A merger happens where two wraps touch with NO local intensity evidence -- the model can't see a boundary, so it
fuses them. This measures, for every GT contact between two different sheet instances, the cross-normal intensity
DIP (how much darker the boundary is than the two cores it separates), in units of the local grain sigma. The
"invisible-contact fraction" is the share of contact area whose dip is below ~1 grain sigma: a real fused region
is mostly invisible contacts, our pre-fix synth was ~all dim-line contacts.

Runs on synthetic compose() output (contacts known exactly from `inst`); the same routine, given a real labelled
volume + its instance ids, yields the TARGET (see measure_fusion_real in archive/synth/slab_merger_eval.py).

Usage: python measure_fusion.py --cubes <dir> --tif <tif> --seeds 200,201,202,203  [--nofuse]
"""
import os, sys, argparse
import numpy as np
import tifffile
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sheet_compose import compose
from render_blind import ref_windows


def grain_sigma_of(ct):
    """Robust local grain sigma = 1.4826 * MAD of the sigma-1.2 high-pass residual (same grain band the
    composer's g3 matches). MAD is insensitive to the sparse large edge responses."""
    hp = ct.astype(np.float32) - ndi.gaussian_filter(ct.astype(np.float32), 1.2)
    return float(1.4826 * np.median(np.abs(hp - np.median(hp))) + 1e-6)


def contact_dips(ct, inst, axis=2, maxgap=4, core=3, grain_sigma=7.3):
    """Two DIFFERENT wraps separated by a thin gap along `axis` are a merger risk: if the CT shows no intensity
    dip across the gap, the model has no local cue and fuses them. For each such near-contact (gap of inst==0 of
    width 1..maxgap between run A and run B, A!=B), measure dip = min(coreA,coreB) - min(gap CT). Small dip =
    locally INVISIBLE boundary = the hard case the carve label must teach. Per-column scan along the normal.

    A gap of 0 (cores directly adjacent) counts with gap-value = the boundary voxels themselves."""
    a = np.moveaxis(inst, axis, 0)
    xa = np.moveaxis(ct.astype(np.float32), axis, 0)
    N, H, W = a.shape
    cols = np.moveaxis(a, 0, -1).reshape(-1, N)            # (H*W, N)
    xcols = np.moveaxis(xa, 0, -1).reshape(-1, N)
    have = (cols > 0).sum(1) >= 2
    dips = []
    for c in np.nonzero(have)[0]:
        col = cols[c]; xc = xcols[c]
        # run-length encode the instance profile
        idx = np.nonzero(np.diff(col))[0] + 1
        starts = np.concatenate(([0], idx)); ends = np.concatenate((idx, [N]))
        vals = col[starts]
        for r in range(len(vals) - 1):
            # A run (vals[r]>0), optional single 0-gap run, then B run (>0), A!=B
            if vals[r] <= 0:
                continue
            if vals[r + 1] > 0:                            # directly adjacent A|B (gap 0)
                aid, bid, ke = vals[r], vals[r + 1], ends[r]
                if aid == bid:
                    continue
                g0, g1 = ke, ke                            # boundary voxels ke-1,ke
                gapmin = min(xc[ke - 1], xc[ke])
            elif r + 2 < len(vals) and vals[r + 2] > 0 and vals[r + 2] != vals[r] \
                    and (ends[r + 1] - starts[r + 1]) <= maxgap:
                aid, bid = vals[r], vals[r + 2]
                g0, g1 = starts[r + 1], ends[r + 1]
                gapmin = xc[g0:g1].min()
            else:
                continue
            core_a = xc[max(starts[r], ends[r] - core):ends[r]].max()
            bs = starts[r + 1] if vals[r + 1] > 0 else starts[r + 2]
            be = ends[r + 1] if vals[r + 1] > 0 else ends[r + 2]
            core_b = xc[bs:min(be, bs + core)].max()
            dips.append(min(core_a, core_b) - gapmin)
    dips = np.array(dips, np.float32)
    return dips, (dips / grain_sigma if dips.size else dips)


def summarize(tag, ct, inst, axes=(2,), gsig=None):
    gsig = gsig if gsig is not None else grain_sigma_of(ct)
    dips = np.concatenate([contact_dips(ct, inst, axis=ax, grain_sigma=gsig)[0] for ax in axes]) \
        if len(axes) > 1 else contact_dips(ct, inst, axis=axes[0], grain_sigma=gsig)[0]
    if dips.size == 0:
        print(f"{tag}: no contacts"); return None
    dn = dips / gsig
    inv = float((dn < 1.0).mean())                          # dip < 1 grain sigma == locally invisible
    faint = float((dn < 2.0).mean())
    print(f"{tag}: contacts={dips.size:6d} gsig={gsig:4.1f} dip med={np.median(dips):5.1f} "
          f"p10/p90={np.percentile(dips,10):5.1f}/{np.percentile(dips,90):5.1f}  "
          f"invisible(<1s)={inv:.3f}  faint(<2s)={faint:.3f}")
    return inv


def real_slab_target(slab_dir):
    """Invisible-contact-fraction TARGET measured on the real z=10192 slab truth (68 traced wraps). Wraps wind
    around the umbilicus, so adjacent wraps meet along the in-plane radial direction -> measure axes y(1) and
    x(2), pool. This is what the composer's fused contacts should match."""
    import nrrd
    truth = nrrd.read(os.path.join(slab_dir, "truth.nrrd"))[0].astype(np.int32)
    vol = nrrd.read(os.path.join(slab_dir, "volume.nrrd"))[0].astype(np.float32)
    lo, hi = np.percentile(vol, [1, 99.5]); vol = np.clip((vol - lo) / (hi - lo + 1e-6) * 255, 0, 255)
    print(f"slab {truth.shape}, {int((np.unique(truth) > 0).sum())} wraps")
    return summarize("REAL slab (y+x)", vol, truth, axes=(1, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubes"); ap.add_argument("--tif")
    ap.add_argument("--seeds", default="200,201,202,203")
    ap.add_argument("--slab", default=None, help="measure the real-slab target instead of synth")
    a = ap.parse_args()
    if a.slab:
        real_slab_target(a.slab); return
    img = tifffile.imread(a.tif).astype(np.float32)
    lo, hi = np.percentile(img, [1, 99.5]); img = np.clip((img - lo) / (hi - lo + 1e-6) * 255, 0, 255)
    refs = ref_windows(img, 8, np.random.default_rng(4242))
    seeds = [int(s) for s in a.seeds.split(",")]
    invs = []
    for si, s in enumerate(seeds):
        ct, inst, meta = compose(a.cubes, s, hist_ref=refs[si % len(refs)])
        invs.append(summarize(f"seed {s} ({'NOFUSE' if os.environ.get('SC_NOFUSE') else 'fuse'})", ct, inst))
    invs = [v for v in invs if v is not None]
    if invs:
        print(f"MEAN invisible-contact fraction: {np.mean(invs):.3f}")


if __name__ == "__main__":
    main()
