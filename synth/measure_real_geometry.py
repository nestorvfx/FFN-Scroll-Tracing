#!/usr/bin/env python3
"""Measure the REAL dense-region geometry distribution (Rank-1 augmentation improvement).

The synth corpus today has effectively ONE diversity axis (keep_air = linspace(0.30,0) + a fixed shear scalar). To
make the stratified/importance-weighted sampler REAL-FITTED rather than guessed, first measure what real dense
Herculaneum cubes actually look like:
  - INTER-SHEET GAP distribution: air-run lengths (CT<Otsu) between papyrus along the cross-sheet axis. Tells us the
    real residual-gap regime; the synth deliberately pushes BELOW this (toward fused) for the failure mode, so the
    easy end of keep_air should land near these gaps and the hard end at 0.
  - CONTACT AREA per cube: inter_sheet_boundary voxels / fiber voxels. The merger-critical cubes (large near-parallel
    contact) should get MORE variants (importance weight).
  - CURVATURE proxy: angular spread of the sheet-normal field -> caps deformation severity (Rank-2) at real values.
Outputs a JSON of aggregate percentiles + per-cube rows, and prints a summary. Pure numpy/scipy/skimage, CPU-parallel.

Usage: python measure_real_geometry.py --cubes <instance_cubes_dir> [--jobs 40] [--out real_geom.json]
"""
import os, glob, json, argparse
import numpy as np
from scipy import ndimage as ndi
import multiprocessing as mp
from synth_merge import load_cube, cross_sheet_axis, inter_sheet_boundary


def air_run_lengths(v_axis_last, otsu, stride=4, max_cols=4000):
    """Interior air-run (gap) lengths along the last axis, on a subsample of in-plane columns. Interior = between the
    first and last papyrus voxel of the column (ignores the outer air margins)."""
    pap = v_axis_last >= otsu
    flat = pap.reshape(-1, pap.shape[-1])
    n = flat.shape[0]
    sel = np.arange(0, n, stride)[:max_cols]
    runs = []
    for c in sel:
        p = flat[c]
        idx = np.flatnonzero(p)
        if idx.size < 2:
            continue
        interior = ~p[idx[0]:idx[-1] + 1]            # air voxels strictly between first/last papyrus
        if not interior.any():
            continue
        # run lengths of True (air) in `interior`
        d = np.diff(np.concatenate([[0], interior.view(np.int8), [0]]))
        starts = np.flatnonzero(d == 1); ends = np.flatnonzero(d == -1)
        runs.extend((ends - starts).tolist())
    return runs


def normal_angular_spread(inst, fiber, sample=20000, rng=None):
    """Curvature proxy: local angular deviation (deg) of the sheet-normal field. Normal = grad of signed distance to
    the nearest DIFFERENT structure; spread = angle between a voxel's normal and its 3x3x3-smoothed normal."""
    rng = rng or np.random.default_rng(0)
    sdf = ndi.distance_transform_edt(fiber) - ndi.distance_transform_edt(~fiber)
    gz, gy, gx = np.gradient(sdf.astype(np.float32))
    mag = np.sqrt(gz * gz + gy * gy + gx * gx) + 1e-6
    nz, ny, nx = gz / mag, gy / mag, gx / mag
    sz = ndi.uniform_filter(nz, 3); sy = ndi.uniform_filter(ny, 3); sx = ndi.uniform_filter(nx, 3)
    sm = np.sqrt(sz * sz + sy * sy + sx * sx) + 1e-6
    dot = np.clip((nz * sz + ny * sy + nx * sx) / sm, -1, 1)
    ang = np.degrees(np.arccos(dot))
    pts = np.flatnonzero(fiber.ravel())
    if pts.size > sample:
        pts = rng.choice(pts, sample, replace=False)
    return ang.ravel()[pts]


def edge_spread_L(vol, axis, otsu, stride=3, max_edges=6000, R=8):
    """Paganin-blur calibration: estimate the edge-spread regularization length L (voxels) from air->papyrus rising
    edges along the cross-sheet axis. The 1D LSF of the Lorentzian H=1/(1+(2*pi*L*f)^2) is (1/2L)exp(-|x|/L), whose
    ESF has 10-90% rise width = 2*ln(9)*L ~ 4.39*L  =>  L = width_1090 / 4.39. Returns a list of per-edge L. Used to
    set the [L_lo,L_hi] band that clamps the synthetic Paganin re-blur so it MATCHES (doesn't exceed) real edge-spread."""
    v = np.moveaxis(vol, axis, -1).astype(np.float32)
    flat = v.reshape(-1, v.shape[-1])
    N = flat.shape[-1]
    Ls = []
    for r in range(0, flat.shape[0], stride):
        p = flat[r]
        below = p < otsu
        # rising edges: index i where p[i-1]<otsu<=p[i], with air plateau before + papyrus plateau after
        for i in range(R, N - R):
            if below[i - 1] and not below[i]:
                air = float(np.median(p[i - R:i - 1]))
                pap = float(np.median(p[i + 1:i + R]))
                if pap - air < 25 or not below[i - R:i - 1].all() or below[i + 1:i + R].any():
                    continue                                   # require clean isolated monotone-ish edge w/ contrast
                w = p[i - R:i + R + 1]
                n = (w - air) / (pap - air + 1e-6)             # normalized 0..1 across the edge
                x = (np.arange(len(n)) - R).astype(np.float32)
                o = np.argsort(n)                              # np.interp REQUIRES monotonic xp (real edges are noisy)
                x10 = np.interp(0.1, n[o], x[o]); x90 = np.interp(0.9, n[o], x[o])
                if x90 > x10 and (x90 - x10) < 2 * R:
                    Ls.append((x90 - x10) / 3.2189)            # 2*ln5: Lorentzian-LSF ESF 10-90 rise width = 2L*ln5
                if len(Ls) >= max_edges:
                    return Ls
    return Ls


def measure_one(cube_dir):
    try:
        from skimage.filters import threshold_otsu
        vol, inst = load_cube(cube_dir)
        otsu = float(threshold_otsu(vol))
        axis = cross_sheet_axis(inst)
        fiber = vol >= otsu
        v = np.moveaxis(vol, axis, -1)
        runs = air_run_lengths(v, otsu)
        contact = int(inter_sheet_boundary(inst).sum())
        fib = int(fiber.sum())
        ninst = int(len([i for i in np.unique(inst) if i != 0]))
        ang = normal_angular_spread(inst, fiber)
        esf = edge_spread_L(vol, axis, otsu)
        return dict(cube=os.path.basename(cube_dir.rstrip('/')), axis=axis, n_inst=ninst,
                    fiber_frac=round(100 * fib / fiber.size, 2),
                    contact_frac=round(100 * contact / max(1, fib), 3),
                    gap_runs=runs, esf_L=esf, ang_p=[round(float(np.percentile(ang, p)), 1) for p in (50, 90, 99)])
    except Exception as e:
        return dict(cube=os.path.basename(cube_dir.rstrip('/')), error=str(e)[:80])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cubes', required=True)
    ap.add_argument('--jobs', type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument('--out', default='/root/surf/data/real_geom.json')
    a = ap.parse_args()
    cubes = sorted(glob.glob(os.path.join(a.cubes, '*/')))
    with mp.Pool(a.jobs) as pool:
        rows = pool.map(measure_one, cubes)
    ok = [r for r in rows if 'error' not in r]
    all_gaps = np.array([g for r in ok for g in r['gap_runs']]) if ok else np.array([0])
    cfrac = np.array([r['contact_frac'] for r in ok])
    ffrac = np.array([r['fiber_frac'] for r in ok])
    ninst = np.array([r['n_inst'] for r in ok])
    ang90 = np.array([r['ang_p'][1] for r in ok])
    all_esf = np.array([l for r in ok for l in r.get('esf_L', [])]) if ok else np.array([])
    pct = lambda x, ps: {str(p): round(float(np.percentile(x, p)), 2) for p in ps} if len(x) else {}
    agg = dict(n_cubes=len(ok), n_failed=len(rows) - len(ok),
               gap_vox=pct(all_gaps, [10, 25, 50, 75, 90, 99]),
               gap_mean=round(float(all_gaps.mean()), 2), gap_n=int(all_gaps.size),
               contact_frac_pct=pct(cfrac, [10, 25, 50, 75, 90]),
               fiber_frac_pct=pct(ffrac, [10, 50, 90]),
               n_inst_pct=pct(ninst, [10, 50, 90]),
               normal_ang90_pct=pct(ang90, [50, 90, 99]),
               esf_L_vox=pct(all_esf, [10, 25, 50, 75, 90]), esf_n=int(all_esf.size))
    # strip the bulky per-run lists from the saved per-cube rows (keep summaries)
    for r in ok:
        r['gap_n'] = len(r.pop('gap_runs', [])); r['esf_n'] = len(r.pop('esf_L', []))
    json.dump(dict(agg=agg, per_cube=ok), open(a.out, 'w'), indent=2)
    print(json.dumps(agg, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == '__main__':
    main()
