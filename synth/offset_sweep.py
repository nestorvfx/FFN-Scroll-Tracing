#!/usr/bin/env python3
"""Where does the NEXT WRAP actually live? Sweep offset radius and measure the cross-wrap rate.

Motivation: the affinity offsets are {1,3,9,27} axis-aligned, chosen as a generic log-spaced ladder. But the
decision we need the network to make is "is the fiber voxel one lamination period away the NEXT wrap or the SAME
wrap?", and that question is only ASKED by an offset whose length matches the period P. Measured per-offset
same-rates pi_k on the corpus were [0.95 0.95 0.92 | 0.87 0.86 0.75 | 0.68 0.64 0.17 | 0.41 0.32 0.00] -- i.e.
r=1/3 are almost pure "same" (they carry the boundary but not the decision) and one r=27 channel is a CONSTANT
target. So the ladder may be probing everywhere except where the decision is.

For each radius r and each direction d we report, over VALID (fiber->fiber) edges only:
    cross_rate(r,d) = P(different wrap | both endpoints fiber)
The radius maximizing cross_rate is the "next wrap" band. A radius whose cross_rate is ~0 or ~1 is a near-constant
channel: it consumes capacity and distorts the global class balance while contributing no discriminative gradient.
`n_valid` matters too -- a high cross-rate over a handful of edges is noise, not a usable channel.

Directions: axis-aligned plus in-plane diagonals (unit-normalized to length r), because sheet normals in a spiral
are only axis-aligned for a minority of the wrap; an axis-aligned offset reaches r*cos(theta) along the normal.

Usage:
  python offset_sweep.py --corpus /root/surf/data/synthfuse_corpus --n 6 --rmax 40
  python offset_sweep.py --slab /root/data/slab --rmax 40
"""
import os, sys, glob, json, argparse
import numpy as np


def _dirs(r, wind_axis):
    """Unit directions scaled to radius r. In-plane = the two axes perpendicular to the winding axis."""
    ax = [a for a in range(3) if a != wind_axis]
    out = {}
    for a in ax:                                          # in-plane axis-aligned
        v = [0, 0, 0]; v[a] = r
        out[f"ip_ax{a}"] = v
    s = int(round(r / np.sqrt(2)))                        # in-plane diagonals, same Euclidean length
    if s > 0:
        for sg in (1, -1):
            v = [0, 0, 0]; v[ax[0]] = s; v[ax[1]] = s * sg
            out[f"ip_diag{'+' if sg > 0 else '-'}"] = v
    v = [0, 0, 0]; v[wind_axis] = r                       # along the winding axis, for contrast
    out["wind"] = v
    return out


def cross_rate(inst, off):
    dz, dy, dx = off
    Z, Y, X = inst.shape
    z0, z1 = max(0, -dz), Z - max(0, dz)
    y0, y1 = max(0, -dy), Y - max(0, dy)
    x0, x1 = max(0, -dx), X - max(0, dx)
    if z1 <= z0 or y1 <= y0 or x1 <= x0:
        return 0, 0
    a = inst[z0:z1, y0:y1, x0:x1]
    b = inst[z0 + dz:z1 + dz, y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    v = (a > 0) & (b > 0)
    return int(v.sum()), int((v & (a != b)).sum())


def sweep(insts, rmax, wind_axis, step):
    rows = []
    for r in range(1, rmax + 1, step):
        for name, off in _dirs(r, wind_axis).items():
            tot = cx = 0
            for inst in insts:
                t, c = cross_rate(inst, off)
                tot += t; cx += c
            rows.append(dict(r=r, dir=name, off=off, n_valid=tot,
                             cross_rate=round(cx / tot, 4) if tot else None))
    return rows


def report(tag, rows):
    print(f"\n=== {tag} ===")
    print(f"{'r':>3} {'dir':<10} {'n_valid':>11} {'cross_rate':>10}   (cross_rate = P(different wrap | both fiber))")
    for x in rows:
        flag = ""
        if x["cross_rate"] is not None and x["n_valid"] > 1000:
            if x["cross_rate"] < 0.02:
                flag = "  <- near-constant SAME (no decision asked)"
            elif x["cross_rate"] > 0.98:
                flag = "  <- near-constant DIFFERENT"
        print(f"{x['r']:>3} {x['dir']:<10} {x['n_valid']:>11} "
              f"{('%.4f' % x['cross_rate']) if x['cross_rate'] is not None else '   n/a':>10}{flag}")
    ip = [x for x in rows if x["dir"].startswith("ip_") and x["n_valid"] > 5000 and x["cross_rate"] is not None]
    if ip:
        best = max(ip, key=lambda x: x["cross_rate"])
        # the most INFORMATIVE band is the one nearest a balanced 50/50 split, not the max
        bal = min(ip, key=lambda x: abs(x["cross_rate"] - 0.5))
        print(f"\n{tag}: max in-plane cross-rate  r={best['r']} {best['dir']} -> {best['cross_rate']}")
        print(f"{tag}: most BALANCED in-plane    r={bal['r']} {bal['dir']} -> {bal['cross_rate']}  "
              f"<- highest-information band; this is where the 'next wrap' decision lives")
        wind = [x for x in rows if x["dir"] == "wind" and x["n_valid"] > 5000 and x["cross_rate"] is not None]
        if wind:
            print(f"{tag}: winding-axis cross-rate range "
                  f"{min(w['cross_rate'] for w in wind):.4f}..{max(w['cross_rate'] for w in wind):.4f}"
                  f"  <- if flat/low, winding-axis offsets beyond r=1 are dead channels")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus"); ap.add_argument("--slab")
    ap.add_argument("--n", type=int, default=6); ap.add_argument("--rmax", type=int, default=40)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--wind_axis", type=int, default=0, help="axis along the winding/scroll direction (default z=0)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = {}
    if a.slab:
        import nrrd
        truth = nrrd.read(os.path.join(a.slab, "truth.nrrd"))[0].astype(np.int32)
        rows = sweep([truth], a.rmax, a.wind_axis, a.step)
        report("REAL slab", rows); out["real_slab"] = rows
    if a.corpus:
        import tifffile
        insts = [tifffile.imread(f).astype(np.int32)
                 for f in sorted(glob.glob(f"{a.corpus}/labelsTr_inst/*.tif"))[:a.n]]
        rows = sweep(insts, a.rmax, a.wind_axis, a.step)
        report("SYNTH corpus", rows); out["synth"] = rows
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    print("\nOFFSET_SWEEP_DONE")


if __name__ == "__main__":
    main()
