#!/usr/bin/env python3
"""Pre-harvest all source cubes' sheet banks ONCE and pickle them (SC_BANK cache for compose()).

extract_sheets dominates compose() wall time (per-column Python loops, ~3-6 min per 3-cube draw) and is
re-run for every composition despite being deterministic per source cube. One pass here (parallel), then
compose() loads the pickle in ~2s. Output identical by construction.

Usage: python preharvest.py --cubes <dir> --out /root/data/bank --jobs 40
"""
import os, sys, glob, argparse, pickle
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_merge import load_cube, cross_sheet_axis
from sheet_compose import extract_sheets


def one(job):
    cd, out = job
    cname = os.path.basename(cd.rstrip("/"))
    pk = os.path.join(out, cname + ".pkl")
    if os.path.exists(pk):
        return cname, "cached"
    vol, msk = load_cube(cd)
    ax = cross_sheet_axis(msk)
    sh, air = extract_sheets(vol, msk, ax)
    tmp = pk + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump((sh, air), f, protocol=4)
    os.replace(tmp, pk)
    return cname, f"{len(sh)} sheets"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubes", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=40)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dirs = sorted(glob.glob(os.path.join(a.cubes, "*/")))
    with mp.Pool(a.jobs) as p:
        for i, (cname, msg) in enumerate(p.imap_unordered(one, [(d, a.out) for d in dirs])):
            print(f"[{i + 1}/{len(dirs)}] {cname}: {msg}", flush=True)
    print("PREHARVEST_DONE")


if __name__ == "__main__":
    main()
