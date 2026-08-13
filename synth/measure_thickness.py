#!/usr/bin/env python3
"""Per-sheet THICKNESS distribution: REAL wound scroll (the 68-wrap slab truth = actual compact scroll data) vs
REAL source cubes (the 80 harmonized instance cubes, pre-augmentation) vs our AUGMENTED corpus (after
compose+squeeze+carve). Answers: are we thinning sheets BELOW their real wound-region thickness (losing data),
and how much does the binary carve then remove?

Thickness metric (consistent across all three): per instance, isolate its mask, EDT -> local thickness = 2*EDT.
Report the distribution of the per-instance MEDIAN local thickness (typical sheet thickness), plus the p10 (thin
tail). EDT is on the ISOLATED sheet so touching neighbors don't inflate it.

Usage:
  python measure_thickness.py --slab /root/data/slab                       # real wound reference
  python measure_thickness.py --cubes /root/data/cubes/instance-labels-harmonized   # real source
  python measure_thickness.py --corpus /root/surf/data/synthfuse_corpus [--carve]    # augmented (opt: post-carve)
"""
import os, sys, glob, json, argparse
import numpy as np
from scipy import ndimage as ndi


def sheet_thicknesses(inst, min_vox=300, max_sheets=200):
    """Per-instance median local thickness (2*EDT on the isolated mask). Returns list of medians + p10s.
    EDT is computed on the per-instance BOUNDING BOX (+1 pad) so a full-volume EDT is never run per sheet."""
    ids = np.unique(inst[inst > 0])
    locs = ndi.find_objects(inst.astype(np.int32)) if inst.max() < 5_000_000 else None
    meds, p10s = [], []
    for i in ids[:max_sheets]:
        if locs is not None and i - 1 < len(locs) and locs[i - 1] is not None:
            sl = tuple(slice(max(0, s.start - 1), s.stop + 1) for s in locs[i - 1])
            m = inst[sl] == i
        else:
            m = inst == i
        if m.sum() < min_vox:
            continue
        edt = ndi.distance_transform_edt(m)
        t = 2.0 * edt[m]                      # local thickness at each voxel
        # medial ridge = the thick core; typical sheet thickness = median over the top-half EDT voxels
        ridge = t[t >= np.percentile(t, 50)]
        meds.append(float(np.median(ridge)))
        p10s.append(float(np.percentile(t, 10)))
    return meds, p10s


def report(tag, meds, p10s):
    if not meds:
        print(f"{tag}: no sheets"); return
    m = np.array(meds)
    print(f"{tag}: n_sheets={len(m)}  median-thickness  p10/p25/p50/p75/p90 = "
          f"{np.percentile(m,10):.1f}/{np.percentile(m,25):.1f}/{np.percentile(m,50):.1f}/"
          f"{np.percentile(m,75):.1f}/{np.percentile(m,90):.1f} vox  |  min={m.min():.1f} max={m.max():.1f}")
    return dict(n=len(m), p10=round(float(np.percentile(m,10)),1), p50=round(float(np.median(m)),1),
                p90=round(float(np.percentile(m,90)),1), min=round(float(m.min()),1))


def from_slab(slab):
    import nrrd
    truth = nrrd.read(os.path.join(slab, "truth.nrrd"))[0].astype(np.int32)
    meds, p10s = sheet_thicknesses(truth)
    return report("REAL WOUND (slab 68 wraps)", meds, p10s)


def from_cubes(cubes):
    import nrrd
    ds = sorted(glob.glob(cubes + "/*/"))[:25]
    allm = []
    for d in ds:
        mk = glob.glob(d + "/*_mask.nrrd") or glob.glob(d + "/*mask*.nrrd")
        if not mk:
            # harmonized cubes may store the instance labels directly
            mk = [f for f in glob.glob(d + "/*.nrrd") if "volume" not in f.lower()]
        if not mk:
            continue
        inst = nrrd.read(mk[0])[0].astype(np.int32)
        m, _ = sheet_thicknesses(inst)
        allm += m
    return report("REAL SOURCE (harmonized cubes)", allm, [])


def from_corpus(corpus, carve=False):
    import tifffile
    META = json.load(open(f"{corpus}/synthfuse_meta.json")) if os.path.exists(f"{corpus}/synthfuse_meta.json") else {}
    insts = sorted(glob.glob(f"{corpus}/labelsTr_inst/*.tif"))
    for strat in ("deep", "std"):
        sel = [f for f in insts if META.get(os.path.basename(f)[:-4], {}).get("stratum", "std") == strat][:25]
        allm = []
        for f in sel:
            inst = tifffile.imread(f).astype(np.int32)
            if carve:
                cid = os.path.basename(f)[:-4]
                lab = tifffile.imread(f"{corpus}/labelsTr/{cid}.tif")
                inst = np.where(lab > 0, inst, 0)          # thickness of the SURVIVING (post-carve) fiber
            m, _ = sheet_thicknesses(inst)
            allm += m
        report(f"AUGMENTED {strat}{' POST-CARVE' if carve else ''}", allm, [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slab"); ap.add_argument("--cubes"); ap.add_argument("--corpus")
    ap.add_argument("--carve", action="store_true")
    a = ap.parse_args()
    if a.slab: from_slab(a.slab)
    if a.cubes: from_cubes(a.cubes)
    if a.corpus:
        from_corpus(a.corpus, carve=False)
        from_corpus(a.corpus, carve=True)     # also show post-carve thickness (what the binary head sees)
    print("THICKNESS_DONE")


if __name__ == "__main__":
    main()
