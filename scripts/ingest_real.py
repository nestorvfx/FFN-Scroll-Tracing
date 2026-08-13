#!/usr/bin/env python
"""Ingest the REAL harmonized instance cubes into the FFN corpus (AUDIT FIX 1).

The official Scroll1 `volumetric-instance-labels/instance-labels-harmonized` set is 80
real 256^3 cubes, each a `<coord>_volume.nrrd` (uint16 CT) + `<coord>_mask.nrrd`
(harmonized instance ids). These contain REAL blind contacts with REAL CT appearance --
the exact hard signal the merge-averse FFN needs, which synth fusion only approximates.
The implementation trained synth-only on a false "not published" assumption; this script
fixes that.

What it does, per cube:
  1. (optional) rclone-fetch the cube dir from ash2txt.
  2. Harmonize CT: linear intensity map so the cube's (background-median, foreground-p95)
     anchors match the held-out eval slab's -- i.e. real cubes end up looking like the
     slab (and the slab is what the synth corpus was already calibrated to). uint16 -> uint8.
  3. Tile into up to `--crops-per-cube` informative 192^3 crops (matching the synth cube
     shape exactly, required by the GPU-resident stack), keeping the crops richest in
     instances. Crops inherit the parent coord so the split can hold out whole cubes.
  4. Write into the corpus as `real_<coord>_c<k>_0000.tif` (uint8) + `.tif` (uint16 inst),
     and record `real_meta.json` with source="real" and the parent coord.

preprocess.py merges real_meta.json, holds out `real_val_n` PARENT cubes for the real
val gate, and flags real candidates (is_real) so training can mix them into Phase C.

Usage (on the box):
  python -m scripts.ingest_real --corpus /root/surf/data/synthfuse_corpus \
      --slab /root/data/slab --stage /root/real_stage --fetch [--limit N]
"""
import argparse
import functools
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import nrrd
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffn.config import FFNConfig

ASH = "http://<USERNAME>:<PASSWORD>@dl.ash2txt.org/"
REMOTE = (":http:/full-scrolls/Scroll1/PHercParis4.volpkg/"
          "volumetric-instance-labels/instance-labels-harmonized/")
CROP = 192   # must equal the synth cube size (GPU-resident stack needs one shape)


def slab_anchors(slab_dir):
    vol, _ = nrrd.read(os.path.join(slab_dir, "volume.nrrd"))
    truth, _ = nrrd.read(os.path.join(slab_dir, "truth.nrrd"))
    fg = truth > 0
    bg = float(np.median(vol[~fg])) if (~fg).any() else 0.0
    fg95 = float(np.percentile(vol[fg], 95)) if fg.any() else 255.0
    return bg, fg95


def harmonize(vol_u16, mask, slab_bg, slab_fg95):
    """Linear map real uint16 CT -> uint8, matching (bg-median, fg-p95) to the slab."""
    fg = mask > 0
    rbg = float(np.median(vol_u16[~fg])) if (~fg).any() else 0.0
    rfg95 = float(np.percentile(vol_u16[fg], 95)) if fg.any() else float(vol_u16.max())
    a = (slab_fg95 - slab_bg) / max(1.0, rfg95 - rbg)
    out = (vol_u16.astype(np.float32) - rbg) * a + slab_bg
    return np.clip(out, 0, 255).astype(np.uint8)


def list_cubes():
    r = subprocess.run(["rclone", "lsf", "--http-url", ASH, REMOTE, "--dirs-only"],
                       capture_output=True, text=True)
    return sorted(d.strip("/") for d in r.stdout.split("\n") if d.strip())


def fetch_all(stage, transfers=32):
    """One bulk rclone of the WHOLE harmonized dir -> all cubes download in parallel
    (skips already-present files). Vastly faster than one rclone per cube."""
    os.makedirs(stage, exist_ok=True)
    subprocess.run(["rclone", "copy", "--http-url", ASH, REMOTE, stage,
                    "--transfers", str(transfers), "--multi-thread-streams", "4",
                    "--fast-list", "--stats", "10s", "--stats-one-line"], check=True)


def informative_crops(mask, n_crops, min_size, crop=CROP):
    """Pick up to n_crops offsets whose 192^3 window holds the most instances
    (>= min_size), diverse in position (L-inf >= crop//4 apart)."""
    Z, Y, X = mask.shape
    axis_offsets = sorted(set([0, (Z - crop) // 2, Z - crop]))   # 0, center, far
    cands = []
    for oz in axis_offsets:
        for oy in axis_offsets:
            for ox in axis_offsets:
                sub = mask[oz:oz + crop, oy:oy + crop, ox:ox + crop]
                ids, cnts = np.unique(sub, return_counts=True)
                n_inst = int(sum(1 for i, c in zip(ids, cnts) if i != 0 and c >= min_size))
                cands.append(((oz, oy, ox), n_inst))
    cands.sort(key=lambda t: -t[1])
    picked = []
    for off, n_inst in cands:
        if n_inst < 2:
            continue
        if all(max(abs(off[i] - p[i]) for i in range(3)) >= crop // 4 for p in picked):
            picked.append(off)
        if len(picked) >= n_crops:
            break
    if not picked and cands:                        # fall back to the single best
        picked = [cands[0][0]]
    return picked


def process_one(coord, stage, corpus, slab_bg, slab_fg95, crops, min_size):
    """Harmonize one real cube and write its informative 192^3 crops. Independent per
    cube -> safe to run in a process pool (distinct output filenames). Returns meta dict."""
    d = os.path.join(stage, coord)
    vpath = os.path.join(d, f"{coord}_volume.nrrd")
    mpath = os.path.join(d, f"{coord}_mask.nrrd")
    if not (os.path.exists(vpath) and os.path.exists(mpath)):
        return {}
    vol, _ = nrrd.read(vpath)
    mask, _ = nrrd.read(mpath)
    img8 = harmonize(vol, mask, slab_bg, slab_fg95)
    meta = {}
    for k, off in enumerate(informative_crops(mask, crops, min_size)):
        oz, oy, ox = off
        sub_img = img8[oz:oz + CROP, oy:oy + CROP, ox:ox + CROP]
        sub_msk = mask[oz:oz + CROP, oy:oy + CROP, ox:ox + CROP].astype(np.uint16)
        rid = f"real_{coord}_c{k}"
        tifffile.imwrite(os.path.join(corpus, "imagesTr", f"{rid}_0000.tif"), sub_img)
        tifffile.imwrite(os.path.join(corpus, "labelsTr_inst", f"{rid}.tif"), sub_msk)
        fg = float((sub_msk > 0).mean())
        ids, cnts = np.unique(sub_msk, return_counts=True)
        sheets = int(sum(1 for i, c in zip(ids, cnts) if i != 0 and c >= min_size))
        meta[rid] = {"source": "real", "parent": coord, "fg": round(fg, 3),
                     "sheets": sheets, "stratum": "deep" if fg > 0.5 else "std"}
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="synth corpus dir to add real_ cubes to")
    ap.add_argument("--slab", required=True, help="slab dir for intensity anchors")
    ap.add_argument("--stage", default="/root/real_stage")
    ap.add_argument("--fetch", action="store_true", help="bulk rclone-fetch cubes first")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--crops-per-cube", type=int, default=2)
    ap.add_argument("--min-size", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=min(32, (os.cpu_count() or 8)),
                    help="parallel workers for the harmonize/tile step")
    args = ap.parse_args()

    cfg = FFNConfig()
    min_size = args.min_size if args.min_size is not None else cfg.min_instance_size
    os.makedirs(args.stage, exist_ok=True)
    os.makedirs(os.path.join(args.corpus, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(args.corpus, "labelsTr_inst"), exist_ok=True)

    slab_bg, slab_fg95 = slab_anchors(args.slab)
    print(f"[anchors] slab bg-median={slab_bg:.1f} fg-p95={slab_fg95:.1f}", flush=True)

    if args.fetch:
        print("[fetch] bulk rclone (32 parallel transfers) ...", flush=True)
        fetch_all(args.stage, transfers=args.jobs)
    coords = sorted(d for d in os.listdir(args.stage)
                    if os.path.isdir(os.path.join(args.stage, d)))
    if args.limit:
        coords = coords[:args.limit]
    print(f"[ingest] {len(coords)} real cubes, {args.jobs} workers", flush=True)

    fn = functools.partial(process_one, stage=args.stage, corpus=args.corpus,
                           slab_bg=slab_bg, slab_fg95=slab_fg95,
                           crops=args.crops_per_cube, min_size=min_size)
    meta = {}
    done = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(fn, c): c for c in coords}
        for f in as_completed(futs):
            meta.update(f.result())
            done += 1
            if done % 10 == 0 or done == len(coords):
                print(f"[ingest] {done}/{len(coords)} cubes -> {len(meta)} crops", flush=True)

    out = os.path.join(args.corpus, "real_meta.json")
    json.dump(meta, open(out, "w"), indent=2)
    parents = len(set(v["parent"] for v in meta.values()))
    print(f"[ingest] wrote {len(meta)} real crops from {parents} cubes -> {out}", flush=True)


if __name__ == "__main__":
    main()
