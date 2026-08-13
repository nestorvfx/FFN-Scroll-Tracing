#!/usr/bin/env python
"""Whole-volume production decode.

  python -m tracer.run --vol scroll_chunk.tif --out seg.npy \
      [--raw] [--block 192 --halo 48] [--no-barrier] [--device cuda:1]

--raw: input is raw CT (uint8/16) -> harmonize to slab units first (production).
Without --raw the input is assumed to already be in slab units (training corpus cubes).

Outputs: <out>.npy int32 labels + <out>.stats.json (per-block fills, truncations,
stitch matches -- the provenance production needs to audit a decode after the fact).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/root/ds_probe/pylibs")     # imagecodecs BEFORE tifffile
import tifffile                                  # noqa: E402

sys.path.insert(0, "/root/surface_detection/FFN")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.evaluate import load_model, make_predict_fn   # noqa: E402

from tracer.config import TracerConfig          # noqa: E402
from tracer.blocks import tile                  # noqa: E402
from tracer.decode import decode_block          # noqa: E402
from tracer.normalize import harmonize          # noqa: E402
from tracer.stitch import assemble              # noqa: E402


def load_volume(path):
    if path.endswith(".npy"):
        return np.load(path)
    return tifffile.imread(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--halo", type=int, default=None)
    ap.add_argument("--no-barrier", action="store_true")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    tc = TracerConfig().load_anchors()
    if args.block:
        tc.block = args.block
    if args.halo is not None:
        tc.halo = args.halo
    if args.no_barrier:
        tc.barrier = False
    if args.tta:
        tc.tta_rot = True
    if args.ckpt:
        tc.ckpt = args.ckpt
    if args.device:
        tc.device = args.device

    import torch
    dev = torch.device(tc.device)
    model, cfg = load_model(tc.ckpt, dev)
    import dataclasses
    cfg = dataclasses.replace(cfg, fill_max_steps=tc.fill_max_steps,
                              consensus_scales=[1])
    pf = make_predict_fn(model, cfg, dev)

    vol = load_volume(args.vol)
    if args.raw:
        if not np.isfinite(tc.slab_bg):
            raise SystemExit("anchors.json missing -- run tracer.normalize.calibrate first")
        vol = harmonize(vol, tc.slab_bg, tc.slab_fg95)
    elif vol.dtype != np.uint8:
        raise SystemExit("non-uint8 input without --raw; refusing to guess units")

    blocks = tile(vol.shape, tc.block, tc.halo)
    labels, all_stats = [], []
    t0 = time.time()
    for bi, b in enumerate(blocks):
        cz0, cy0, cx0, cz1, cy1, cx1 = b.canvas
        canvas = vol[cz0:cz1, cy0:cy1, cx0:cx1]
        lab, st = decode_block(pf, canvas, cfg, tc)
        labels.append(lab)
        st["core"] = list(b.core)
        all_stats.append(st)
        print(f"[block {bi + 1}/{len(blocks)}] objects={st['objects']} "
              f"seeds={st['seeds']} truncated={st['truncated']} "
              f"lam={st['lam']} s={st['s']} mode={st['mode']} "
              f"({time.time() - t0:.0f}s)", flush=True)

    glob, n_obj, n_match = assemble(vol.shape, blocks, labels,
                                    min_overlap=tc.stitch_min_overlap)
    np.save(args.out if args.out.endswith(".npy") else args.out + ".npy", glob)
    meta = dict(vol=args.vol, ckpt=tc.ckpt, block=tc.block, halo=tc.halo,
                barrier=tc.barrier, tta=tc.tta_rot, objects=int(n_obj),
                stitch_matches=int(n_match), seconds=round(time.time() - t0, 1),
                blocks=all_stats)
    json.dump(meta, open(os.path.splitext(args.out)[0] + ".stats.json", "w"), indent=1)
    print(f"done: {n_obj} objects, {n_match} stitch matches, "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
