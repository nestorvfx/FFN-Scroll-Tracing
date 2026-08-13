"""Stitch equivalence: does blocked decode + mutual-max stitching reproduce the
monolithic decode on the same volume?

One official 320^3 cube, production settings, barrier ON:
  MONO   single 320^3 canvas (the reference -- what a big-enough GPU would do)
  TILED  2x2x2 blocks (core 160, halo 48 -> 256^3 canvases) -> assemble()

Compared on: object count, merges/NERL vs GT, and cross-face identity -- the
fraction of GT wraps whose voxels land in ONE stitched object vs how many the
mono decode gives. Stitching cannot be BETTER than mono (it sees strictly less
context per fill); the question is how little it loses and that it adds no merges.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/root/ds_probe/pylibs")

import dataclasses           # noqa: E402
import os                    # noqa: E402

import numpy as np           # noqa: E402
import tifffile              # noqa: E402
import torch                 # noqa: E402

sys.path.insert(0, "/root/surface_detection/FFN")
sys.path.insert(0, "/root")
from scripts.evaluate import load_model, make_predict_fn   # noqa: E402
from inference.v1_separation import adjacent_pairs         # noqa: E402

from tracer.blocks import tile            # noqa: E402
from tracer.config import TracerConfig    # noqa: E402
from tracer.decode import decode_block    # noqa: E402
from tracer.normalize import harmonize    # noqa: E402
from tracer.stitch import assemble        # noqa: E402
from tracer.test_gt import score, thread_to_body   # noqa: E402

CUBE = "/root/ds_probe/kaggle/images/sample_00050.tif"


def main():
    tc = TracerConfig().load_anchors()
    dev = torch.device(tc.device)
    model, cfg = load_model(tc.ckpt, dev)
    cfg = dataclasses.replace(cfg, fill_max_steps=tc.fill_max_steps,
                              consensus_scales=[1])
    pf = make_predict_fn(model, cfg, dev)

    ct_raw = tifffile.imread(CUBE)
    inst, band = thread_to_body(ct_raw, tifffile.imread(CUBE.replace("images", "labels")))
    pairs = adjacent_pairs(inst)
    ct = harmonize(ct_raw, tc.slab_bg, tc.slab_fg95) if np.isfinite(tc.slab_bg) else ct_raw
    tc.barrier = True

    lab_mono, st = decode_block(pf, ct, cfg, tc)
    lab_mono[~band] = 0
    r = score(inst, lab_mono, pairs)
    print(f"MONO   objects={st['objects']:3d} NERL={r['nerl']:.3f} "
          f"merges={r['merges']}/{r['pairs']} frags={r['frags']}", flush=True)

    blocks = tile(ct.shape, 160, tc.halo)
    labs = []
    for i, b in enumerate(blocks):
        cz0, cy0, cx0, cz1, cy1, cx1 = b.canvas
        lab, stb = decode_block(pf, ct[cz0:cz1, cy0:cy1, cx0:cx1], cfg, tc)
        labs.append(lab)
        print(f"  block {i + 1}/{len(blocks)} objects={stb['objects']}", flush=True)
    glob, n_obj, n_match = assemble(ct.shape, blocks, labs,
                                    min_overlap=tc.stitch_min_overlap)
    glob[~band] = 0
    r2 = score(inst, glob, pairs)
    print(f"TILED  objects={n_obj:3d} NERL={r2['nerl']:.3f} "
          f"merges={r2['merges']}/{r2['pairs']} frags={r2['frags']} "
          f"stitch_matches={n_match}", flush=True)
    print("\nACCEPT iff TILED merges <= MONO merges and NERL within ~15% of MONO.")


if __name__ == "__main__":
    main()
