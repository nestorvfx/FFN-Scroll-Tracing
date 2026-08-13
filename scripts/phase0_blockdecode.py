#!/usr/bin/env python
"""Block decode + mid-halo mutual-max stitching, on the REAL slab with the trained network.

DESIGN item 19 was never implemented and its test never tiled (FINDINGS 3.4). This runs the real
comparison on the z=10192 slab (68 traced wraps):

  whole   -- segment_block over the whole tile (what the pipeline does today)
  blocked -- decode_blocked: per-block decode on core+halo, link on the mid-halo plane by
             MUTUAL-maximal overlap, reconcile with union-find, emit cores only

Reported on the same GT: adjacent-wrap merge rate, NERL, coverage, object count. The stitch rule is
the one measured at -84% mergers / +28% splits (Januszewski 2018); the shipped `stitch_blocks` is a
one-sided greedy majority rule and is dead code besides.

Also sweeps the FAFB fibre-mask restrictor, which can only BLOCK growth and therefore cannot add a
merge -- it is free coverage-neutral compute savings if it holds.

Usage:
  python -m scripts.phase0_blockdecode --slab DIR --work DIR --ckpt PATH [--tile 640]
"""
import argparse
import dataclasses
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ffn import metrics as M
from ffn import seeds as S
from ffn.blocks import decode_blocked
from ffn.config import FFNConfig
from ffn.ctstats import air_papyrus_threshold
from ffn.inference import segment_block
from scripts.phase0_decode import make_predict


def score(gt, pred, tag, secs, extra=""):
    r = M.adjacent_wrap_merge_rate(gt, pred, radius=3, theta=0.10)
    e = M.erl(gt, pred)
    cov = float(((gt > 0) & (pred > 0)).sum()) / max(int((gt > 0).sum()), 1)
    n = len(np.unique(pred[pred > 0]))
    print(f"{tag:>22s} {r['merge_rate']:7.4f} {r['n_pairs']:6d} {e['nerl']:7.4f} "
          f"{cov:7.4f} {n:6d} {secs:7.0f}s {extra}", flush=True)
    return dict(merge=r["merge_rate"], pairs=r["n_pairs"], nerl=e["nerl"], cov=cov, n=n)


def main():
    import nrrd
    ap = argparse.ArgumentParser()
    ap.add_argument("--slab", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--n-tiles", type=int, default=2)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--halo", type=int, default=0)
    ap.add_argument("--seed-spacing", type=int, default=8)
    ap.add_argument("--seed-cap", type=int, default=4000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--devices", default="")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    cfg0 = FFNConfig.load(os.path.join(a.work, "config.json"))
    cfg = dataclasses.replace(cfg0, fill_max_steps=cfg0.eval_fill_max_steps, agglo_enabled=False)
    halo = a.halo or (cfg.fov // 2 + cfg.delta + 8)
    pfs, devs, step = [], [], -1
    for d in ([a.device] if not a.devices else a.devices.split(",")):
        pf, step = make_predict(cfg0, a.ckpt, torch.device(d))
        pfs.append(pf)
        devs.append(torch.device(d))
    vol, _ = nrrd.read(os.path.join(a.slab, "volume.nrrd"))
    tru, _ = nrrd.read(os.path.join(a.slab, "truth.nrrd"))
    print(f"ckpt {step} | slab {vol.shape} | tile {a.tile} block {a.block} halo {halo} "
          f"| devices {len(pfs)}", flush=True)
    print(f"{'config':>22s} {'merge':>7s} {'pairs':>6s} {'nerl':>7s} {'cov':>7s} {'obj':>6s} "
          f"{'time':>8s}", flush=True)
    Z, Y, X = vol.shape
    T = a.tile
    done = 0
    agg = {}
    for y0 in range(0, Y - T + 1, T):
        for x0 in range(0, X - T + 1, T):
            if done >= a.n_tiles:
                break
            gt = np.ascontiguousarray(tru[:, y0:y0 + T, x0:x0 + T]).astype(np.int32)
            if (gt > 0).sum() < 50000:
                continue
            ct = np.ascontiguousarray(vol[:, y0:y0 + T, x0:x0 + T]).astype(np.uint8)
            img = torch.from_numpy(((ct.astype(np.float32) / 255.0 - 0.5) / 0.5)).to(dev)
            fib = ct >= air_papyrus_threshold(ct)
            sd, _ = S.inference_seeds(fib.astype(np.float32), thr=0.5,
                                      min_edt=cfg.inf_seed_min_edt, spacing=a.seed_spacing)
            sd = sd[:a.seed_cap]
            print(f"--- tile y{y0} x{x0}: {len(sd)} seeds, {len([i for i in np.unique(gt) if i])} "
                  f"GT wraps ---", flush=True)
            t0 = time.time()
            with torch.no_grad():
                whole = segment_block(pfs[0], img, sd, cfg, K=128)
            r1 = score(gt, whole, "whole", time.time() - t0)
            t0 = time.time()
            blocked, st = decode_blocked(pfs, img, sd, cfg, block=(Z, a.block, a.block),
                                         halo=halo, K=128, devices=devs)
            r2 = score(gt, blocked, "blocked+mutualmax", time.time() - t0,
                       f"[{st['n_blocks']} blocks, {st['n_links']} links]")
            t0 = time.time()
            blocked_m, st2 = decode_blocked(pfs, img, sd, cfg, block=(Z, a.block, a.block),
                                            halo=halo, K=128, fibre_mask=fib, devices=devs)
            r3 = score(gt, blocked_m, "blocked+fibremask", time.time() - t0,
                       f"[{st2['n_links']} links]")
            for k, r in (("whole", r1), ("blocked", r2), ("blocked_fib", r3)):
                a_ = agg.setdefault(k, dict(merge_n=0, pairs=0, nerl=[], cov=[]))
                a_["merge_n"] += r["merge"] * r["pairs"]
                a_["pairs"] += r["pairs"]
                a_["nerl"].append(r["nerl"])
                a_["cov"].append(r["cov"])
            done += 1
        if done >= a.n_tiles:
            break
    print(f"\n=== AGGREGATE over {done} tiles ===")
    for k, v in agg.items():
        print(f"  {k:14s} merge {v['merge_n']/max(v['pairs'],1):.4f} ({v['pairs']} pairs)  "
              f"nerl {np.mean(v['nerl']):.4f}  cov {np.mean(v['cov']):.4f}")
    print("\nGATE: blocked must not raise merge vs whole (the stitch is supposed to REDUCE it), "
          "and must retain NERL. The fibre-mask arm can only block growth, so any merge increase "
          "there is a bug, not a trade.")


if __name__ == "__main__":
    main()
