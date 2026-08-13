#!/usr/bin/env python
"""Capacity adjudication (research/12_next_moves.md item 4): can a BIGGER net fit what we ask?

Overfit a handful of cubes with augmentation and weight decay OFF, at depth 8 (shipped) vs a
larger arm, and compare the excess balanced-BCE over the exact soft-target floor:

    floor = -(tgt_hi*ln(tgt_hi) + tgt_lo*ln(tgt_lo)) evaluated per-half = 0.19852 at 0.95/0.05

If both arms plateau at the same excess, the residual is Bayes error (ambiguous targets at blind
contacts) and NO architecture change fixes it -- the user's underfitting hypothesis is refuted on
its own terms. If the bigger arm reaches a materially lower excess, capacity binds and FFN v1.5
depth is justified.

This is a MEMORISATION probe: overfitting is the POINT (no aug, no wd, few cubes). Val numbers
are meaningless here and are not reported.

Usage:
  python -m scripts.phase0_memprobe --work DIR --steps 3000 --cubes 8 --depths 8,20
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import tifffile
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ffn.config import FFNConfig
from ffn.model import build_model
from ffn.train_step import TrainStepper
from ffn.volumes import GpuVolumeCache
from ffn.curriculum import Curriculum
from ffn.partitions import interior_seeds_gpu


def floor_bce(tgt_hi, tgt_lo):
    """Irreducible balanced BCE when the model predicts the soft targets exactly."""
    def h(p):
        return -(p * math.log(p) + (1 - p) * math.log(1 - p))
    return 0.5 * (h(tgt_hi) + h(tgt_lo))


def build_index(cfg, ids, device, max_per_cube=4000):
    """Minimal candidate index over a few cubes: interior (single-instance-3^3) voxels
    respecting the training-patch margin, mirroring data_prep's contract."""
    cols = {k: [] for k in ("vol", "z", "y", "x", "inst", "pbin", "hard", "is_std", "is_real")}
    imgs, insts = [], []
    m = cfg.margin
    for vi, cid in enumerate(ids):
        img = tifffile.imread(os.path.join(cfg.corpus_dir, "imagesTr", f"{cid}_0000.tif"))
        inst = tifffile.imread(os.path.join(cfg.corpus_dir, "labelsTr_inst",
                                            f"{cid}.tif")).astype(np.int32)
        imgs.append(torch.from_numpy(img.astype(np.uint8)))
        insts.append(torch.from_numpy(inst.astype(np.int16)))
        it = interior_seeds_gpu(torch.from_numpy(inst).to(device)).cpu().numpy()
        Z, Y, X = inst.shape
        it[:m] = it[-m:] = False
        it[:, :m] = it[:, -m:] = False
        it[:, :, :m] = it[:, :, -m:] = False
        co = np.argwhere(it)
        if len(co) > max_per_cube:
            co = co[np.random.default_rng(vi).choice(len(co), max_per_cube, replace=False)]
        n = len(co)
        cols["vol"].append(np.full(n, vi, np.int32))
        cols["z"].append(co[:, 0]); cols["y"].append(co[:, 1]); cols["x"].append(co[:, 2])
        cols["inst"].append(inst[co[:, 0], co[:, 1], co[:, 2]])
        cols["pbin"].append(np.full(n, 8, np.int32))       # mid fill-fraction bin
        cols["hard"].append(np.zeros(n, bool))
        cols["is_std"].append(np.ones(n, bool))
        cols["is_real"].append(np.full(n, cid.startswith("real_")))
    index = {k: np.concatenate(v) for k, v in cols.items()}
    return torch.stack(imgs).to(device), torch.stack(insts).to(device), index


def run_arm(cfg, depth, image_stack, inst_stack, index, device, steps, log_every):
    cfg = __import__("dataclasses").replace(cfg, depth=depth, augment=False, weight_decay=0.0,
                                            compile=False)
    torch.manual_seed(1234)
    model = build_model(cfg).to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last_3d)
    nparam = sum(p.numel() for p in model.parameters())
    cache = GpuVolumeCache(image_stack, inst_stack, index, cfg, device)
    stepper = TrainStepper(cfg, device)
    curr = Curriculum(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    g = torch.Generator(device=device); g.manual_seed(7)
    hist = []
    t0 = time.time()
    for step in range(steps):
        st = curr.state(cfg.max_steps - 1)              # full curriculum (T=max_unroll)
        opt.zero_grad(set_to_none=True)
        out = stepper.run(model, None, cache, st, opt, g)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        hist.append(out["loss"])
        if (step + 1) % log_every == 0:
            w = float(np.mean(hist[-log_every:]))
            print(f"    depth{depth} step {step+1:5d}  loss {w:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return dict(depth=depth, params=nparam, final=float(np.mean(hist[-log_every:])),
                best=float(min(np.convolve(hist, np.ones(log_every) / log_every, "valid"))),
                hist=hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--cubes", type=int, default=8)
    ap.add_argument("--depths", default="8,20")
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    cfg = FFNConfig.load(os.path.join(a.work, "config.json"))
    splits = json.load(open(os.path.join(a.work, "splits.json")))
    tr = splits["train"]
    syn = [i for i in tr if not i.startswith("real_")
           and i.rsplit("_", 1)[-1].isdigit() and int(i.rsplit("_", 1)[-1]) >= 39000]
    real = [i for i in tr if i.startswith("real_")]
    ids = sorted(syn)[:max(1, a.cubes - 2)] + sorted(real)[:2]
    print(f"memorisation probe: {len(ids)} cubes ({len(ids)-2} new-geo synth + 2 real), "
          f"{a.steps} steps, aug OFF, wd 0, T={cfg.max_unroll}")
    fl = floor_bce(cfg.tgt_hi, cfg.tgt_lo)
    print(f"exact soft-target floor = {fl:.5f}")
    image_stack, inst_stack, index = build_index(cfg, ids, device)
    print(f"index: {index['vol'].size:,} candidates over {len(ids)} cubes")
    res = []
    for d in [int(x) for x in a.depths.split(",")]:
        print(f"\n--- arm depth={d} ---", flush=True)
        r = run_arm(cfg, d, image_stack, inst_stack, index, device, a.steps, a.log_every)
        r["excess"] = r["final"] - fl
        r["excess_best"] = r["best"] - fl
        res.append(r)
        print(f"  depth {d}: params {r['params']:,}  final {r['final']:.4f}  "
              f"EXCESS over floor {r['excess']:.4f}")
    print("\n=== CAPACITY VERDICT ===")
    for r in res:
        print(f"  depth {r['depth']:2d} ({r['params']:>9,} params): final {r['final']:.4f}  "
              f"excess {r['excess']:.4f}  (best-window excess {r['excess_best']:.4f})")
    if len(res) >= 2:
        d = res[0]["excess_best"] - res[-1]["excess_best"]
        rel = d / max(res[0]["excess_best"], 1e-9)
        print(f"\n  depth {res[0]['depth']} -> {res[-1]['depth']}: excess reduced by {d:.4f} "
              f"({100*rel:.1f}%)")
        print("  INTERPRETATION: <10% => capacity does NOT bind, the residual is target "
              "ambiguity (underfitting hypothesis refuted). >25% => capacity binds, FFN v1.5 "
              "depth is justified.")
    if a.out:
        json.dump([{k: v for k, v in r.items() if k != "hist"} for r in res],
                  open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
