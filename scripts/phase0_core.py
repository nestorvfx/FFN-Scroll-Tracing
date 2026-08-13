#!/usr/bin/env python
"""core_score re-baseline with the CORRECTED wrap period (research/12_next_moves.md Move 1.2).

`inline_eval._core_validity` scores an object VALID iff p90(2*EDT) <= core_thick_frac * lambda,
i.e. "it did not fuse across a wrap". Lambda came from `ctstats.wrap_period`, measured here at
5.79x (Scroll-3) / 2.67x (Scroll-4) too large -- so the gate fired at ~36 voxels where the physics
says ~6.5, and `core_score` (a checkpoint-selection column) ranked against a rubber ruler.

This decodes the unlabelled core cubes with a checkpoint and reports the validity statistics under
BOTH lambdas, so the size of the correction is explicit rather than asserted. Parallel seed prep
across cores; decode on the requested GPU.

Usage:
  python -m scripts.phase0_core --work DIR --ckpt PATH --dirs D1,D2 [--per-dir 6 --device cuda:0]
"""
import argparse
import dataclasses
import glob
import os
import sys
import time

import numpy as np
import tifffile
import torch
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ffn.config import FFNConfig
from ffn.model import build_model
from ffn import inference as I
from ffn import seeds as S
from ffn.ctstats import air_papyrus_threshold, wrap_period


def _prep(args):
    path, min_edt, spacing, cap, crop = args
    from tracer.rescale import measure_lambda
    ct = tifffile.imread(path)
    if crop and ct.shape[0] > crop:
        o = [(s - crop) // 2 for s in ct.shape]
        ct = ct[o[0]:o[0] + crop, o[1]:o[1] + crop, o[2]:o[2] + crop]
    fiber = (ct >= air_papyrus_threshold(ct)).astype(np.float32)
    co, _ = S.inference_seeds(fiber, thr=0.5, min_edt=min_edt, spacing=spacing)
    return dict(cid=os.path.basename(path)[:-9], ct=ct, seeds=co[:cap],
                fibre=int(fiber.sum()), lam_fix=float(measure_lambda(ct)),
                lam_old=float(wrap_period(ct)))


def validity(pred, lam, thick_frac, topk):
    """Port of inline_eval._core_validity: (sum size^2 over VALID objects, n_obj, n_valid)."""
    ids, cnts = np.unique(pred[pred > 0], return_counts=True)
    if not len(ids) or lam <= 0:
        return None, len(ids), 0
    thick_max = thick_frac * lam
    order = np.argsort(-cnts)[:topk]
    objs = ndi.find_objects(pred)
    vsq, nval = 0.0, 0
    for k in order:
        iid = int(ids[k])
        sl = objs[iid - 1]
        if sl is None:
            continue
        m = pred[sl] == iid
        lab_m, nc = ndi.label(m, structure=np.ones((3, 3, 3), bool))
        if nc > 1:
            cc = np.bincount(lab_m.ravel())[1:]
            if cc.max() < 0.9 * cc.sum():
                continue
        edt = ndi.distance_transform_edt(m)
        if float(np.percentile(edt[m], 90)) * 2.0 > thick_max:
            continue
        vsq += float(cnts[k]) ** 2
        nval += 1
    return vsq, len(ids), nval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--per-dir", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--jobs", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    cfg = FFNConfig.load(os.path.join(a.work, "config.json"))
    ecfg = dataclasses.replace(cfg, fill_max_steps=cfg.eval_fill_max_steps, agglo_enabled=False)
    m = build_model(cfg).to(dev)
    if cfg.channels_last:
        m = m.to(memory_format=torch.channels_last_3d)
    ck = torch.load(a.ckpt, map_location=dev)
    m.load_state_dict(ck["model"])
    m.eval()

    def pf(inp):
        if cfg.channels_last:
            inp = inp.to(memory_format=torch.channels_last_3d)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return m(inp).float()

    import multiprocessing as mp
    for d in a.dirs.split(","):
        fs = sorted(glob.glob(os.path.join(d, "imagesTr", "*_0000.tif")))
        if not fs:
            print(f"{d}: no cubes")
            continue
        fs = fs[::max(1, len(fs) // (a.per_dir + 1))][:a.per_dir]
        args = [(f, cfg.inf_seed_min_edt, cfg.eval_seed_spacing, cfg.eval_core_seed_cap,
                 cfg.eval_crop) for f in fs]
        n = a.jobs or min(len(args), max(1, (os.cpu_count() or 8) - 2))
        t0 = time.time()
        with mp.Pool(n) as pool:
            cubes = pool.map(_prep, args)
        print(f"\n=== {os.path.basename(d.rstrip('/'))}: {len(cubes)} cubes "
              f"(prep {time.time()-t0:.0f}s on {n} workers) ===", flush=True)
        tot = dict(fix_sq=0.0, old_sq=0.0, fib_sq=0.0, nobj=0, nval_fix=0, nval_old=0,
                   claimed=0, fibre=0, excluded_old=0, excluded_fix=0)
        for cb in cubes:
            g = torch.from_numpy(((cb["ct"].astype(np.float32) / 255.0 - 0.5) / 0.5)).to(dev)
            with torch.no_grad():
                pred = I.segment_block(pf, g, cb["seeds"], ecfg, K=128)
            vf, nobj, nvf = validity(pred, cb["lam_fix"], cfg.core_thick_frac,
                                     cfg.eval_core_topk)
            vo, _, nvo = validity(pred, cb["lam_old"], cfg.core_thick_frac, cfg.eval_core_topk)
            claimed = int((pred > 0).sum())
            tot["claimed"] += claimed; tot["fibre"] += cb["fibre"]; tot["nobj"] += nobj
            tot["fib_sq"] += float(cb["fibre"]) ** 2
            if vf is None:
                tot["excluded_fix"] += 1
            else:
                tot["fix_sq"] += vf; tot["nval_fix"] += nvf
            if vo is None:
                tot["excluded_old"] += 1
            else:
                tot["old_sq"] += vo; tot["nval_old"] += nvo
            print(f"  {cb['cid'][:36]:36s} lam fix {cb['lam_fix']:5.2f} old {cb['lam_old']:6.2f} "
                  f"| obj {nobj:3d} valid fix {nvf:3d} old {nvo:3d}", flush=True)
        vfrac_fix = tot["nval_fix"] / max(tot["nobj"], 1)
        vfrac_old = tot["nval_old"] / max(tot["nobj"], 1)
        print(f"  ---- core_valid_frac  FIXED {vfrac_fix:.3f}   OLD(shipped) {vfrac_old:.3f}")
        print(f"  ---- core_score       FIXED {tot['fix_sq']/max(tot['fib_sq'],1):.5f}   "
              f"OLD(shipped) {tot['old_sq']/max(tot['fib_sq'],1):.5f}")
        print(f"  ---- core_cov {tot['claimed']/max(tot['fibre'],1):.4f}  "
              f"cubes excluded (lam unmeasurable): fixed {tot['excluded_fix']} "
              f"old {tot['excluded_old']}")
    print("\nA LOWER fixed core_valid_frac is the CORRECT reading: the old gate accepted objects "
          "up to ~6x thicker than one wrap. Historical core_score values are not comparable.")


if __name__ == "__main__":
    main()
