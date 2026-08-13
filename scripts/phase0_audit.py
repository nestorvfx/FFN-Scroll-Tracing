#!/usr/bin/env python
"""Phase-0 instrument audits (research/12_next_moves.md Move 1) -- CPU/GPU, no training.

Sub-commands:
  lambda   -- wrap-period estimator A/B: tracer.rescale.measure_lambda vs ctstats.wrap_period on
              real Scroll-1 crops and synthetic cubes. The docstring of ctstats.wrap_period claims
              ~42 vox on Scroll-1 crops; research/10 A.1's validated estimator measures 16.69.
              Whichever is right, `core_score`/`core_valid_frac` are scaled by it.
  seeds    -- seed audit (Move 3's T0 gate): for every coord `inference_seeds` returns on the
              UNION mask, what fraction would `partitions.interior_seeds_gpu` also emit (i.e. is
              one-lamina-deep), reported AS A FUNCTION OF SEED RANK, plus the research/07
              statistic (>1 GT instance in a 5^3 box) so the number is directly comparable to the
              existing 2.3% refutation. Also reports the per-instance seed field for contrast.

Usage:
  python -m scripts.phase0_audit lambda --corpus DIR [--n 12]
  python -m scripts.phase0_audit seeds  --corpus DIR --work DIR [--n 8]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ffn.config import FFNConfig
from ffn import seeds as S
from ffn.ctstats import wrap_period
from ffn.partitions import interior_seeds_gpu


def _val_ids(work, corpus, kind, n):
    sp = os.path.join(work, "splits.json")
    if os.path.exists(sp):
        val = json.load(open(sp))["val"]
        if kind == "real":
            ids = [i for i in val if i.startswith("real_")]
        elif kind == "newsynth":
            ids = [i for i in val if not i.startswith("real_")
                   and i.rsplit("_", 1)[-1].isdigit() and int(i.rsplit("_", 1)[-1]) >= 39000]
        else:
            ids = [i for i in val if not i.startswith("real_")]
        return sorted(ids)[:n]
    pat = "real_*" if kind == "real" else "synthfuse_*"
    fs = sorted(glob.glob(os.path.join(corpus, "labelsTr_inst", pat + ".tif")))
    return [os.path.basename(f)[:-4] for f in fs[:n]]


def cmd_lambda(a):
    from tracer.rescale import measure_lambda
    print(f"{'cube':38s} {'measure_lambda':>14s} {'wrap_period':>12s} {'ratio':>7s}")
    rows = []
    for kind in ("real", "newsynth"):
        ids = _val_ids(a.work, a.corpus, kind, a.n)
        for cid in ids:
            ct = tifffile.imread(os.path.join(a.corpus, "imagesTr", f"{cid}_0000.tif"))
            lam = float(measure_lambda(ct))
            lac = float(wrap_period(ct))
            r = (lac / lam) if lam > 0 else float("nan")
            rows.append((kind, cid, lam, lac, r))
            print(f"{cid:38s} {lam:14.2f} {lac:12.2f} {r:7.2f}")
    for kind in ("real", "newsynth"):
        sub = [r for r in rows if r[0] == kind]
        if not sub:
            continue
        lam = np.array([r[2] for r in sub])
        lac = np.array([r[3] for r in sub])
        rat = np.array([r[4] for r in sub])
        print(f"\n{kind}: measure_lambda med {np.median(lam):.2f}  "
              f"wrap_period med {np.median(lac):.2f}  ratio med {np.median(rat):.2f} "
              f"(min {np.nanmin(rat):.2f} max {np.nanmax(rat):.2f})")
    print("\nIMPACT: _core_validity tests p90(2*EDT) <= core_thick_frac * lam. A ratio R means the "
          "thickness gate has been R times too permissive wherever wrap_period was used.")


def _box_multi_instance(inst, coords, box=5):
    """research/07 criterion: >1 distinct GT instance within a box^3 neighbourhood."""
    r = box // 2
    Z, Y, X = inst.shape
    out = np.zeros(len(coords), bool)
    for i, (z, y, x) in enumerate(coords):
        sub = inst[max(0, z - r):z + r + 1, max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
        u = np.unique(sub[sub > 0])
        out[i] = len(u) > 1
    return out


def cmd_seeds(a):
    import torch
    cfg = FFNConfig.load(os.path.join(a.work, "config.json")) if \
        os.path.exists(os.path.join(a.work, "config.json")) else FFNConfig()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"cfg: inf_seed_min_edt={cfg.inf_seed_min_edt} eval_seed_spacing={cfg.eval_seed_spacing} "
          f"eval_crop={cfg.eval_crop}")
    for kind in ("real", "newsynth"):
        ids = _val_ids(a.work, a.corpus, kind, a.n)
        agg = {"n": 0, "in_dist": [], "multi": [], "rank": [], "own_edt": [],
               "pi_in_dist": [], "pi_multi": []}
        for cid in ids:
            inst = tifffile.imread(os.path.join(a.corpus, "labelsTr_inst",
                                                f"{cid}.tif")).astype(np.int32)
            c = cfg.eval_crop
            o = [(s - c) // 2 for s in inst.shape]
            inst = inst[o[0]:o[0] + c, o[1]:o[1] + c, o[2]:o[2] + c]
            if (inst > 0).sum() < 1000:
                continue
            interior = interior_seeds_gpu(torch.from_numpy(inst).to(dev)).cpu().numpy()
            # --- the SHIPPED union-mask field ---
            fiber = (inst > 0).astype(np.float32)
            co, _ = S.inference_seeds(fiber, thr=0.5, min_edt=cfg.inf_seed_min_edt,
                                      spacing=cfg.eval_seed_spacing)
            if not len(co):
                continue
            ind = interior[co[:, 0], co[:, 1], co[:, 2]]
            multi = _box_multi_instance(inst, co)
            agg["in_dist"].append(ind)
            agg["multi"].append(multi)
            agg["rank"].append(np.arange(len(co)))
            agg["n"] += 1
            # --- the per-instance field, for contrast ---
            co2, _ = S.instance_inference_seeds(inst, min_edt=cfg.inf_seed_min_edt,
                                                spacing=cfg.eval_seed_spacing)
            if len(co2):
                agg["pi_in_dist"].append(interior[co2[:, 0], co2[:, 1], co2[:, 2]])
                agg["pi_multi"].append(_box_multi_instance(inst, co2))
        if not agg["n"]:
            print(f"\n{kind}: no usable cubes")
            continue
        ind = np.concatenate(agg["in_dist"])
        multi = np.concatenate(agg["multi"])
        rank = np.concatenate(agg["rank"])
        print(f"\n=== {kind} ({agg['n']} cubes, {len(ind)} seeds) ===")
        print(f"UNION-mask field (SHIPPED): in-distribution {100*ind.mean():.1f}%   "
              f"multi-instance-in-5^3 {100*multi.mean():.1f}%   "
              f"(research/07 measured 2.3-3.5% on Scroll-1)")
        for lo, hi in ((0, 50), (50, 200), (200, 1000), (1000, 10**9)):
            m = (rank >= lo) & (rank < hi)
            if m.sum():
                print(f"   rank {lo:5d}-{hi if hi < 10**9 else 'inf':>5}: "
                      f"in-dist {100*ind[m].mean():5.1f}%  multi {100*multi[m].mean():5.1f}%  "
                      f"(n={m.sum()})")
        if agg["pi_in_dist"]:
            pind = np.concatenate(agg["pi_in_dist"])
            pmul = np.concatenate(agg["pi_multi"])
            print(f"PER-INSTANCE field (proposed): in-distribution {100*pind.mean():.1f}%   "
                  f"multi-instance-in-5^3 {100*pmul.mean():.1f}%")
    print("\nGATE (Move 3): if the union field's in-distribution fraction is >= 80%, the "
          "instance-aware-seeding proposal is REFUTED and the budget moves elsewhere.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("lambda", "seeds"):
        s = sub.add_parser(name)
        s.add_argument("--corpus", required=True)
        s.add_argument("--work", default="")
        s.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    if a.cmd == "lambda":
        cmd_lambda(a)
    else:
        cmd_seeds(a)


if __name__ == "__main__":
    main()
