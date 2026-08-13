#!/usr/bin/env python
"""Phase-1 decode experiments (research/12_next_moves.md) -- GPU, no training.

  kdist   -- THE decisive experiment. Re-decode the held-out panel with POM level retention on,
             then for every GT-adjacent pair the tau=0.5 decode MERGED, find the minimum level k
             at which it separates into two 26-connected components each matching one GT wrap.
             A finite k for most merges => merge control moves off the commit threshold and into
             a free post-hoc hierarchy. No finite k => the merges are genuinely fused and every
             threshold lever is capped.
             ALSO reports the three-bucket decomposition of UNCLAIMED foreground:
               (a) below tau anywhere      -> representation
               (b) above tau but dropped   -> policy (min-size floor / claimed veto / seed-CC)
               (c) never visited by a fill -> seeding / budget
  sweep   -- ratchet freeze-threshold sweep (cfg.pom_ratchet_freeze_p) over the panel.

Usage:
  python -m scripts.phase0_decode kdist --work DIR --ckpt PATH [--n-real 8 --n-synth 4]
  python -m scripts.phase0_decode sweep --work DIR --ckpt PATH --freeze 0.5,0.3,0.2,0.1
"""
import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np
import tifffile
import torch
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ffn.config import FFNConfig
from ffn.model import build_model
from ffn import inference as I
from ffn import seeds as S
from ffn.inline_eval import fair_seeds
from ffn.metrics import adjacency_pairs, erl as erl_metric


def panel_ids(work, n_real, n_synth):
    splits = json.load(open(os.path.join(work, "splits.json")))
    val = splits["val"]
    real = sorted(i for i in val if i.startswith("real_"))[:n_real]

    def sid(c):
        t = c.rsplit("_", 1)[-1]
        return int(t) if t.isdigit() else -1
    new_g = [i for i in val if not i.startswith("real_") and sid(i) >= 39000]
    old_g = [i for i in val if not i.startswith("real_") and 0 <= sid(i) < 39000]
    syn = (sorted(new_g)[:max(1, n_synth // 2)] + sorted(old_g)[:n_synth - max(1, n_synth // 2)])
    return real + syn


def _prep_one(args):
    """Worker: crop one cube and compute its (fixed, model-independent) seed set.

    The per-instance seed field is ~25 distance transforms per cube -- embarrassingly parallel
    across cubes and previously run single-threaded, which idled 47 of 48 cores while both GPUs
    sat empty."""
    cid, corpus, crop, inst_seeds, min_edt, spacing, cap, per_inst = args
    img = tifffile.imread(os.path.join(corpus, "imagesTr", f"{cid}_0000.tif"))
    inst = tifffile.imread(os.path.join(corpus, "labelsTr_inst", f"{cid}.tif")).astype(np.int32)
    o = [(s - crop) // 2 for s in inst.shape]
    img = img[o[0]:o[0] + crop, o[1]:o[1] + crop, o[2]:o[2] + crop]
    inst = inst[o[0]:o[0] + crop, o[1]:o[1] + crop, o[2]:o[2] + crop]
    if inst_seeds:
        co, _ = S.instance_inference_seeds(inst, min_edt=min_edt, spacing=spacing)
    else:
        co, _ = S.inference_seeds((inst > 0).astype(np.float32), thr=0.5,
                                  min_edt=min_edt, spacing=spacing)
    co = fair_seeds(co, inst, cap, per_inst)
    return dict(cid=cid, is_real=cid.startswith("real_"), img=img, inst=inst, seeds=co)


def load_panel(cfg, work, n_real, n_synth, cache="", jobs=0):
    """Build (or load) the fixed eval panel. Seeds depend only on (labels, cfg), never on the
    model, so the whole panel is cached to `cache` and reused by every experiment."""
    ids = panel_ids(work, n_real, n_synth)
    key = f"{'inst' if getattr(cfg, 'eval_instance_seeds', False) else 'union'}_" \
          f"{cfg.eval_crop}_{cfg.eval_seed_spacing}_{cfg.eval_seed_cap}"
    if cache and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        if str(d["key"]) == key and list(d["ids"]) == ids:
            print(f"[panel] loaded {len(ids)} cubes from cache {cache}", flush=True)
            return list(d["panel"])
    import multiprocessing as mp
    args = [(cid, cfg.corpus_dir, cfg.eval_crop,
             bool(getattr(cfg, "eval_instance_seeds", False)), cfg.inf_seed_min_edt,
             cfg.eval_seed_spacing, cfg.eval_seed_cap,
             int(getattr(cfg, "eval_seed_per_inst", 0))) for cid in ids]
    n = jobs or min(len(args), max(1, (os.cpu_count() or 8) - 2))
    t0 = time.time()
    with mp.Pool(n) as pool:
        out = pool.map(_prep_one, args)
    print(f"[panel] built {len(out)} cubes on {n} workers in {time.time()-t0:.0f}s", flush=True)
    if cache:
        np.savez(cache, key=key, ids=np.array(ids, object), panel=np.array(out, object))
    return out


def make_predict(cfg, ckpt, dev):
    m = build_model(cfg).to(dev)
    if cfg.channels_last:
        m = m.to(memory_format=torch.channels_last_3d)
    ck = torch.load(ckpt, map_location=dev)
    m.load_state_dict(ck["model"])
    m.eval()

    def pf(inp):
        if cfg.channels_last:
            inp = inp.to(memory_format=torch.channels_last_3d)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return m(inp).float()
    return pf, int(ck.get("step", -1))


def cmd_kdist(a):
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    cfg = FFNConfig.load(os.path.join(a.work, "config.json"))
    ecfg = dataclasses.replace(cfg, fill_max_steps=cfg.eval_fill_max_steps, agglo_enabled=False)
    pf, step = make_predict(cfg, a.ckpt, dev)
    panel = load_panel(cfg, a.work, a.n_real, a.n_synth, a.cache, a.jobs)
    print(f"ckpt step {step}, panel {len(panel)} cubes, tau={cfg.commit_threshold}", flush=True)
    all_k, unresolved = [], 0
    B = dict(repr=0, policy=0, seed=0, total_unclaimed=0)
    for cb in panel:
        img_gpu = torch.from_numpy(((cb["img"].astype(np.float32) / 255.0 - 0.5) / 0.5)).to(dev)
        levels = np.zeros(cb["inst"].shape, np.uint8)
        with torch.no_grad():
            pred = I.segment_block(pf, img_gpu, cb["seeds"], ecfg, K=128, levels_out=levels)
        gt = cb["inst"]
        pairs = adjacency_pairs(gt, cfg.adjacency_radius)
        theta = cfg.merge_theta
        sizes = {int(i): int((gt == i).sum()) for i in np.unique(gt) if i}
        merged = []
        for (i, j) in pairs:
            for L in np.unique(pred[(gt == i) & (pred > 0)]):
                if L == 0:
                    continue
                fi = (pred[gt == i] == L).sum() / max(sizes[i], 1)
                fj = (pred[gt == j] == L).sum() / max(sizes[j], 1)
                if fi >= theta and fj >= theta:
                    merged.append((i, j, int(L)))
                    break
        # minimum separating level k
        for (i, j, L) in merged:
            obj = pred == L
            found = None
            for k in range(2, 17):
                sub = obj & (levels >= k)
                if not sub.any():
                    break
                lab, n = ndi.label(sub, structure=np.ones((3, 3, 3), bool))
                if n < 2:
                    continue
                ci = {int(v) for v in np.unique(lab[(gt == i) & sub]) if v}
                cj = {int(v) for v in np.unique(lab[(gt == j) & sub]) if v}
                if ci and cj and not (ci & cj):
                    found = k
                    break
            if found is None:
                unresolved += 1
            else:
                all_k.append(found)
        # three-bucket decomposition of unclaimed foreground
        fg = gt > 0
        unclaimed = fg & (pred == 0)
        B["total_unclaimed"] += int(unclaimed.sum())
        # (c) never visited: no seed's fill ever wrote a level there AND no committed voxel near
        near = ndi.binary_dilation(pred > 0, np.ones((3, 3, 3), bool), 3)
        B["seed"] += int((unclaimed & ~near).sum())
        rest = unclaimed & near
        B["policy"] += int((rest & (levels > 0)).sum())
        B["repr"] += int((rest & (levels == 0)).sum())
        print(f"  {cb['cid']:38s} merged_pairs={len(merged):3d} "
              f"unclaimed={100*unclaimed.sum()/max(fg.sum(),1):5.1f}%")
    n_m = len(all_k) + unresolved
    print(f"\n=== K-DISTRIBUTION over {n_m} merged GT-adjacent pairs ===")
    if all_k:
        ak = np.array(all_k)
        print(f"  separable at a finite level: {len(ak)}/{n_m} ({100*len(ak)/max(n_m,1):.0f}%)")
        for q in (25, 50, 75, 90):
            print(f"    k p{q} = {np.percentile(ak, q):.0f}  "
                  f"(p = {cfg.commit_threshold + (0.99-cfg.commit_threshold)*np.percentile(ak,q)/16:.3f})")
    print(f"  NEVER separable up to level 16: {unresolved}/{n_m} "
          f"({100*unresolved/max(n_m,1):.0f}%) -- genuinely fused")
    t = max(B["total_unclaimed"], 1)
    print(f"\n=== UNCLAIMED FOREGROUND DECOMPOSITION ({t:,} voxels) ===")
    print(f"  representation (below tau, near a fill) : {100*B['repr']/t:5.1f}%")
    print(f"  policy (above tau but dropped)          : {100*B['policy']/t:5.1f}%")
    print(f"  seeding/budget (never reached)          : {100*B['seed']/t:5.1f}%")
    print("\nVERDICT: finite-k majority => build the assembly stage / ship the hierarchy. "
          "Mostly unresolved => threshold levers capped, go to seeding+training.")


def cmd_sweep(a):
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    cfg0 = FFNConfig.load(os.path.join(a.work, "config.json"))
    pf, step = make_predict(cfg0, a.ckpt, dev)
    panel = load_panel(cfg0, a.work, a.n_real, a.n_synth, a.cache, a.jobs)
    print(f"ckpt step {step}, panel {len(panel)} cubes ({sum(c['is_real'] for c in panel)} real)", flush=True)
    print(f"{'freeze_p':>9s} {'real_nerl':>9s} {'real_merge':>10s} {'synth_nerl':>10s} "
          f"{'coverage':>9s} {'inst':>5s}", flush=True)
    for fp in [float(x) for x in a.freeze.split(",")]:
        cfg = dataclasses.replace(cfg0, fill_max_steps=cfg0.eval_fill_max_steps,
                                  agglo_enabled=False, pom_ratchet_freeze_p=fp)
        rr = rp = sr = sp = 0.0
        nm = npair = 0
        cov_n = cov_d = 0
        ninst = 0
        for cb in panel:
            img_gpu = torch.from_numpy(
                ((cb["img"].astype(np.float32) / 255.0 - 0.5) / 0.5)).to(dev)
            with torch.no_grad():
                pred = I.segment_block(pf, img_gpu, cb["seeds"], cfg, K=128)
            gt = cb["inst"]
            e = erl_metric(gt, pred)
            if cb["is_real"]:
                rr += e["sum_runsq"]; rp += e["sum_perfsq"]
            else:
                sr += e["sum_runsq"]; sp += e["sum_perfsq"]
            pairs = adjacency_pairs(gt, cfg.adjacency_radius)
            sizes = {int(i): int((gt == i).sum()) for i in np.unique(gt) if i}
            for (i, j) in pairs:
                npair += 1
                hit = False
                for L in np.unique(pred[(gt == i) & (pred > 0)]):
                    if L and (pred[gt == i] == L).sum() / max(sizes[i], 1) >= cfg.merge_theta \
                       and (pred[gt == j] == L).sum() / max(sizes[j], 1) >= cfg.merge_theta:
                        hit = True
                        break
                nm += int(hit)
            cov_n += int(((gt > 0) & (pred > 0)).sum()); cov_d += int((gt > 0).sum())
            ninst += len(np.unique(pred[pred > 0]))
        print(f"{fp:9.2f} {rr/max(rp,1e-9):9.4f} {nm/max(npair,1):10.4f} "
              f"{sr/max(sp,1e-9):10.4f} {cov_n/max(cov_d,1):9.4f} {ninst:5d}", flush=True)
    print("\nSHIP the value that holds the merge reduction at the highest coverage. "
          "Respect the instrument: NERL sigma 0.029-0.057, merge quantum 1/n_pairs.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("kdist", "sweep"):
        s = sub.add_parser(name)
        s.add_argument("--work", required=True)
        s.add_argument("--ckpt", required=True)
        s.add_argument("--n-real", type=int, default=8)
        s.add_argument("--n-synth", type=int, default=4)
        s.add_argument("--cache", default="/root/panel_cache.npz")
        s.add_argument("--jobs", type=int, default=0)
        s.add_argument("--device", default="cuda:0")
        if name == "sweep":
            s.add_argument("--freeze", default="0.5,0.3,0.2,0.1")
    a = ap.parse_args()
    (cmd_kdist if a.cmd == "kdist" else cmd_sweep)(a)


if __name__ == "__main__":
    main()
