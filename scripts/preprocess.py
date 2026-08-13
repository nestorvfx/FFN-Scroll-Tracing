#!/usr/bin/env python
"""Preprocess the synth corpus into an FFN training index (DESIGN.md 5, M0-M1).

Produces, under cfg.work_dir:
  splits.json      - train / synth-val cube ids (stratified 12 deep / 48 std)
  train_index.npz  - flat candidate-center arrays (vol,z,y,x,inst,pbin,hard,is_std)
  manifest.json    - per-cube shape, n_instances, stratum, fg

The candidate `vol` index refers to the position of the cube in splits["train"],
so train.py must load the train cubes in that exact order.

Usage:
  python -m scripts.preprocess --work /root/surf/ffn_work \
      --corpus /root/surf/data/synthfuse_corpus --workers 32 [--limit N]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffn.config import FFNConfig
from ffn.data_prep import process_cube


def make_split(meta: dict, cfg: FFNConfig, limit=None):
    """Stratified split. Synth cubes: deep/std-balanced synth_val_n holdout (as before).
    Real cubes (source=='real', AUDIT FIX 1): hold out real_val_n whole PARENT cubes (all
    their crops) for the real val gate -- never split a parent across train/val. Returns
    (train, val) id lists; real ids carry source in `meta`."""
    rng = np.random.default_rng(cfg.seed)
    synth = sorted(i for i in meta if meta[i].get("source", "synth") != "real")
    real = sorted(i for i in meta if meta[i].get("source", "synth") == "real")
    if limit:
        synth = synth[:limit]
    # ---- synth val (deep/std balanced) ----
    deep = [i for i in synth if meta[i]["stratum"] == "deep"]
    std = [i for i in synth if meta[i]["stratum"] == "std"]
    n_deep = min(len(deep), max(1, round(cfg.synth_val_n * 80 / 400)))
    n_std = min(len(std), cfg.synth_val_n - n_deep)
    val_deep = set(rng.choice(deep, size=n_deep, replace=False).tolist()) if deep else set()
    val_std = set(rng.choice(std, size=n_std, replace=False).tolist()) if std else set()
    val_synth = val_deep | val_std
    # ---- real val (whole-parent holdout) ----
    parents = sorted(set(meta[i]["parent"] for i in real))
    n_valp = min(len(parents), cfg.real_val_n)
    val_parents = set(rng.choice(parents, size=n_valp, replace=False).tolist()) if parents else set()
    val_real = {i for i in real if meta[i]["parent"] in val_parents}
    val = sorted(val_synth | val_real)
    train = [i for i in synth if i not in val_synth] + [i for i in real if i not in val_real]
    if real:
        print(f"[split] real: {len(real)} crops / {len(parents)} cubes; "
              f"val={len(val_parents)} cubes ({len(val_real)} crops), train={len(real)-len(val_real)} crops")
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 2),
                    help="(unused; kept for CLI compatibility -- prep is single-GPU)")
    ap.add_argument("--max-cands", dest="max_cands", type=int, default=20000,
                    help="cap candidate centers per cube")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val", type=int, default=None, help="override synth_val_n (for subset runs)")
    ap.add_argument("--real-meta", dest="real_meta", default=None,
                    help="path to real_meta.json (default: <corpus>/real_meta.json if present)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = FFNConfig.load(args.config) if args.config else FFNConfig()
    if args.val is not None:
        cfg.synth_val_n = args.val
    if args.corpus:
        cfg.corpus_dir = args.corpus
    if args.work:
        cfg.work_dir = args.work
    os.makedirs(cfg.work_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.work_dir, "config.json"))

    meta = json.load(open(os.path.join(cfg.corpus_dir, "synthfuse_meta.json")))
    for v in meta.values():
        v.setdefault("source", "synth")
    # AUDIT FIX 1: merge the real harmonized cubes (produced by scripts.ingest_real)
    real_meta_path = args.real_meta or os.path.join(cfg.corpus_dir, "real_meta.json")
    if os.path.exists(real_meta_path):
        rmeta = json.load(open(real_meta_path))
        meta.update(rmeta)
        print(f"[meta] merged {len(rmeta)} real crops from {real_meta_path}")
    train_ids, val_ids = make_split(meta, cfg, limit=args.limit)
    sources = {i: meta[i].get("source", "synth") for i in train_ids + val_ids}
    json.dump({"train": train_ids, "val": val_ids, "sources": sources,
               "val_strata": {i: meta[i]["stratum"] for i in val_ids}},
              open(os.path.join(cfg.work_dir, "splits.json"), "w"), indent=2)
    n_real_train = sum(sources[i] == "real" for i in train_ids)
    print(f"[split] train={len(train_ids)} (real={n_real_train}) val={len(val_ids)} "
          f"(real val={sum(sources[i]=='real' for i in val_ids)})")

    # single-process GPU loop (fast + robust; ~1-2 s/cube on a 5090)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for k, cid in enumerate(train_ids):
        results[cid] = process_cube(cfg.corpus_dir, cid, cfg, device=dev,
                                    max_cands=args.max_cands, seed=k)
        if (k + 1) % 20 == 0 or k + 1 == len(train_ids):
            print(f"[prep] {k+1}/{len(train_ids)} cubes", flush=True)

    # assemble flat index in train order
    cols = {k: [] for k in ["vol", "z", "y", "x", "inst", "pbin", "hard", "is_std", "is_real"]}
    manifest = {}
    for vi, cid in enumerate(train_ids):
        d = results[cid]
        n = d["z"].size
        is_real = meta[cid].get("source", "synth") == "real"
        cols["vol"].append(np.full(n, vi, np.int32))
        cols["z"].append(d["z"]); cols["y"].append(d["y"]); cols["x"].append(d["x"])
        cols["inst"].append(d["inst"]); cols["pbin"].append(d["pbin"])
        cols["hard"].append(d["hard"])
        # is_std gates the Phase-A warmup pool -> keep it pure synth-std (real excluded)
        cols["is_std"].append(np.full(n, (meta[cid]["stratum"] == "std") and not is_real, bool))
        cols["is_real"].append(np.full(n, is_real, bool))
        manifest[cid] = {"shape": list(d["shape"]), "n_inst": d["n_inst"],
                         "stratum": meta[cid]["stratum"], "fg": meta[cid]["fg"],
                         "source": meta[cid].get("source", "synth"),
                         "n_contact": int(d.get("n_contact", 0)), "n_cand": int(n)}
    index = {k: np.concatenate(v) if v else np.zeros(0) for k, v in cols.items()}
    np.savez(os.path.join(cfg.work_dir, "train_index.npz"), **index)
    json.dump(manifest, open(os.path.join(cfg.work_dir, "manifest.json"), "w"), indent=2)
    tot = index["vol"].size
    print(f"[index] {tot:,} candidate centers  hard/blind={int(index['hard'].sum()):,} "
          f"({100*index['hard'].mean():.1f}%)  std={int(index['is_std'].sum()):,}  "
          f"real={int(index['is_real'].sum()):,}")
    # bin histogram
    b, c = np.unique(index["pbin"], return_counts=True)
    print("[bins]", dict(zip(b.tolist(), c.tolist())))


if __name__ == "__main__":
    main()
