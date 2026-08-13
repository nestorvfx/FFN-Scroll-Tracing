#!/usr/bin/env python
"""Merge per-shard Gate-A result JSONs (from a multi-GPU cube-sharded run) and
score the combined panel with the canonical synth/real/overall verdict.

Each `scripts.evaluate --gate A --n-shards N --shard i --out shard_i.json` process
decodes an interleaved subset of the val panel and writes its per-cube results;
this recombines them and applies `report_gate_A` once so the gate decision is
identical to a single-process run (cube decode is independent of sharding)."""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffn.config import FFNConfig
from scripts.evaluate import report_gate_A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+", help="per-shard result JSON files")
    ap.add_argument("--ckpt", required=True, help="ckpt whose cfg holds gate thresholds")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    results = []
    for s in args.shards:
        d = json.load(open(s))
        results += d.get("results", [])
    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = FFNConfig(**{k: v for k, v in ck["cfg"].items()
                       if k in FFNConfig.__dataclass_fields__})
    out = report_gate_A(results, cfg)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2, default=float)
        print(f"[merge] wrote {args.out}")


if __name__ == "__main__":
    main()
