#!/usr/bin/env python
"""Pick the best checkpoint(s) from the inline eval history (val_progress.jsonl),
under BOTH selection criteria, and (optionally) materialize them.

  NERL-best  : max NERL among evals with all-panel merge <= select_merge_eps.
               Coverage-safe, stable; the "highest-ERL" half of the FFN protocol.
  MERGE-best : min REAL merge among evals with all-panel NERL >= select_nerl_floor.
               The "fewest-mergers-first" half of the FFN protocol (Januszewski 2018),
               which better matches our merge=fatal / split=free economics. The NERL
               floor keeps a collapsed segmentation (NERL~0) from ever winning it.

This works for a RUN THAT WAS LAUNCHED BEFORE the in-loop selection existed: evals land
on steps that are multiples of eval_every (also multiples of ckpt_every), so each eval
has a matching ckpt_<step>.pt. Run any time; with --apply it copies the winners to
ckpt_best.pt / ckpt_merge_best.pt (stamping the selection metadata). The held-out slab
(Gate B) is the final arbiter between the two.

  python -m scripts.select_best --work /root/surf/ffn_work [--apply]
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffn.config import FFNConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--apply", action="store_true", help="copy winners to ckpt_best/ckpt_merge_best")
    args = ap.parse_args()
    cfg = FFNConfig.load(os.path.join(args.work, "config.json"))
    rows = [json.loads(l) for l in open(os.path.join(args.work, "val_progress.jsonl"))]

    def ck(step):
        p = os.path.join(args.work, f"ckpt_{step}.pt")
        return p if os.path.exists(p) else None

    # NERL-best: max all_nerl s.t. all_merge <= eps  (and its ckpt exists)
    nerl_cands = [r for r in rows if r.get("all_merge") is not None
                  and r["all_merge"] <= cfg.select_merge_eps and r.get("all_nerl") is not None
                  and ck(r["step"])]
    nerl_best = max(nerl_cands, key=lambda r: r["all_nerl"], default=None)
    # MERGE-best: min real_merge s.t. all_nerl >= floor (tiebreak: blind, then NERL)
    merge_cands = [r for r in rows if r.get("all_nerl") is not None
                   and r["all_nerl"] >= cfg.select_nerl_floor and r.get("real_merge") is not None
                   and ck(r["step"])]
    def mkey(r):
        blind = r.get("blind_merge"); blind = 9.9 if blind is None else blind
        return (r["real_merge"], blind, -r["all_nerl"])
    merge_best = min(merge_cands, key=mkey, default=None)

    print(f"[select] {len(rows)} evals; merge_eps={cfg.select_merge_eps} nerl_floor={cfg.select_nerl_floor}")
    for tag, r in [("NERL-best ", nerl_best), ("MERGE-best", merge_best)]:
        if r is None:
            print(f"  {tag}: (none eligible yet)"); continue
        print(f"  {tag}: step {r['step']:6d}  NERL_all={r['all_nerl']}  "
              f"merge_all={r.get('all_merge')}  real_merge={r.get('real_merge')}  "
              f"blind={r.get('blind_merge')}  inst={r.get('instances')}")

    if args.apply:
        for r, name, sel in [(nerl_best, "ckpt_best.pt", "all_nerl"),
                             (merge_best, "ckpt_merge_best.pt", "real_merge")]:
            if r is None:
                continue
            src = ck(r["step"]); d = torch.load(src, map_location="cpu")
            d["sel_nerl"] = r.get("all_nerl"); d["sel_merge"] = r.get("real_merge")
            d["sel_step"] = r["step"]
            torch.save(d, os.path.join(args.work, name))
            print(f"  wrote {name} <- ckpt_{r['step']}.pt")


if __name__ == "__main__":
    main()
