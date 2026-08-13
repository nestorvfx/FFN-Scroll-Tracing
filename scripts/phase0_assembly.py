#!/usr/bin/env python
"""Does an over-segment -> resegment-agglomerate pipeline beat the single-pass decode?

THE HYPOTHESIS UNDER TEST. Today's k-distribution showed 77% of merges are genuinely fused (no
threshold separates them), so merge control cannot come from the decode. The alternative is to buy
merge-safety by over-segmenting -- the ratchet freeze sweep gives a monotone knob for this
(freeze 0.9 => merge 0.041 vs 0.107 at the shipped 0.5, at the cost of coverage and 1.5x the
objects) -- and then recover the splits with `ffn.agglomerate.agglomerate_resegment`, which is
merge-safe by construction (two independent regrowths must agree).

WHAT WOULD MAKE THIS A WIN. Against the shipped operating point (freeze 0.5, no assembly):
  merge   <= baseline           (never buy NERL with a merge; that is a rejection)
  NERL     > baseline           (the splits that over-segmentation cost are recovered)
Report also how many pair-merges the criterion accepted vs rejected, and WHY it rejected them --
if it accepts almost nothing, the assembly stage is inert and that is the finding.

Usage:
  python -m scripts.phase0_assembly --work DIR --ckpt PATH [--freeze 0.5,0.7,0.9] [--device cuda:0]
"""
import argparse
import dataclasses
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ffn.agglomerate import agglomerate_resegment
from ffn.config import FFNConfig
from ffn import inference as I
from ffn.metrics import adjacency_pairs, erl as erl_metric
from scripts.phase0_decode import load_panel, make_predict


def panel_metrics(panel, preds, cfg):
    rr = rp = sr = sp = 0.0
    nm = npair = 0
    cov_n = cov_d = ninst = 0
    for cb, pred in zip(panel, preds):
        gt = cb["inst"]
        e = erl_metric(gt, pred)
        if cb["is_real"]:
            rr += e["sum_runsq"]; rp += e["sum_perfsq"]
        else:
            sr += e["sum_runsq"]; sp += e["sum_perfsq"]
        sizes = {int(i): int((gt == i).sum()) for i in np.unique(gt) if i}
        for (i, j) in adjacency_pairs(gt, cfg.adjacency_radius):
            npair += 1
            for L in np.unique(pred[(gt == i) & (pred > 0)]):
                if L and (pred[gt == i] == L).sum() / max(sizes[i], 1) >= cfg.merge_theta \
                   and (pred[gt == j] == L).sum() / max(sizes[j], 1) >= cfg.merge_theta:
                    nm += 1
                    break
        cov_n += int(((gt > 0) & (pred > 0)).sum()); cov_d += int((gt > 0).sum())
        ninst += len(np.unique(pred[pred > 0]))
    return dict(real_nerl=rr / max(rp, 1e-9), synth_nerl=sr / max(sp, 1e-9),
                merge=nm / max(npair, 1), n_pairs=npair, coverage=cov_n / max(cov_d, 1),
                inst=ninst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--freeze", default="0.5,0.7,0.9")
    ap.add_argument("--n-real", type=int, default=8)
    ap.add_argument("--n-synth", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache", default="/root/panel_cache.npz")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--max-pairs", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    cfg0 = FFNConfig.load(os.path.join(a.work, "config.json"))
    pf, step = make_predict(cfg0, a.ckpt, dev)
    panel = load_panel(cfg0, a.work, a.n_real, a.n_synth, a.cache, a.jobs)
    print(f"ckpt {step} | panel {len(panel)} cubes | agglo iou>{cfg0.agglo_iou} "
          f"cons>{cfg0.agglo_consistency} del<{cfg0.agglo_deleted_frac} r={cfg0.agglo_radius}",
          flush=True)
    print(f"\n{'config':>22s} {'real_nerl':>9s} {'synth_nerl':>10s} {'merge':>7s} "
          f"{'coverage':>8s} {'inst':>5s}", flush=True)
    base = None
    for fp in [float(x) for x in a.freeze.split(",")]:
        cfg = dataclasses.replace(cfg0, fill_max_steps=cfg0.eval_fill_max_steps,
                                  agglo_enabled=False, pom_ratchet_freeze_p=fp)
        preds = []
        for cb in panel:
            g = torch.from_numpy(((cb["img"].astype(np.float32) / 255.0 - 0.5) / 0.5)).to(dev)
            with torch.no_grad():
                preds.append(I.segment_block(pf, g, cb["seeds"], cfg, K=128))
        m = panel_metrics(panel, preds, cfg)
        tag = f"freeze {fp:.2f} raw"
        print(f"{tag:>22s} {m['real_nerl']:9.4f} {m['synth_nerl']:10.4f} {m['merge']:7.4f} "
              f"{m['coverage']:8.4f} {m['inst']:5d}", flush=True)
        if abs(fp - 0.5) < 1e-9:
            base = m
        # ---- now agglomerate ----
        t0 = time.time()
        aggs, reasons, n_ok, n_try = [], Counter(), 0, 0
        for cb, pred in zip(panel, preds):
            g = torch.from_numpy(((cb["img"].astype(np.float32) / 255.0 - 0.5) / 0.5)).to(dev)
            st = {}
            out = agglomerate_resegment(pf, g, pred, cfg, K=128, stats=st,
                                        max_pairs=a.max_pairs)
            aggs.append(out)
            n_ok += st.get("n_merged", 0); n_try += st.get("n_pairs", 0)
            for p in st.get("pairs", []):
                reasons[p["reason"] or "ACCEPT"] += 1
        ma = panel_metrics(panel, aggs, cfg)
        tag = f"freeze {fp:.2f} +agglo"
        print(f"{tag:>22s} {ma['real_nerl']:9.4f} {ma['synth_nerl']:10.4f} {ma['merge']:7.4f} "
              f"{ma['coverage']:8.4f} {ma['inst']:5d}   "
              f"[{n_ok}/{n_try} pairs merged, {time.time()-t0:.0f}s]", flush=True)
        print(f"{'':>22s} reject reasons: {dict(reasons.most_common(6))}", flush=True)
        if base is not None:
            dn = ma["real_nerl"] - base["real_nerl"]
            dm = ma["merge"] - base["merge"]
            verdict = ("WIN" if dn > 0 and dm <= 0 else
                       "REJECT (bought NERL with a merge)" if dn > 0 and dm > 0 else
                       "no gain")
            print(f"{'':>22s} vs shipped baseline: dNERL {dn:+.4f} dmerge {dm:+.4f} -> {verdict}",
                  flush=True)
    print("\nGround rule: a NERL gain that adds a merge is a REJECTION. Instrument: NERL sigma "
          "0.029-0.057, merge quantum 1/n_pairs.")


if __name__ == "__main__":
    main()
