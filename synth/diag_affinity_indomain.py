#!/usr/bin/env python3
"""IN-DOMAIN affinity diagnostic: run the network on SYNTH corpus cubes (the exact domain it trains on, with
per-voxel instance labels) and measure within-wrap vs cross-wrap affinity AUC.

Why this exists: diag_affinity.py scores the REAL slab and returned AUC 0.42 (sub-chance, near-constant ~0.25
output). That is consistent with TWO very different diagnoses, which this script separates:
  (a) the head never learned the task at all  -> in-domain AUC will ALSO be ~0.5
  (b) the head learned synth affinities fine but does not TRANSFER to real CT -> in-domain AUC high, slab AUC ~0.5
(b) would mean the affinity idea is sound and the problem is domain gap; (a) means training/plumbing is broken.
A third check is included: the TRAINING-TARGET sanity -- the fraction of valid r=1 edges whose target is 1. If the
head cannot even fit that trivially-imbalanced channel in-domain, the defect is structural, not a domain gap.

Runs the cube through the network in ONE forward (corpus cubes are the patch size), so it is fast.

Usage:
  python diag_affinity_indomain.py --model_dir <...FTMalisV2__...3d_fullres> --ckpt checkpoint_final.pth \
      --corpus /root/surf/data/synthfuse_corpus --ranges 1,3,9,27 --n 6 [--gpu 0]
"""
import os, sys, glob, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tstr_eval_aff import build_offsets, build_aff_network, preprocess
from diag_affinity import auc_mannwhitney


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--ckpt", default="checkpoint_final.pth")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--ranges", default="1,3,9,27")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import torch, tifffile
    device = torch.device(f"cuda:{a.gpu}")
    offsets = build_offsets(a.ranges)
    p, net = build_aff_network(a.model_dir, a.fold, a.ckpt, device)
    n_seg = int(p.label_manager.num_segmentation_heads)
    insts = sorted(glob.glob(f"{a.corpus}/labelsTr_inst/*.tif"))[:a.n]
    print(f"cubes={len(insts)} n_seg={n_seg} n_aff={len(offsets)}", flush=True)
    W, C = [], []
    Wc = {c: [] for c in range(len(offsets))}             # per-channel: POOLED AUC is an invalid gate -- an
    Cc = {c: [] for c in range(len(offsets))}             # untrained prior-bias head scores pooled 0.837 purely
    tgt1_frac_unit = []                                   # from the cross-channel spread of pi_k
    for f in insts:
        cid = os.path.basename(f)[:-4]
        img = sorted(glob.glob(f"{a.corpus}/imagesTr/{cid}_0000.tif"))
        if not img:
            continue
        vol = tifffile.imread(img[0]).astype(np.float32)
        inst = tifffile.imread(f).astype(np.int32)
        data = preprocess(p, vol)
        with torch.no_grad():
            x = torch.from_numpy(data)[None].to(device)
            with torch.autocast(device.type, enabled=True):
                out = net(x)
            out = (out[0] if isinstance(out, (tuple, list)) else out).float()[0].cpu().numpy()
        aff = 1.0 / (1.0 + np.exp(-out[n_seg:]))
        Z, Y, X = inst.shape
        assert aff.shape[1:] == inst.shape, f"aff {aff.shape} vs inst {inst.shape}"
        for c, (dz, dy, dx) in enumerate(offsets):
            z0, z1 = max(0, -dz), Z - max(0, dz)
            y0, y1 = max(0, -dy), Y - max(0, dy)
            x0, x1 = max(0, -dx), X - max(0, dx)
            s = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
            t = (slice(z0 + dz, z1 + dz), slice(y0 + dy, y1 + dy), slice(x0 + dx, x1 + dx))
            ia, ib = inst[s], inst[t]
            av = aff[c][s]
            wm = (ia > 0) & (ib > 0) & (ia == ib)
            cm = (ia > 0) & (ib > 0) & (ia != ib)
            if wm.any():
                sub = av[wm][:: max(1, int(wm.sum()) // 200_000)]
                W.append(sub); Wc[c].append(sub)
            if cm.any():
                sub = av[cm][:: max(1, int(cm.sum()) // 200_000)]
                C.append(sub); Cc[c].append(sub)
            if max(abs(dz), abs(dy), abs(dx)) == 1 and (wm.sum() + cm.sum()) > 0:
                tgt1_frac_unit.append(float(wm.sum()) / float(wm.sum() + cm.sum()))
    Wv = np.concatenate(W) if W else np.array([])
    Cv = np.concatenate(C) if C else np.array([])
    res = dict(
        n_cubes=len(insts),
        within_mean=round(float(Wv.mean()), 4) if Wv.size else None,
        cross_mean=round(float(Cv.mean()), 4) if Cv.size else None,
        separation=round(float(Wv.mean() - Cv.mean()), 4) if (Wv.size and Cv.size) else None,
        auc=round(auc_mannwhitney(Wv, Cv), 4) if (Wv.size and Cv.size) else None,
        r1_target1_fraction=round(float(np.mean(tgt1_frac_unit)), 4) if tgt1_frac_unit else None,
        within_pct=[round(float(np.percentile(Wv, q)), 4) for q in (5, 50, 95)] if Wv.size else None,
        cross_pct=[round(float(np.percentile(Cv, q)), 4) for q in (5, 50, 95)] if Cv.size else None,
    )
    per_ch = {}
    for c, off in enumerate(offsets):
        wv = np.concatenate(Wc[c]) if Wc[c] else np.array([])
        cv = np.concatenate(Cc[c]) if Cc[c] else np.array([])
        per_ch[str(off)] = round(auc_mannwhitney(wv, cv), 4) if (wv.size and cv.size) else None
    res["per_channel_auc"] = per_ch
    print(json.dumps(res, indent=1))
    # THE GATE IS PER-CHANNEL, never pooled: pooled AUC is dominated by the spread of per-channel priors (an
    # untrained prior-bias head scores pooled 0.837). Judge on the channels that carry the decode (r=1) and the
    # decision band (long in-plane).
    vals = [v for v in per_ch.values() if v is not None]
    n_ok = sum(1 for v in vals if v > 0.7)
    print(f"PER-CHANNEL AUC: {n_ok}/{len(vals)} channels > 0.7  |  " +
          "  ".join(f"{k}:{v}" for k, v in per_ch.items()))
    if vals and min(vals[:3] or [0]) > 0.7 and n_ok >= len(vals) * 0.7:
        print("IN_DOMAIN_OK -- head learned per-channel discrimination on its training domain")
    else:
        print("IN_DOMAIN_WEAK -- one or more channels near chance ON TRAINING DATA; loss/schedule issue, "
              "do not proceed to decode conclusions")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print("DIAG_INDOMAIN_DONE")


if __name__ == "__main__":
    main()
