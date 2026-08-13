#!/usr/bin/env python
"""Standalone FFN evaluation harness (DESIGN.md 7).

Gate A (synth-val): fast in-domain gate on held-out synth cubes -- adjacent-wrap
    merge rate + VOI_merge + carved-skeleton bridge count. Must pass before slab.
Gate B (held-out slab): the real test -- merge rate (+ blind-contact subset),
    ERL/NERL, VOI(split/merge), adapted-Rand, mean +/- sigma over >=N seed sets.

The FFN separates a *given* foreground; seeding therefore uses a fiber mask
(synth: the binary label; slab: a CT threshold by default -- no GT leak -- or
--fiber-from-truth). The metrics score separation against the truth wraps.

Usage:
  python -m scripts.evaluate --gate A --work W --ckpt W/ckpt_last.pt --cubes 20
  python -m scripts.evaluate --gate B --work W --ckpt W/ckpt_last.pt --slab /root/data/slab
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffn.config import FFNConfig
from ffn.model import build_model
from ffn import inference as I
from ffn import seeds as S
from ffn import metrics as M


# FOV-forward batch buckets: the decode issues forwards at many batch sizes
# (screen=256, fill K=64 and every partial frontier 1..64). torch.compile in
# reduce-overhead mode wraps each *static* shape in a CUDA-graph; to bound the
# number of captured graphs we round every batch up to one of a few buckets and
# slice the padded rows back off. Padding is compute (the net is compute-bound
# above ~B=16) but the fused+graphed kernel more than pays for it, and the tiny
# tail batches -- where eager is ~90% launch overhead -- get a ~3.5x cut.
_BUCKETS = (8, 16, 32, 64, 128, 256)


def make_predict_fn(model, cfg, device, use_compile: bool = True):
    amp = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    net = model
    if use_compile and os.environ.get("FFN_NO_COMPILE") != "1":
        try:
            net = torch.compile(model, mode="reduce-overhead", dynamic=False)
        except Exception as e:  # pragma: no cover
            print(f"[eval] torch.compile unavailable ({e}); running eager", flush=True)
            net = model

    @torch.no_grad()
    def fn(inp):
        B = inp.shape[0]
        tgt = next((b for b in _BUCKETS if b >= B), None)
        if tgt is not None and tgt != B:
            pad = torch.zeros((tgt - B,) + tuple(inp.shape[1:]),
                              dtype=inp.dtype, device=inp.device)
            inp = torch.cat([inp, pad], 0)
        if cfg.channels_last:
            inp = inp.to(memory_format=torch.channels_last_3d)
        with torch.autocast("cuda", dtype=amp):
            out = net(inp)
        out = out.float()
        # reduce-overhead reuses a static graph output buffer; clone so callers
        # that hold the tensor across the next forward see stable values.
        return out[:B].clone() if out.shape[0] != B else out.clone()
    return fn


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = FFNConfig(**{k: v for k, v in ck["cfg"].items()
                       if k in FFNConfig.__dataclass_fields__})
    model = build_model(cfg).to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last_3d)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg


def downsample(vol, f):
    return vol[::f, ::f, ::f]


def upsample_labels(lab, f, shape):
    out = np.repeat(np.repeat(np.repeat(lab, f, 0), f, 1), f, 2)
    return out[:shape[0], :shape[1], :shape[2]]


def segment_consensus(predict_fn, image_np, fiber_np, cfg, seed_shuffle=None, device="cuda",
                      seed_cache=None):
    """Full inference: forward+reverse x {1x,2x} -> consensus -> agglomerate (if enabled).
    `seed_cache`: optional per-cube dict -- skeletonization depends only on the fiber
    mask, so seeds are cached across seed-sets/checkpoints (saves minutes per cube)."""
    dev = torch.device(device)
    image = torch.from_numpy(((image_np.astype(np.float32) / 255.0 - 0.5) / 0.5)).to(dev)

    def seeds_and_viable(scale, img_gpu, fiber_src):
        """Cached (coords, viability) for a scale. Both depend only on the fiber
        mask + model, NOT on seed order/direction/seed-set, so we compute the
        (expensive) CPU skeleton seeds and the batched dead-seed pre-screen ONCE
        per cube-scale and reuse them across forward/reverse and every seed-set."""
        if seed_cache is not None and scale in seed_cache:
            coords = seed_cache[scale]
        else:
            if scale == 1:
                coords, _ = S.inference_seeds(fiber_src.astype(np.float32), thr=0.5,
                                              min_edt=cfg.inf_seed_min_edt,
                                              spacing=cfg.seed_spacing)
            else:
                coords, _ = S.inference_seeds(fiber_src.astype(np.float32), thr=0.5,
                                              min_edt=1.0,
                                              spacing=max(2, cfg.seed_spacing // scale))
            if seed_cache is not None:
                seed_cache[scale] = coords
        vkey = ("viable", scale)
        if seed_cache is not None and vkey in seed_cache:
            viable = seed_cache[vkey]
        else:
            viable = (I.screen_seeds(predict_fn, img_gpu, coords, cfg)
                      if len(coords) else np.zeros(0, bool))
            if seed_cache is not None:
                seed_cache[vkey] = viable
        return coords, viable

    coords, viable = seeds_and_viable(1, image, fiber_np)
    if seed_shuffle is not None and len(coords):
        rng = np.random.default_rng(seed_shuffle)
        perm = rng.permutation(len(coords))
        coords, viable = coords[perm], viable[perm]  # keep coords/viable aligned
    runs = []
    runs.append(I.segment_block(predict_fn, image, coords, cfg, reverse=False, viable=viable))
    runs.append(I.segment_block(predict_fn, image, coords, cfg, reverse=True, viable=viable))
    # 2x scale
    for f in cfg.consensus_scales:
        if f == 1:
            continue
        img2 = downsample(image_np, f)
        image2 = torch.from_numpy(((img2.astype(np.float32) / 255.0 - 0.5) / 0.5)).to(dev)
        c2, v2 = seeds_and_viable(f, image2, downsample(fiber_np, f))
        r2 = I.segment_block(predict_fn, image2, c2, cfg, reverse=False, viable=v2)
        runs.append(upsample_labels(r2, f, image_np.shape))
    cons = I.consensus(runs, min_size=cfg.min_instance_size // 4)
    # AUDIT FIX 2: agglomeration regrow-test false-merges blind contacts; default OFF so
    # the merge-safe oversegmentation-consensus is canonical (splits are downstream-free).
    if getattr(cfg, "agglo_enabled", False):
        return I.agglomerate(predict_fn, image, cons, cfg)
    return cons


def gate_A(model, cfg, work, n_cubes, seed_sets, device, shard=0, n_shards=1):
    splits = json.load(open(os.path.join(work, "splits.json")))
    # balanced panel: half real, half synth (sorted val would otherwise be all-real first)
    reals = [i for i in splits["val"] if i.startswith("real_")]
    synths = [i for i in splits["val"] if not i.startswith("real_")]
    val_ids = (reals[:n_cubes - n_cubes // 2] + synths[:n_cubes // 2]) or splits["val"][:n_cubes]
    # multi-GPU cube sharding: each process decodes an interleaved subset (keeps the
    # real/synth mix balanced across shards). Results are merged downstream. Exact --
    # cube decode is independent, so this only changes *which* process runs a cube.
    if n_shards > 1:
        val_ids = val_ids[shard::n_shards]
    predict_fn = make_predict_fn(model, cfg, device)
    results = []
    for cid in val_ids:
        img = tifffile.imread(os.path.join(cfg.corpus_dir, "imagesTr", f"{cid}_0000.tif"))
        inst = tifffile.imread(os.path.join(cfg.corpus_dir, "labelsTr_inst", f"{cid}.tif")).astype(np.int32)
        fiber = (inst > 0).astype(np.float32)
        mr, vm, br, rq, pq = [], [], [], [], []
        cube_seeds = {}                      # skeleton cache shared across seed-sets
        for s in range(seed_sets):
            pred = segment_consensus(predict_fn, img, fiber, cfg, seed_shuffle=s, device=device,
                                     seed_cache=cube_seeds)
            r = M.adjacent_wrap_merge_rate(inst, pred, cfg.adjacency_radius, cfg.merge_theta)
            _, vmerge = M.voi_split_merge(inst, pred)
            e = M.erl(inst, pred)                            # NERL: the coverage-floored objective
            mr.append(r["merge_rate"]); vm.append(vmerge); br.append(M.bridge_count(pred))
            rq.append(e["sum_runsq"]); pq.append(e["sum_perfsq"])
        rq_m, pq_m = float(np.mean(rq)), float(np.mean(pq))
        res = dict(cube=cid, is_real=cid.startswith("real_"),
                   merge_rate=float(np.mean(mr)), merge_rate_std=float(np.std(mr)),
                   voi_merge=float(np.mean(vm)), bridge=float(np.mean(br)),
                   sum_runsq=rq_m, sum_perfsq=pq_m,
                   nerl=(rq_m / pq_m if pq_m > 0 else 0.0))
        results.append(res)
        print(f"[A]{'R' if res['is_real'] else ' '} {cid} "
              f"NERL={res['nerl']:.3f} merge={res['merge_rate']:.3f}+/-{res['merge_rate_std']:.3f} "
              f"voi_m={res['voi_merge']:.3f} bridge={res['bridge']:.1f}", flush=True)

    return report_gate_A(results, cfg)


def report_gate_A(results, cfg):
    """Aggregate per-cube gate-A results into the synth/real/overall verdict. Kept
    separate so a multi-GPU run can merge per-shard result lists and score once."""
    def agg(subset):
        if not subset:
            return None
        rq = sum(r.get("sum_runsq", 0.0) for r in subset)
        pq = sum(r.get("sum_perfsq", 0.0) for r in subset)
        return dict(nerl=(rq / pq if pq > 0 else 0.0),   # exact panel NERL = Σrunsq / Σperfsq
                    merge_rate=float(np.mean([r["merge_rate"] for r in subset])),
                    voi_merge=float(np.mean([r["voi_merge"] for r in subset])),
                    bridge_ok_frac=float(np.mean([r["bridge"] == 0 for r in subset])),
                    n=len(subset))
    synth = agg([r for r in results if not r["is_real"]])
    real = agg([r for r in results if r["is_real"]])       # AUDIT FIX 1: REAL val gate
    overall = agg(results)
    nerl = overall["nerl"]
    mrate, vmerge, bridge_ok = overall["merge_rate"], overall["voi_merge"], overall["bridge_ok_frac"]
    # SOTA gate: NERL is the coverage-floored OBJECTIVE (empty seg -> 0 -> FAIL); merge-rate,
    # VOI_merge, bridges are CONSTRAINTS. This closes the hole where a near-empty
    # segmentation passed all the old (coverage-blind) terms.
    passed = (nerl >= cfg.gateA_nerl and mrate < cfg.gateA_merge_rate
              and vmerge < cfg.gateA_voi_merge and bridge_ok >= cfg.gateA_bridge_ok_frac)
    if synth:
        print(f"[GATE A synth] NERL={synth['nerl']:.3f} merge={synth['merge_rate']:.3f} "
              f"voi_m={synth['voi_merge']:.3f} bridge0={synth['bridge_ok_frac']:.2f} (n={synth['n']})", flush=True)
    if real:
        rpass = real["nerl"] >= cfg.gateA_nerl and real["merge_rate"] < cfg.gateA_merge_rate
        print(f"[GATE A REAL ] NERL={real['nerl']:.3f} merge={real['merge_rate']:.3f} "
              f"voi_m={real['voi_merge']:.3f} bridge0={real['bridge_ok_frac']:.2f} (n={real['n']}) -> "
              f"{'PASS' if rpass else 'FAIL'}   <-- real-appearance gate", flush=True)
    print(f"[GATE A all  ] NERL={nerl:.3f}(>={cfg.gateA_nerl})  "
          f"merge_rate={mrate:.3f}(<{cfg.gateA_merge_rate}) "
          f"voi_merge={vmerge:.3f}(<{cfg.gateA_voi_merge}) "
          f"bridge0_frac={bridge_ok:.2f}(>={cfg.gateA_bridge_ok_frac}) -> "
          f"{'PASS' if passed else 'FAIL'}", flush=True)
    return dict(results=results, nerl=float(nerl), merge_rate=float(mrate), voi_merge=float(vmerge),
                bridge_ok_frac=float(bridge_ok), passed=bool(passed),
                synth=synth, real=real)


def gate_B(model, cfg, slab_dir, seed_sets, device, fiber_from_truth=False):
    import nrrd
    vol, _ = nrrd.read(os.path.join(slab_dir, "volume.nrrd"))
    truth, _ = nrrd.read(os.path.join(slab_dir, "truth.nrrd"))
    truth = truth.astype(np.int32)
    if fiber_from_truth:
        fiber = (truth > 0).astype(np.float32)
    else:
        thr = np.percentile(vol[vol > 0], 60)  # CT-threshold fiber proxy (no GT leak)
        fiber = (vol >= thr).astype(np.float32)
    predict_fn = make_predict_fn(model, cfg, device)
    mr, bmr, nerl, vm, vs, are = [], [], [], [], [], []
    slab_seeds = {}                          # skeleton cache shared across seed-sets
    for s in range(seed_sets):
        pred = segment_consensus(predict_fn, vol, fiber, cfg, seed_shuffle=s, device=device,
                                 seed_cache=slab_seeds)
        r = M.adjacent_wrap_merge_rate(truth, pred, cfg.adjacency_radius, cfg.merge_theta, image=vol)
        e = M.erl(truth, pred)
        vsplit, vmerge = M.voi_split_merge(truth, pred)
        ar = M.adapted_rand(truth, pred)
        mr.append(r["merge_rate"]); bmr.append(r["blind_merge_rate"])
        nerl.append(e["nerl"]); vm.append(vmerge); vs.append(vsplit); are.append(ar["are"])
        print(f"[B seed {s}] merge={r['merge_rate']:.3f} blind={r['blind_merge_rate']:.3f} "
              f"nerl={e['nerl']:.3f} voi_m={vmerge:.3f} voi_s={vsplit:.3f} are={ar['are']:.3f}",
              flush=True)
    def ms(a):
        a = np.array(a, float); a = a[~np.isnan(a)]
        return (float(a.mean()), float(a.std())) if a.size else (float("nan"), 0.0)
    out = dict(merge_rate=ms(mr), blind_merge_rate=ms(bmr), nerl=ms(nerl),
               voi_merge=ms(vm), voi_split=ms(vs), adapted_rand=ms(are))
    print(f"[GATE B] merge={out['merge_rate']} blind={out['blind_merge_rate']} "
          f"nerl={out['nerl']} voi_merge={out['voi_merge']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", choices=["A", "B"], required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--slab", default="/root/data/slab")
    ap.add_argument("--cubes", type=int, default=20)
    ap.add_argument("--seed-sets", type=int, default=None)
    ap.add_argument("--fiber-from-truth", action="store_true")
    ap.add_argument("--seed-spacing", type=int, default=None,
                    help="override inference seed grid spacing (coarser = faster)")
    ap.add_argument("--shard", type=int, default=0, help="gate-A cube-shard index")
    ap.add_argument("--n-shards", type=int, default=1,
                    help="gate-A number of cube shards (one process/GPU each)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, device)
    # respect work_dir corpus path
    wc = FFNConfig.load(os.path.join(args.work, "config.json"))
    cfg.corpus_dir, cfg.slab_dir = wc.corpus_dir, wc.slab_dir
    if args.seed_spacing:
        cfg.seed_spacing = args.seed_spacing
    ss = args.seed_sets or cfg.eval_seed_sets
    if args.gate == "A":
        out = gate_A(model, cfg, args.work, args.cubes, ss, device,
                     shard=args.shard, n_shards=args.n_shards)
    else:
        out = gate_B(model, cfg, args.slab, ss, device, args.fiber_from_truth)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2, default=float)
        print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
