#!/usr/bin/env python
"""Dual-GPU (DDP) FFN training (DESIGN.md 6). One command:

  torchrun --nproc_per_node=2 -m scripts.train --work /root/surf/ffn_work

Each rank loads the whole GPU-resident cube stack and draws independent seed
batches from the shared candidate index (DESIGN.md 6.4); gradients all-reduce.
Rank 0 logs and checkpoints. Runs cleanly single-GPU too (no torchrun).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffn.config import FFNConfig
from ffn.model import build_model
from ffn.volumes import GpuVolumeCache, AsyncPatchLoader
from ffn.train_step import TrainStepper
from ffn.curriculum import Curriculum
from ffn.inline_eval import InlineEvaluator, log_val


def setup_ddp():
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local = int(os.environ.get("LOCAL_RANK", rank))
        world = dist.get_world_size()
        torch.cuda.set_device(local)
        return rank, local, world, True
    local = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    return 0, local, 1, False


def load_cubes(cfg, train_ids, device, rank):
    """Load the cube stacks with a PREALLOCATED destination.

    The previous form appended every cube to a Python list and then called `torch.stack(...)`,
    which allocates a second full-size tensor while the list still holds the first -- peak RAM is
    2x the resident stack. At 3470 cubes of 192^3 the stack is ~74 GB (uint8 image 24.6 GB +
    int16 inst 49.2 GB), so the peak was ~148 GB: fine on the 192 GB box this was written on,
    OOM-killed at cube 3450/3470 on a 125 GB box. Filling a preallocated tensor in place holds the
    peak at the resident size, and skips a full 74 GB copy.
    """
    n = len(train_ids)
    stack_dev = device if getattr(cfg, "cubes_on_gpu", True) else torch.device("cpu")
    image_stack = inst_stack = None
    mx = 0
    for k, cid in enumerate(train_ids):
        img = tifffile.imread(os.path.join(cfg.corpus_dir, "imagesTr", f"{cid}_0000.tif"))
        inst = tifffile.imread(os.path.join(cfg.corpus_dir, "labelsTr_inst", f"{cid}.tif"))
        if image_stack is None:                      # shape known only after the first read
            image_stack = torch.empty((n, *img.shape), dtype=torch.uint8, device=stack_dev)
            inst_stack = torch.empty((n, *inst.shape), dtype=torch.int16, device=stack_dev)
        image_stack[k] = torch.from_numpy(img.astype(np.uint8))
        inst_stack[k] = torch.from_numpy(inst.astype(np.int16))
        mx = max(mx, int(inst.max()))
        if rank == 0 and (k + 1) % 50 == 0:
            print(f"[load] {k+1}/{n} cubes", flush=True)
    if mx < 256:
        # instance ids fit uint8 (composer caps sheets at 140): halves the CPU-gather traffic.
        # Transient +N/2 bytes during the cast, then the int16 buffer is released.
        inst_stack = inst_stack.to(torch.uint8)
    if rank == 0:
        gb = (image_stack.numel() + inst_stack.numel() * inst_stack.element_size()) / 1e9
        print(f"[load] inst dtype {inst_stack.dtype} (max id {mx}) | stacks ~{gb:.1f} GB",
              flush=True)
    return image_stack, inst_stack


def load_optimizer_across_growth(opt, saved, rank=0):
    """Restore Adam moments when the model has GROWN (research/13 Levers #1/#2).

    `opt.load_state_dict(saved)` raises "parameter group that doesn't match the size of optimizer's
    group" the moment a new parameter appears (the move head) or an existing one changes shape
    (in_conv1 2->3 channels). Dropping the optimizer state instead would restart Adam's second
    moment from zero on a converged model, which produces a large transient in the first few hundred
    steps -- indistinguishable from the mechanism under test, and therefore fatal to an A/B.

    Parameter registration order is preserved (the move head registers last and existing modules are
    untouched), so saved index i maps to new index i. Grown tensors are zero-padded on the new
    slice, matching the zero-init of the weights themselves; genuinely new parameters simply start
    with no state, which is what a fresh Adam does anyway.
    """
    cur = opt.state_dict()
    ssd = saved.get("state", {})
    n_new = sum(len(g["params"]) for g in cur["param_groups"])
    params = [p for g in opt.param_groups for p in g["params"]]
    remapped, grown, fresh = {}, 0, 0
    for i in range(n_new):
        st = ssd.get(i, ssd.get(str(i)))
        if st is None:
            fresh += 1
            continue
        out = {}
        for k, v in st.items():
            if torch.is_tensor(v) and v.dim() > 0 and tuple(v.shape) != tuple(params[i].shape):
                nv = torch.zeros_like(params[i])
                sl = tuple(slice(0, min(a, b)) for a, b in zip(v.shape, params[i].shape))
                nv[sl] = v[sl].to(nv.device, nv.dtype)
                out[k] = nv
                grown += 1
            else:
                out[k] = v
        remapped[i] = out
    groups = []
    for gnew, gold in zip(cur["param_groups"], saved.get("param_groups", cur["param_groups"])):
        g = dict(gold)
        g["params"] = gnew["params"]          # indices must describe the NEW parameter list
        groups.append(g)
    opt.load_state_dict({"state": remapped, "param_groups": groups})
    if rank == 0:
        print(f"[resume] optimizer state remapped across growth: {len(remapped)} restored "
              f"({grown} tensors zero-padded), {fresh} new parameters start fresh", flush=True)


def lr_at(step, cfg, warmup=1000):
    """LR at `step`: cosine for a fresh run, WSD when extending a completed one.

    A cosine is horizon-dependent -- it needs the total length up front and anneals to ~0 at it.
    Extending such a run by raising max_steps recomputes the same curve, so the LR jumps back up
    (for run-5: 2.7e-6 -> ~3.5e-4, a ~100x step change into a converged model). Re-warming from the
    minimum is the documented cause of instability and forgetting (Ibrahim et al. 2024).

    WSD (Hu et al. 2024; Singh et al. ICML 2025) replaces it with warm -> stable -> decay anchored at
    `extend_base_step`, which is horizon-FLEXIBLE: extending again later lengthens the STABLE phase
    instead of restarting the schedule, so no further discontinuity is ever needed.
    """
    b = getattr(cfg, "extend_base_step", 0)
    if b and step >= b:
        t = step - b
        n = max(1, cfg.max_steps - b)
        w = max(1, getattr(cfg, "extend_warm_steps", 5000))
        peak = getattr(cfg, "extend_peak_lr", 1e-4)
        d = max(1, int(getattr(cfg, "extend_decay_frac", 0.25) * n))
        if t < w:                                  # re-warm out of the annealed state
            return peak * t / w
        if t < n - d:                              # stable plateau -- the bulk of the extension
            return peak
        p = (t - (n - d)) / d                      # final anneal
        return 0.5 * peak * (1 + math.cos(math.pi * min(1.0, p)))
    if step < warmup:
        return cfg.lr * step / warmup
    p = (step - warmup) / max(1, cfg.max_steps - warmup)
    return 0.5 * cfg.lr * (1 + math.cos(math.pi * min(1.0, p)))


def fmt_eta(seconds):
    seconds = max(0, int(seconds))
    d, r = divmod(seconds, 86400)
    h, r = divmod(r, 3600)
    m, _ = divmod(r, 60)
    return (f"{d}d{h:02d}h{m:02d}m" if d else f"{h}h{m:02d}m")


def write_progress(work, step, max_steps, st, loss, lr, fov_sps, sps_ema, eta, elapsed,
                   val=None):
    """Machine-readable snapshot for the PowerShell tracking command."""
    prog = dict(step=step, max_steps=max_steps, pct=round(100.0 * step / max_steps, 2),
                phase=st.phase, T=st.T, real_frac=round(st.real_frac, 3),
                loss=round(float(loss), 4), lr=float(f"{lr:.3e}"),
                fov_steps_s=round(fov_sps), steps_s=round(sps_ema, 2),
                eta_seconds=int(eta), eta=fmt_eta(eta),
                elapsed_seconds=int(elapsed), elapsed=fmt_eta(elapsed),
                val=val, updated_unix=int(time.time()))
    tmp = os.path.join(work, "progress.json.tmp")
    with open(tmp, "w") as f:
        json.dump(prog, f)
    os.replace(tmp, os.path.join(work, "progress.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--steps", type=int, default=None, help="override max_steps")
    ap.add_argument("--smoke", action="store_true", help="short end-to-end run")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--compile-mode", default=None)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--resume", nargs="?", const="auto", default=None,
                    help="resume from a checkpoint: --resume (auto-picks the highest "
                         "ckpt_*.pt in --work) or --resume /path/to/ckpt.pt. Restores "
                         "model (+optimizer when present) and continues the "
                         "curriculum/LR schedule from the saved step.")
    ap.add_argument("--extend", type=int, default=None, metavar="N",
                    help="continue a COMPLETED run for N more steps using a WSD schedule "
                         "(warm -> stable -> decay) anchored at the resumed checkpoint's step. "
                         "Implies --resume. Cosine cannot be extended: raising --steps alone "
                         "recomputes the same curve and jumps the LR back up ~100x.")
    ap.add_argument("--tracer", action="store_true",
                    help="apply the tracer retrain profile (FFNConfig.as_tracer): face-max movement "
                         "gate, delta 6, walk_radius 12, off-centre training starts (threshold/"
                         "pom_init/w_other stay at their defaults -- the old help text advertised "
                         "reverted values). NOTE run-5b/run-6 checkpoints already carry this "
                         "profile in their saved config; --resume inherits it without this flag.")
    ap.add_argument("--core-dirs", default=None,
                    help="comma-separated dirs of UNLABELLED held-out core cubes (each with "
                         "imagesTr/*_0000.tif) for the movement probe. The labelled val panel is the "
                         "easy regime -- it read frac_1step 0.067 on run-2 while the cores read 0.556 "
                         "-- so core_frac_1step is the column that exposes a non-tracing model.")
    args = ap.parse_args()

    if args.extend and not args.resume:
        args.resume = "auto"                      # extending necessarily continues a checkpoint
    # cuDNN autotuning: pick the fastest algorithm per conv configuration. Our shapes are FIXED
    # for a whole run (fov 33^3, batch_per_gpu constant), which is exactly the case the benchmark
    # cache is designed for -- it pays its one-off search cost in the first few steps and then wins
    # for the rest of the run. It was never set anywhere in this repo. Algorithm SELECTION only:
    # it does not change numerics beyond the usual non-determinism of cuDNN algo choice, so it is
    # off when deterministic mode is requested.
    if torch.cuda.is_available() and not getattr(FFNConfig, "deterministic", False):
        torch.backends.cudnn.benchmark = True
    rank, local, world, ddp = setup_ddp()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    cfg = FFNConfig.load(os.path.join(args.work, "config.json"))
    if args.tracer:
        cfg = cfg.as_tracer()
    if args.core_dirs:
        cfg.eval_core_dirs = [d for d in args.core_dirs.split(",") if d]
    if args.steps:
        cfg.max_steps = args.steps
    if args.compile_mode:
        cfg.compile_mode = args.compile_mode
    if args.no_compile:
        cfg.compile = False
    if args.smoke:
        cfg.max_steps = args.steps or 300
        cfg.warmup_steps = min(cfg.warmup_steps, 120)
        cfg.rampB_steps = min(cfg.rampB_steps, 240)
        cfg.teacher_force_steps = min(cfg.teacher_force_steps, 180)
        cfg.compile_mode = args.compile_mode or "default"

    if rank == 0:
        # Persist the EFFECTIVE config. preprocess.py writes its own config.json (overwriting any
        # hand-edit), and --tracer used to leave the work dir describing a DIFFERENT run -- so a
        # `--resume` without --tracer silently continued with the centre gate against tracer-trained
        # weights. Writing it here makes the work dir always describe the run that actually ran.
        cfg.save(os.path.join(args.work, "config.json"))
    splits = json.load(open(os.path.join(args.work, "splits.json")))
    index = dict(np.load(os.path.join(args.work, "train_index.npz")))
    if rank == 0 and args.tracer:
        print(f"[cfg] TRACER profile: gate={cfg.move_gate} thr={cfg.move_threshold} "
              f"pom_init={cfg.pom_init} delta={cfg.delta} offcentre={cfg.train_offcentre} "
              f"min_fov_steps={cfg.min_fov_steps} w_other={cfg.w_other}", flush=True)
    if rank == 0:
        print(f"[cfg] world={world} batch/gpu={cfg.batch_per_gpu} fov={cfg.fov} "
              f"depth={cfg.depth} compile={cfg.compile}({cfg.compile_mode}) "
              f"cubes={len(splits['train'])} cand={index['vol'].size:,}", flush=True)

    t0 = time.time()
    image_stack, inst_stack = load_cubes(cfg, splits["train"], device, rank)
    cache = GpuVolumeCache(image_stack, inst_stack, index, cfg, device)
    if rank == 0:
        gb = (image_stack.element_size() * image_stack.nelement() +
              inst_stack.element_size() * inst_stack.nelement()) / 1e9
        print(f"[load] cubes resident on GPU: {gb:.1f} GB in {time.time()-t0:.0f}s", flush=True)

    raw_model = build_model(cfg).to(device)
    if cfg.channels_last:
        raw_model = raw_model.to(memory_format=torch.channels_last_3d)

    # ---- DEFINITIVE cudagraph-crash fix (do NOT rely on the mode string alone) ----
    # This recurrent unroll does forward->backward with GRADIENT ACCUMULATION T times
    # per optimizer step. CUDA-graph trees are fundamentally incompatible with that
    # pattern: the graph's static output buffer that feeds the *accumulating* backward
    # gets overwritten by the next micro-step's forward, so the backward reads stale
    # memory ("output of CUDAGraphs ... overwritten"). This is a known, unresolved
    # limitation (pytorch/pytorch#169545): cudagraph_mark_step_begin() and cloning do
    # NOT fix it for the backward+accumulate case -- which is why the mark_step attempt
    # never took. So we HARD-disable cudagraph capture globally (independent of whatever
    # `mode` a stale config.json/CLI requests) while KEEPING the autotuned Triton kernels.
    import torch._inductor.config as _ind_cfg
    _ind_cfg.triton.cudagraphs = False           # never capture cudagraphs
    if hasattr(_ind_cfg.triton, "cudagraph_trees"):
        _ind_cfg.triton.cudagraph_trees = False
    # DDPOptimizer graph-splitting adds multiple subgraphs per forward (a known cudagraph
    # breaker) for negligible comm/compute-overlap benefit on this 0.47M-param net; the
    # whole-graph compile is more robust for a 15h+ run. Numerics are identical either way.
    torch._dynamo.config.optimize_ddp = False
    # Normalize any cudagraph-enabling mode to its no-cudagraphs twin (keeps max-autotune
    # Triton kernels; drops only the fragile graph capture). Belt-and-suspenders with the
    # inductor flag above so a persisted `max-autotune` in config.json can't re-enable it.
    _mode = cfg.compile_mode
    if _mode == "max-autotune":
        _mode = "max-autotune-no-cudagraphs"
    if rank == 0:
        print(f"[compile] mode={_mode} cudagraphs=OFF optimize_ddp=OFF", flush=True)

    # DDP first, then compile the wrapped module (recommended order for 2.x)
    sync_module = None
    base = raw_model
    if ddp:
        base = torch.nn.parallel.DistributedDataParallel(
            raw_model, device_ids=[local],
            gradient_as_bucket_view=True, broadcast_buffers=False)
        sync_module = base
    fwd_net = torch.compile(base, mode=_mode) if cfg.compile else base

    opt = torch.optim.AdamW(raw_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ---- resume: restore model (+optimizer when present) and continue the schedule ----
    start_step = 0
    if args.resume:
        ck_path = args.resume
        if ck_path == "auto":
            import glob as _glob
            # ONLY numeric-stemmed snapshots carry their step in the filename. The TAGGED
            # checkpoints (ckpt_best / ckpt_merge_best / ckpt_core_best) do not, and the old
            # filter excluded just ckpt_last.pt -- so `int("merge")` crashed every auto-resume
            # the moment a selected checkpoint existed. --resume had therefore never worked
            # alongside selection; verifying that the FILES restore is not the same as
            # verifying that a resume RUNS.
            cands = []
            for p in _glob.glob(os.path.join(args.work, "ckpt_*.pt")):
                stem = os.path.basename(p)[5:-3]          # strip "ckpt_" and ".pt"
                if stem.isdigit():
                    cands.append((int(stem), p))
            if os.path.exists(os.path.join(args.work, "ckpt_last.pt")):
                last = torch.load(os.path.join(args.work, "ckpt_last.pt"),
                                  map_location="cpu")
                cands.append((int(last.get("step", 0)), os.path.join(args.work, "ckpt_last.pt")))
                del last
            if not cands:
                sys.exit("[resume] no ckpt_*.pt found in " + args.work)
            ck_path = max(cands)[1]
        ck = torch.load(ck_path, map_location=device)
        # CHANNEL GROWTH (cfg.neg_channel / research/13 Lever #1): a checkpoint trained with 2 input
        # channels loads into a 3-channel model by zero-initialising the new slice, which is exactly
        # function-preserving -- so the negative-evidence channel fine-tunes from ckpt_712500 instead
        # of forcing a from-scratch run. Verified bitwise in tests/test_neg_channel.py.
        _sd = ck["model"]
        _wk = "in_conv1.weight"
        if _wk in _sd and _sd[_wk].shape[1] != raw_model.in_conv1.weight.shape[1]:
            _old, _new = _sd[_wk].shape[1], raw_model.in_conv1.weight.shape[1]
            if _new < _old:
                sys.exit(f"[resume] refusing to SHRINK in_channels {_old} -> {_new}")
            _w = torch.zeros_like(raw_model.in_conv1.weight)
            _w[:, :_old] = _sd[_wk].to(_w.device, _w.dtype)
            _sd[_wk] = _w
            if rank == 0:
                print(f"[resume] grew in_channels {_old} -> {_new} (new slice zero-init, "
                      f"function-preserving)", flush=True)
        _missing, _unexpected = raw_model.load_state_dict(_sd, strict=False)
        if rank == 0 and (_missing or _unexpected):
            # move_head params are legitimately absent from a pre-Lever-#2 checkpoint
            print(f"[resume] missing={list(_missing)} unexpected={list(_unexpected)}", flush=True)
        if _unexpected:
            sys.exit(f"[resume] unexpected keys in checkpoint: {list(_unexpected)}")
        if any(not k.startswith("move_head.") for k in _missing):
            sys.exit(f"[resume] missing non-move_head keys: {list(_missing)}")
        if "opt" in ck:
            # A SHAPE change alone does NOT raise. When only `neg_channel` is enabled the parameter
            # COUNT is unchanged (in_conv1.weight merely goes 2->3 channels), so load_state_dict
            # succeeds and silently installs Adam moments of the OLD shape -- the first opt.step()
            # then dies on a non-broadcastable add, after the cube stacks have already loaded.
            # Detect shape drift explicitly instead of relying on the exception.
            _params = [p_ for gg in opt.param_groups for p_ in gg["params"]]
            _ssd = ck["opt"].get("state", {})
            _drift = any(
                torch.is_tensor(v) and v.dim() > 0 and tuple(v.shape) != tuple(_params[i].shape)
                for i in range(len(_params))
                for v in (_ssd.get(i, _ssd.get(str(i))) or {}).values())
            _count_changed = sum(len(gg["params"]) for gg in ck["opt"].get("param_groups", []))                 != len(_params)
            if _drift or _count_changed:
                load_optimizer_across_growth(opt, ck["opt"], rank)
            else:
                try:
                    opt.load_state_dict(ck["opt"])
                except ValueError:
                    load_optimizer_across_growth(opt, ck["opt"], rank)
            # load_state_dict restores the SAVED param_groups, which carry the weight_decay the
            # checkpoint was trained with. Without this re-assert, changing cfg.weight_decay for a
            # resumed run is silently a no-op -- the old value wins. (lr is exempt: lr_at() rewrites
            # pg["lr"] every step.) Found while raising wd 1e-4 -> 1e-2 to fight the measured
            # real-arm overfitting; the change would have done nothing.
            for pg in opt.param_groups:
                pg["weight_decay"] = cfg.weight_decay
        start_step = int(ck["step"]) + 1
        if args.extend:
            # Anchor WSD at the checkpoint's own step so the re-warm starts from where the run
            # actually ended, not from an arbitrary point on a recomputed cosine.
            cfg.extend_base_step = int(ck["step"])
            cfg.max_steps = cfg.extend_base_step + int(args.extend)
            if rank == 0:
                n = int(args.extend)
                d = int(cfg.extend_decay_frac * n)
                print(f"[extend] WSD from step {cfg.extend_base_step} for {n} steps -> "
                      f"max_steps={cfg.max_steps} | warm {cfg.extend_warm_steps} -> "
                      f"peak {cfg.extend_peak_lr:.1e} -> stable {n - d - cfg.extend_warm_steps} "
                      f"-> decay {d}", flush=True)
                cfg.save(os.path.join(args.work, "config.json"))   # so a later --resume matches
        if rank == 0:
            opt_msg = ("restored" if "opt" in ck
                       else "FRESH (pre-resume ckpt; AdamW moments rebuild in ~1k steps)")
            print(f"[resume] {ck_path} -> continuing at step {start_step} "
                  f"(optimizer {opt_msg})", flush=True)
        del ck

    stepper = TrainStepper(cfg, device)
    curr = Curriculum(cfg)
    # prefetch pipeline: only worthwhile (and only wired) when the stacks are CPU-resident
    loader = None
    if not getattr(cfg, "cubes_on_gpu", True):
        loader = AsyncPatchLoader(cache, cfg, device, cfg.seed, rank,
                                  depth=int(getattr(cfg, "prefetch_depth", 2)))
        if rank == 0:
            print(f"[load] AsyncPatchLoader depth={loader.depth} (CPU stacks)", flush=True)
    # offset the seed by start_step so a resumed run doesn't replay the same batches
    g = torch.Generator(device=device); g.manual_seed(cfg.seed + rank + start_step * 977)

    # inline validation: live merge-rate on held-out val cubes (never trained on)
    evaluator = None
    if cfg.eval_every and not args.smoke:
        try:
            evaluator = InlineEvaluator(cfg, args.work, device, rank, world)
        except Exception as e:                        # never let eval setup kill training
            if rank == 0:
                print(f"[val] inline eval disabled ({type(e).__name__}: {e})", flush=True)
    last_val = None
    # SOTA selection (FFN protocol, Januszewski 2018): keep the checkpoint with the
    # highest NERL among evals whose all-panel merge-rate <= eps. NERL has a coverage
    # floor so a collapsed/empty model can NEVER win this -> run-1's collapse is
    # structurally unselectable. Seeded from any existing ckpt_best (survives --resume).
    best_nerl = -1.0
    best_path = os.path.join(args.work, "ckpt_best.pt")
    if rank == 0 and os.path.exists(best_path):
        try:
            best_nerl = float(torch.load(best_path, map_location="cpu").get("sel_nerl", -1.0))
        except Exception:
            best_nerl = -1.0
    # SECOND best: lowest REAL merge among ckpts clearing the NERL coverage floor (FFN
    # "fewest-mergers-first" protocol). Coverage-safe (collapse has NERL~0 -> excluded).
    best_core = -1.0
    cbest_path = os.path.join(args.work, "ckpt_core_best.pt")
    if args.resume and os.path.exists(cbest_path):
        try:
            best_core = float(torch.load(cbest_path, map_location="cpu").get("sel_core", -1.0))
        except Exception:
            best_core = -1.0
    best_merge = 1e9
    mbest_path = os.path.join(args.work, "ckpt_merge_best.pt")
    if rank == 0 and os.path.exists(mbest_path):
        try:
            best_merge = float(torch.load(mbest_path, map_location="cpu").get("sel_merge", 1e9))
        except Exception:
            best_merge = 1e9

    losses, tstep = [], time.time()
    t_start = time.time()
    sps_ema = None
    for step in range(start_step, cfg.max_steps):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step, cfg)
        opt.zero_grad(set_to_none=True)
        st = curr.state(step)
        batch = None
        if loader is not None:
            for k in range(1, loader.depth + 1):        # keep `depth` future steps in flight
                if step + k < cfg.max_steps:
                    loader.schedule(step + k, curr.state(step + k))
            batch = loader.get(step, st)
        out = stepper.run(fwd_net, sync_module, cache, st, opt, g, batch=batch)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip)
        opt.step()
        losses.append(out["loss"])
        if rank == 0 and (step % args.log_every == 0 or step == cfg.max_steps - 1):
            dt = time.time() - tstep
            sps = args.log_every / max(dt, 1e-9)                       # optimizer steps/s
            sps_ema = sps if sps_ema is None else 0.7 * sps_ema + 0.3 * sps
            ex = sps * cfg.batch_per_gpu * world * st.T                # fov-steps/s
            eta = (cfg.max_steps - step) / max(sps_ema, 1e-9)
            avg_loss = float(np.mean(losses[-args.log_every:]))
            print(f"[{step:6d}] phase {st.phase} T={st.T} real={st.real_frac:.2f} "
                  f"loss={avg_loss:.4f} lr={lr_at(step,cfg):.2e} "
                  f"{ex:.0f} fov-steps/s eta {fmt_eta(eta)}", flush=True)
            write_progress(args.work, step, cfg.max_steps, st, avg_loss, lr_at(step, cfg),
                           ex, sps_ema, eta, time.time() - t_start, val=last_val)
            tstep = time.time()
        if rank == 0 and args.ckpt_every and step > 0 and step % args.ckpt_every == 0:
            save_ckpt(raw_model, cfg, step, args.work, opt=opt)
        # inline NERL + merge eval on the held-out panel (all ranks participate)
        if evaluator is not None and step > 0 and step % cfg.eval_every == 0:
            tv = time.time()
            stats = evaluator.run(raw_model, stepper.amp_dtype, step=step)
            took = time.time() - tv
            last_val = dict(step=step, **stats)
            if rank == 0:
                log_val(args.work, step, stats, took)
                # PRIMARY = NERL (higher better); merge shown as the constraint
                # SELECT ON REAL NERL, not all_nerl. all_nerl pools real and synth as
                # (Sum run^2)/(Sum perfect^2), and synth cubes carry 23-28 instances against real's
                # 8-9, so synth DOMINATES the denominator. With synth_nerl pinned near 0 for the
                # entire run, the primary selection metric was mostly measuring the arm that does
                # not work, and on a domain we never deploy to. Measured at ckpt_405000 on the
                # 26-cube panel: all_nerl would read ~0.25 while real_nerl is 0.4599.
                # The merge CONSTRAINT stays on all_merge (a merge is a merge wherever it happens).
                nerl_a, merge_a = stats.get("real_nerl"), stats.get("all_merge")
                real_m = stats.get("real_merge")
                star = ""
                # (1) NERL-best: max NERL among merge<=eps
                if (merge_a is not None and merge_a <= cfg.select_merge_eps
                        and nerl_a is not None and nerl_a > best_nerl):
                    best_nerl = nerl_a
                    save_ckpt(raw_model, cfg, step, args.work, best=True, opt=opt,
                              sel_nerl=nerl_a, sel_merge=merge_a)
                    star += "  <- NERL-BEST"
                # (2) merge-best: min REAL merge among ckpts clearing the NERL coverage floor
                if (nerl_a is not None and nerl_a >= cfg.select_nerl_floor
                        and real_m is not None and real_m < best_merge - 1e-9):
                    best_merge = real_m
                    save_ckpt(raw_model, cfg, step, args.work, merge_best=True, opt=opt,
                              sel_nerl=nerl_a, sel_merge=real_m)
                    star += "  <- MERGE-BEST"
                # (3) CORE-best: max label-free core_score among ckpts clearing the panel merge
                # constraint. The panel is the easy regime -- it read frac_1step 0.067 on run-2 while
                # the cores read 0.556 -- so neither rule above can see a model that does not trace on
                # real core material. core_score counts only objects that did not cross a wrap period,
                # so it penalises fragmentation (size^2) and merging (validity) at once. Additive, not
                # a gate: an absolute floor on it could not be justified a priori, and set wrong would
                # silently save nothing.
                core_s = stats.get("core_score")
                if (core_s is not None and core_s > best_core
                        and merge_a is not None and merge_a <= cfg.select_merge_eps):
                    best_core = core_s
                    save_ckpt(raw_model, cfg, step, args.work, core_best=True, opt=opt,
                              sel_nerl=nerl_a, sel_merge=merge_a, sel_core=core_s)
                    star += "  <- CORE-BEST"
                print(f"[val {step}] NERL all={nerl_a} (real={stats['real_nerl']} "
                      f"synth={stats['synth_nerl']}) | merge all={merge_a} "
                      f"(real={real_m} blind={stats['blind_merge']}) "
                      f"inst={stats['instances']} | best_nerl={best_nerl:.4f} "
                      f"best_realmerge={best_merge:.4f}{star} ({took:.0f}s)", flush=True)
                # MOVEMENT: printed separately because it is the only mechanism metric here. Every
                # number on the line above is computed on the final label volume, so a model that
                # never moves its FOV can post a clean merge rate while emitting one-window chips --
                # which is exactly what run-2 did. Watch core_1step: it is measured on unlabelled
                # held-out core cubes, the compact regime the labelled panel does not contain.
                print(f"[val {step}] movement: panel_1step={stats.get('frac_1step')} "
                      f"mean_steps={stats.get('mean_steps')} "
                      f"core_1step={stats.get('core_frac_1step')}", flush=True)
                print(f"[val {step}] CORE (unlabelled target regime): score={stats.get('core_score')} "
                      f"cov={stats.get('core_cov')} inst={stats.get('core_inst')} "
                      f"valid={stats.get('core_valid_frac')} | best_core={best_core:.5f}", flush=True)
            tstep = time.time()               # don't count eval time in throughput/ETA

    if rank == 0:
        save_ckpt(raw_model, cfg, cfg.max_steps, args.work, last=True, opt=opt)
        print(f"[done] {cfg.max_steps} steps, final loss "
              f"{np.mean(losses[-20:]):.4f}", flush=True)
    if ddp:
        dist.destroy_process_group()


def save_ckpt(model, cfg, step, work, last=False, opt=None, best=False, merge_best=False,
              core_best=False, sel_nerl=None, sel_merge=None, sel_core=None):
    from dataclasses import asdict
    sd = {k: v for k, v in model.state_dict().items()}
    payload = {"model": sd, "cfg": asdict(cfg), "step": step}
    if opt is not None:                       # enables exact --resume (moments + schedule)
        payload["opt"] = opt.state_dict()
    tagged = best or merge_best or core_best
    if tagged:                    # a selected checkpoint (NERL-best / merge-best / core-best)
        payload["sel_nerl"] = sel_nerl
        payload["sel_merge"] = sel_merge
        payload["sel_core"] = sel_core
        name = ("ckpt_best.pt" if best else
                "ckpt_merge_best.pt" if merge_best else "ckpt_core_best.pt")
    else:
        name = "ckpt_last.pt" if last else f"ckpt_{step}.pt"
    path = os.path.join(work, name)
    torch.save(payload, path)
    print(f"[ckpt] {path}" + (f" (NERL={sel_nerl} merge={sel_merge})" if tagged else ""),
          flush=True)


if __name__ == "__main__":
    main()
