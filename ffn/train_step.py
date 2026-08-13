"""The recurrent FFN training step (one batch, a T-step trajectory).

Faithful to google/ffn and Ging et al. 2019 (arXiv:1905.06236): each step is an
independent voxelwise-logistic supervised update -- "given image + current POM,
predict the same-instance mask" -- with the POM carried between steps as
*stop-gradient* data (not BPTT). This is the reference FFN training scheme; it is
more stable and far more memory-efficient than backpropagating through the whole
unroll, so we get a bigger effective batch and clean DDP.

A trajectory of T steps accumulates gradients (loss scaled 1/T), and the caller
runs one optimizer step per trajectory. Movement is teacher-forced (kept inside
the seed instance) during the early curriculum, then policy-driven.
"""
from __future__ import annotations

import contextlib
import torch
import torch.nn.functional as F

from . import movement as M
from .pom import (init_pom_canvas, batched_crop, pom_to_input, apply_update,
                  make_offset_grid, corrupt_canvas, init_neg_canvas, neg_to_input)
from .movement import training_move, face_in_instance
from .volumes import octahedral_augment, spatial_augment, intensity_augment


def balanced_bce(logits: torch.Tensor, target: torch.Tensor,
                 weight: torch.Tensor | None = None) -> torch.Tensor:
    """Per-FOV class-balanced BCE (collapse fix): in-object and background halves of
    each example weighted equally, then meaned over the batch. Makes the lazy
    predict-the-base-rate solution maximally expensive on thin-sheet FOVs.

    NORMALISATION FIX. The previous version multiplied `bce` by `weight` but divided the negative
    half by the UNWEIGHTED negative COUNT. That makes `w_other` scale the magnitude of the whole
    negative half instead of redistributing emphasis inside it: at w_other=4 with ~42% of a
    hard-negative FOV being other-instance voxels, the negative term was multiplied by ~2.6, so the
    intended 50:50 pos:neg balance silently became ~28:72. Every previous w_other experiment was
    therefore uninterpretable -- it measured a class-balance change, not a merge-asymmetry change.

    Dividing by the SUM OF WEIGHTS makes each half a proper weighted mean, so w_other moves emphasis
    between air and neighbouring-lamina voxels while the pos:neg balance stays exactly 50:50.
    Reduces to the old expression identically when weight is None or all-ones.
    """
    bce = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    w = torch.ones_like(bce) if weight is None else weight.to(bce.dtype)
    posm = (target > 0.5).float()
    bflat = (bce * w).flatten(1)
    wflat, pflat = w.flatten(1), posm.flatten(1)
    pos_mean = (bflat * pflat).sum(1) / (wflat * pflat).sum(1).clamp(min=1.0)
    neg_mean = (bflat * (1 - pflat)).sum(1) / (wflat * (1 - pflat)).sum(1).clamp(min=1.0)
    return 0.5 * (pos_mean + neg_mean).mean()


class TrainStepper:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.fov = cfg.fov
        self.patch = cfg.fov + 2 * cfg.walk_radius
        self.canvas_center = self.patch // 2
        self._coords = None          # lazy [P,P,P] index grids for pom_corrupt_p (built on demand)
        self._cpu_g = torch.Generator().manual_seed(int(getattr(cfg, "seed", 0)) + 8191)
        self.fgrid = make_offset_grid(cfg.fov, device)
        self.amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
        # cudagraph step-marking is only needed/available when compiling (max-autotune)
        self._mark_step = bool(cfg.compile) and hasattr(torch.compiler, "cudagraph_mark_step_begin")

    def run(self, fwd_net, sync_module, cache, sched, optimizer, g: torch.Generator,
            batch=None):
        """fwd_net: callable used for the forward (compiled and/or DDP-wrapped).
        sync_module: the DDP module whose no_sync() gates grad all-reduce (or None
        for single-GPU). Forward/backward always flow through fwd_net.
        batch: optional prefetched (img, inst, seed_inst) from AsyncPatchLoader; when
        None the historical sample+gather path runs inline."""
        cfg = self.cfg
        B = cfg.batch_per_gpu
        T = sched.T
        allowed = sched.allowed_bins
        teacher_force = sched.teacher_force
        hardneg_frac = sched.hardneg_frac

        if batch is not None:
            img, inst, seed_inst = batch                       # [B,1,P,P,P]
        else:
            idx = cache.sample_indices(B, allowed, hardneg_frac, g, std_only=sched.std_only,
                                       real_frac=getattr(sched, "real_frac", 0.0))
            img, inst, seed_inst = cache.get_patches(idx)      # [B,1,P,P,P]
        if cfg.augment:
            # ONE geometric transform per patch (held across the whole trajectory),
            # then label-invariant appearance jitter. Target is built below from the
            # augmented instance labels, so image and labels stay registered.
            if cfg.aug_geom:
                img, inst = spatial_augment(img, inst, g, cfg.aug_rot_deg,
                                            cfg.aug_scale, cfg.aug_shear, cfg.aug_p_geom)
            else:
                img, inst = octahedral_augment(img, inst, g)   # exact base only
            img = intensity_augment(
                img, g, gain=cfg.aug_gain, bias=cfg.aug_bias, gamma=cfg.aug_gamma,
                contrast=cfg.aug_contrast, noise=cfg.aug_noise,
                blur_sigma=cfg.aug_blur_sigma, biasfield=cfg.aug_biasfield,
                p_gain=cfg.aug_p_gain, p_bias=cfg.aug_p_bias, p_gamma=cfg.aug_p_gamma,
                p_contrast=cfg.aug_p_contrast, p_noise=cfg.aug_p_noise,
                p_blur=cfg.aug_p_blur, p_biasfield=cfg.aug_p_biasfield)
        # seed instance = center voxel of the (augmented) inst patch
        cc = self.canvas_center
        seed_inst = inst[:, 0, cc, cc, cc].clone()

        pom = init_pom_canvas(B, self.patch, cfg.seed_pad,
                              getattr(cfg, "pom_init", cfg.tgt_lo), cfg.tgt_hi,
                              cfg.logit_clamp, self.device)
        # RATCHET PARITY (cfg.pom_ratchet_train): historically the training unroll NEVER passed
        # `written`, so the network was optimised as if every POM decision were revisable and
        # then deployed under a freeze rule (measured inference-only: merge 0.071->0.028, blind
        # 0.044->0.000). When enabled, the same rule runs inside the trajectory.
        # NEGATIVE-EVIDENCE CANVAS (cfg.neg_channel, research/13 Lever #1): r = max over visits of
        # (1-p). Separates "visited and rejected" from "never visited", which pom_init == tgt_lo
        # currently makes indistinguishable. See pom.init_neg_canvas.
        neg = (init_neg_canvas(B, self.patch, self.device)
               if getattr(cfg, "neg_channel", False) else None)
        written = None
        freeze_logit = 0.0
        if getattr(cfg, "pom_ratchet_train", False) and getattr(cfg, "pom_ratchet", False):
            written = torch.zeros_like(pom, dtype=torch.bool)
            import math as _math
            fp = min(max(float(getattr(cfg, "pom_ratchet_freeze_p", 0.5)), 1e-4), 1 - 1e-4)
            freeze_logit = _math.log(fp / (1.0 - fp))
        offset = torch.zeros(B, 3, dtype=torch.long, device=self.device)
        if getattr(cfg, "train_offcentre", False):
            # Start the FOV at a random legal walk-lattice node instead of dead-centre. Otherwise
            # P(object at the FOV edge) ~ 0 in training but ~ 1 at inference, so the network is only
            # ever asked to re-predict a centred object, never to extend a mask into fresh territory.
            # INNER HALF only. Drawing from the full +-walk_radius//delta puts a large share of
            # examples exactly ON the clamp limit, where training_move cannot move them outward on
            # that axis at all -- teaching 'you cannot go that way' on real material. Starting
            # inside the inner half always leaves room to move in every direction.
            nsteps = max(1, cfg.walk_radius // (2 * cfg.delta))
            offset = (torch.randint(-nsteps, nsteps + 1, (B, 3), device=self.device)
                      * cfg.delta).long()
        # POM / RECURRENT-STATE CORRUPTION (cfg.pom_corrupt_p, research/11 #1). Decide ONCE per
        # trajectory: which examples get a defect, of which kind, injected before which step.
        # Injection is mid-unroll on purpose -- at t=0 the canvas is `lo` everywhere but the seed
        # disc, so a hole or a soften would be a no-op and only a leak would do anything. Letting t
        # range over the unroll also means the defect appears in a canvas the network itself built,
        # which is the state distribution that actually occurs at inference.
        corrupt_kinds = None
        corrupt_at = None
        cp = float(getattr(cfg, "pom_corrupt_p", 0.0))
        if cp > 0.0:
            kind_names = [k.strip() for k in
                          str(getattr(cfg, "pom_corrupt_kinds", "leak,hole,soften")).split(",")
                          if k.strip()]
            code = {"leak": 1, "hole": 2, "soften": 3}
            avail = torch.tensor([code[k] for k in kind_names if k in code],
                                 device=self.device, dtype=torch.int8)
            if avail.numel():
                # HOST-SIDE schedule. Generating these on the device and then testing them in
                # Python (`if (k_t != 0).any()`) forces a device->host sync on EVERY unroll step,
                # i.e. T syncs per trajectory, which stalls the CPU from queuing the next step's
                # kernels -- the same class of bug the logging scalars below were fixed for
                # (profiled at ~38% of CPU time). These are B-element decisions; make them on the
                # CPU so all control flow is host-side and the device is never queried.
                av = avail.tolist()
                sel_c = torch.rand(B, generator=self._cpu_g) < cp
                pick_c = torch.randint(0, len(av), (B,), generator=self._cpu_g)
                kinds_c = torch.tensor([av[i] for i in pick_c.tolist()], dtype=torch.int8)
                corrupt_kinds_cpu = torch.where(sel_c, kinds_c, torch.zeros_like(kinds_c))
                corrupt_at_cpu = torch.randint(0, max(1, T), (B,), generator=self._cpu_g)
                corrupt_kinds = corrupt_kinds_cpu.to(self.device)
                corrupt_at = corrupt_at_cpu.to(self.device)
                if self._coords is None:
                    ar = torch.arange(self.patch, device=self.device)
                    self._coords = torch.meshgrid(ar, ar, ar, indexing="ij")
        # visited lattice for the walk (see movement.training_move): without it the FOV
        # ping-pongs between two nodes and the POM is never extended.
        _n = max(1, cfg.walk_radius // cfg.delta)
        visited = torch.zeros(B, (2 * _n + 1) ** 3, dtype=torch.bool, device=self.device)
        visited.scatter_(1, M.lattice_index(offset, cfg.delta, cfg.walk_radius).unsqueeze(1), True)
        vidx = torch.arange(B, device=self.device)
        seed_abs = torch.full((B, 3), cc, dtype=torch.long, device=self.device)
        img_s = img[:, 0].contiguous()      # [B,P,P,P] for batched_crop
        inst_s = inst[:, 0].contiguous()
        inst_s_f = inst_s.float()           # cast the stack ONCE, not once per unroll step
        seed_inst_f = seed_inst.view(B, 1, 1, 1, 1).float()

        # Accumulate the (detached) logging scalars ON-GPU and sync ONCE per
        # trajectory. The previous per-micro-step float(loss)/float(mean) forced a
        # device->host cudaStreamSynchronize on every unroll step (T syncs each),
        # which blocked the CPU from queuing the next step's kernels -- profiled at
        # ~38% of CPU time. These scalars are LOGGING ONLY (backward already ran on
        # the real graph), so moving the sync out of the loop is training-neutral;
        # the printed loss avg differs only by fp32-vs-python-float accumulation
        # rounding (< 1e-6), never the gradients/update.
        total = torch.zeros((), device=self.device, dtype=torch.float32)
        frac_active = torch.zeros((), device=self.device, dtype=torch.float32)
        # aux-loss ramp: 0 for the first `move_head_warm` steps so the base objective is untouched
        # while movement is still driven by the deployed facemax gate (DAgger sequencing -- match the
        # policy first, then hand over).
        lam_mv = 0.0
        if getattr(cfg, "move_head", False):
            gs = int(getattr(sched, "global_step", 0) or 0)
            warm = max(1, int(getattr(cfg, "move_head_warm", 10000)))
            lam_mv = float(getattr(cfg, "move_head_lambda", 0.2)) * min(1.0, gs / warm)
        n_corrupt = {}
        for t in range(T):
            if corrupt_kinds is not None:
                # inject BEFORE this step's forward, so step t's own loss supplies the retraction
                # signal on the corrupted state. The gate is evaluated on the HOST copy.
                act_c = (corrupt_at_cpu == t) & (corrupt_kinds_cpu != 0)
                if bool(act_c.any()):
                    k_t = torch.where(corrupt_at == t, corrupt_kinds,
                                      torch.zeros_like(corrupt_kinds))
                    present = sorted({int(v) for v in
                                      corrupt_kinds_cpu[act_c].tolist()})
                    n_corrupt = corrupt_canvas(
                        pom, inst_s, seed_inst, kinds=k_t,
                        leak_frac=float(cfg.pom_corrupt_leak_frac),
                        hole_frac=float(cfg.pom_corrupt_hole_frac),
                        soften=float(cfg.pom_corrupt_soften),
                        tgt_lo=cfg.tgt_lo, tgt_hi=cfg.tgt_hi,
                        pom_init=float(getattr(cfg, "pom_init", cfg.tgt_lo)),
                        clamp=cfg.logit_clamp, coords=self._coords, gen=g, present=present)
            centers = offset + cc                                # [B,3] patch coords
            img_fov = batched_crop(img_s, vidx, centers, self.fov, self.fgrid)
            pom_fov = batched_crop(pom[:, 0], vidx, centers, self.fov, self.fgrid)
            chans = [img_fov, pom_to_input(pom_fov)]
            if neg is not None:
                chans.append(neg_to_input(batched_crop(neg[:, 0], vidx, centers,
                                                       self.fov, self.fgrid)))
            inp = torch.cat(chans, dim=1)
            if cfg.channels_last:
                inp = inp.to(memory_format=torch.channels_last_3d)

            # DDP: only sync grads on the last micro-step of the trajectory
            sync_ctx = contextlib.nullcontext()
            if sync_module is not None and t < T - 1 and hasattr(sync_module, "no_sync"):
                sync_ctx = sync_module.no_sync()

            with sync_ctx:
                # Defensive no-op now that cudagraph capture is hard-disabled at compile
                # setup (scripts/train.py); harmless if a future config re-enables graphs.
                if self._mark_step:
                    torch.compiler.cudagraph_mark_step_begin()
                with torch.autocast("cuda", dtype=self.amp_dtype):
                    out = fwd_net(inp)
                move_logits = None
                if isinstance(out, tuple):
                    logits, move_logits = out
                else:
                    logits = out
                inst_fov = batched_crop(inst_s_f, vidx, centers, self.fov, self.fgrid)
                target = torch.where(inst_fov == seed_inst_f, cfg.tgt_hi, cfg.tgt_lo)
                # MERGE-ASYMMETRIC WEIGHTS: an air voxel and a voxel of the NEIGHBOURING LAMINA
                # currently cost the same, which inverts the economics (a merge shifts the winding
                # number of every outer wrap; a split is interpolated downstream). w_other=1.0
                # reproduces the historical behaviour exactly.
                wmap = None
                w_blind = float(getattr(cfg, "w_blind", 1.0))
                if cfg.w_other != 1.0 or w_blind != 1.0:
                    other = (inst_fov > 0) & (inst_fov != seed_inst_f)
                    wmap = torch.where(other, float(cfg.w_other), 1.0)
                    if w_blind != 1.0:
                        # w_blind was a DEAD KNOB (declared in config, referenced nowhere).
                        # Per-voxel blind proxy mirroring data_prep.blind_contact_map's
                        # definition: an other-instance voxel with NO intensity gap at it --
                        # image (in [-1,1]) above gap_intensity_frac of the FOV's foreground
                        # mean. Bright other-lamina voxels are exactly the blind-contact
                        # material where merges are made.
                        fgm = inst_fov > 0
                        fg_mean = (img_fov * fgm).sum(dim=(1, 2, 3, 4), keepdim=True) / \
                            fgm.float().sum(dim=(1, 2, 3, 4), keepdim=True).clamp(min=1.0)
                        thr_b = -1.0 + (fg_mean + 1.0) * float(
                            getattr(cfg, "gap_intensity_frac", 0.5))
                        blind = other & (img_fov > thr_b)
                        wmap = torch.where(blind, wmap * w_blind, wmap)
                # Per-FOV CLASS-BALANCED BCE (collapse fix, run-1 post-mortem): plain BCE
                # let the ~85% background of a thin-sheet FOV dominate, so "predict low
                # everywhere" scored ~0.35 and single-pass confidence collapsed by 60k
                # steps. Same mechanism that fixed the affinity-arm collapse.
                loss = balanced_bce(logits, target, weight=wmap) / T
                if move_logits is not None:
                    # LEARNED MOVEMENT GATE (research/13 Lever #2). Target is free: the face slab
                    # lies entirely inside the FOV, so `inst_fov` already holds it -- this is the
                    # same quantity training_move computes for teacher forcing and discards.
                    # pos_weight < 1 penalises FALSE-OPEN (the merge-ward error) more than
                    # false-closed, which is the asymmetry the hand-set 0.9 cannot express.
                    tgt_mv = face_in_instance(inst_fov, seed_inst, cfg.fov, cfg.delta)
                    pw = torch.full_like(tgt_mv, float(getattr(cfg, "move_head_pos_weight", 1.0)))
                    mv_w = torch.where(tgt_mv > 0.5, pw, torch.ones_like(pw))
                    mv = F.binary_cross_entropy_with_logits(
                        move_logits.float(), tgt_mv, weight=mv_w)
                    loss = loss + (lam_mv / T) * mv
                loss.backward()

            total += loss.detach() * T
            frac_active += (target > 0.5).float().mean()

            # detached POM feedback (stop-gradient recurrence) + movement
            with torch.no_grad():
                apply_update(pom, centers, self.fov, logits.detach().float(),
                             cfg.logit_clamp, additive=cfg.additive_pom, offset_grid=self.fgrid,
                             written=written, freeze_logit=freeze_logit, neg=neg)
                offset = training_move(
                    offset, pom, self.canvas_center, cfg.delta, cfg.walk_radius,
                    cfg.move_threshold, inst_s.long(), vidx, seed_abs, seed_inst,
                    teacher_force, gate=getattr(cfg, "move_gate", "center"),
                    fov=cfg.fov, visited=visited, gen=g)

        # single device->host sync per trajectory (both scalars in one transfer)
        loss_v, frac_v = (torch.stack([total, frac_active]) / max(T, 1)).tolist()
        return {"loss": loss_v, "frac_active": frac_v}
