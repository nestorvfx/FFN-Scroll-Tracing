"""Production flood fill: the FFN fill with the committed-neighbour barrier.

Forked from ffn.inference.flood_fill_object DELIBERATELY (that path serves live training
evals and must stay byte-stable). Differences, each load-bearing:

  BARRIER   voxels committed to other objects are pinned in the POM canvas at
            logit(tgt_lo) -- "definitely not me" -- before the fill starts and after
            every network write. The classifier SEES prior commitments in its POM input
            channel (the training-eval path only vetoes the movement queue, so the net
            is blind to them). Inhibitory only: a barrier can suppress growth, never
            cause it -- the merge-asymmetric direction.
  STATS     per-fill step count and a truncation flag (hit fill_max_steps), because a
            truncated fill is a silent split factory and production must surface it.
  TTA       optional rot90 logit averaging wrapped around predict_fn (config, off by
            default -- v1 baseline is single-pass).

Semantics otherwise identical: same seed canvas, same ratchet scopes, same face-max
movement gate, same iterated FOV refinement, same calibrated commit.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch

import sys
sys.path.insert(0, "/root/surface_detection/FFN")
from ffn.pom import (logit, make_offset_grid, batched_crop, pom_to_input,  # noqa: E402
                     neg_to_input)
from ffn.commit import commit_mask                                       # noqa: E402
from ffn.movement import FACE_DIRS, face_offsets                         # noqa: E402


def tta_wrap(predict_fn, enabled: bool):
    """Average logits over the 4 exact rotations in the (H,W) plane. Exact (no
    interpolation), logit-space (label voting is banned), single checkpoint."""
    if not enabled:
        return predict_fn

    def fn(inp):
        acc = predict_fn(inp)
        for k in (1, 2, 3):
            r = predict_fn(torch.rot90(inp, k, dims=(3, 4)))
            acc = acc + torch.rot90(r, -k, dims=(3, 4))
        return acc / 4.0
    return fn


def flood_fill(predict_fn, image_gpu: torch.Tensor, seed, cfg,
               K: int = 64, claimed: np.ndarray | None = None,
               barrier: torch.Tensor | None = None,
               pom_buf: torch.Tensor | None = None,
               visited_buf: np.ndarray | None = None, visited_gen: int = 0,
               stats: dict | None = None) -> np.ndarray:
    """Grow one object from `seed` on a block canvas. Returns committed bool mask.

    `claimed`: int label volume (CPU) -- movement veto, as in training-eval.
    `barrier`: bool tensor (device) -- committed-to-OTHERS voxels, pinned in the POM.
    """
    device = image_gpu.device
    Z, Y, X = image_gpu.shape
    fov = cfg.fov
    h = fov // 2
    delta = cfg.delta
    lo = logit(getattr(cfg, "pom_init", cfg.tgt_lo))
    hi = logit(cfg.tgt_hi)
    bar_val = logit(cfg.tgt_lo)          # the trained "definitely background" value
    fgrid = make_offset_grid(fov, device)
    gate = getattr(cfg, "inf_move_gate", "center")
    face_off = face_offsets(fov, delta, device) if gate == "facemax" else None

    pom = pom_buf if pom_buf is not None else \
        torch.empty((Z, Y, X), dtype=torch.float32, device=device)
    pom.fill_(lo)
    z, y, x = seed
    r = cfg.seed_pad
    zz, yy, xx = np.mgrid[max(0, z - r):min(Z, z + r + 1),
                          max(0, y - r):min(Y, y + r + 1),
                          max(0, x - r):min(X, x + r + 1)]
    m = (zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2 <= r * r
    pom[torch.as_tensor(zz[m], device=device),
        torch.as_tensor(yy[m], device=device),
        torch.as_tensor(xx[m], device=device)] = hi
    pom.clamp_(-cfg.logit_clamp, cfg.logit_clamp)
    if barrier is not None:
        # barrier wins everywhere, including over the seed sphere's edge: those voxels
        # already belong to someone else and this fill must treat them as not-me.
        pom.masked_fill_(barrier, bar_val)

    if visited_buf is not None:
        def seen(nc):
            return visited_buf[nc] == visited_gen

        def mark(nc):
            visited_buf[nc] = visited_gen
    else:
        visited = np.zeros((Z, Y, X), dtype=bool)

        def seen(nc):
            return visited[nc]

        def mark(nc):
            visited[nc] = True

    dirs = FACE_DIRS.numpy() * delta

    def inb(c):
        return (h <= c[0] < Z - h) and (h <= c[1] < Y - h) and (h <= c[2] < X - h)

    import math as _math
    _fp = min(max(float(getattr(cfg, "pom_ratchet_freeze_p", 0.5)), 1e-4), 1 - 1e-4)
    _flz = _math.log(_fp / (1.0 - _fp))
    wrote = (torch.zeros((Z, Y, X), dtype=torch.bool, device=device)
             if getattr(cfg, "pom_ratchet", False) else None)
    # NEGATIVE-EVIDENCE CANVAS -- must match training (research/13 Lever #1)
    negc = (torch.zeros((Z, Y, X), dtype=torch.float32, device=device)
            if getattr(cfg, "neg_channel", False) else None)
    q = deque()
    if inb(seed):
        q.append(tuple(seed))
        mark((int(seed[0]), int(seed[1]), int(seed[2])))
    steps = 0
    fov_iters = getattr(cfg, "inf_fov_iters", 1)
    fov_tol = getattr(cfg, "inf_fov_tol", 0.05)
    per_position = getattr(cfg, "pom_ratchet_scope", "step") == "position"
    while q and steps < cfg.fill_max_steps:
        batch = [q.popleft() for _ in range(min(K, len(q)))]
        steps += len(batch)
        centers = torch.tensor(batch, dtype=torch.long, device=device)
        zeros = torch.zeros(len(batch), dtype=torch.long, device=device)
        img_fov = batched_crop(image_gpu.unsqueeze(0), zeros, centers, fov, fgrid)
        prev = None
        slices = []
        for c in batch:
            z0, y0, x0 = c[0] - h, c[1] - h, c[2] - h
            slices.append((slice(z0, z0 + fov), slice(y0, y0 + fov), slice(x0, x0 + fov)))
        snap = ([(pom[sl].clone(), wrote[sl].clone()) for sl in slices]
                if (wrote is not None and per_position) else None)
        for _ in range(max(1, fov_iters)):
            pom_fov = batched_crop(pom.unsqueeze(0), zeros, centers, fov, fgrid)
            chans = [img_fov, pom_to_input(pom_fov)]
            if negc is not None:
                chans.append(neg_to_input(batched_crop(negc.unsqueeze(0), zeros, centers,
                                                       fov, fgrid)))
            inp = torch.cat(chans, dim=1)
            _o = predict_fn(inp)
            logits = (_o[0] if isinstance(_o, tuple) else _o).float().clamp(
                -cfg.logit_clamp, cfg.logit_clamp)
            if negc is not None:
                # post-freeze, matching apply_update (see ffn/inference.py)
                for i, sl in enumerate(slices):
                    negc[sl] = torch.maximum(negc[sl], 1.0 - torch.sigmoid(pom[sl]))
            for i, sl in enumerate(slices):
                new = logits[i, 0]
                if wrote is None:
                    pom[sl] = new
                elif per_position:
                    prior0, w0 = snap[i]
                    frozen = w0 & (prior0 < _flz) & (new > prior0)
                    pom[sl] = torch.where(frozen, prior0, new)
                else:
                    prior = pom[sl]
                    frozen = wrote[sl] & (prior < _flz) & (new > prior)
                    pom[sl] = torch.where(frozen, prior, new)
                    wrote[sl] = True
                if barrier is not None:      # re-pin: the barrier out-ranks the net
                    b = barrier[sl]
                    pom[sl] = torch.where(b, torch.as_tensor(bar_val, device=device),
                                          pom[sl])
            cur = torch.sigmoid(logits)
            if prev is not None and (cur - prev).abs().max().item() < fov_tol:
                break
            prev = cur
        if wrote is not None and per_position:
            for sl in slices:
                wrote[sl] = True
        dirs_t = torch.as_tensor(dirs, dtype=torch.long, device=device)
        cand = centers.unsqueeze(1) + dirs_t.unsqueeze(0)
        if face_off is not None:
            pc = centers[:, None, None, :] + face_off[None]
            probs = torch.sigmoid(pom[pc[..., 0].clamp(0, Z - 1),
                                      pc[..., 1].clamp(0, Y - 1),
                                      pc[..., 2].clamp(0, X - 1)]).amax(dim=2)
        else:
            probs = torch.sigmoid(pom[cand[..., 0].clamp(0, Z - 1),
                                      cand[..., 1].clamp(0, Y - 1),
                                      cand[..., 2].clamp(0, X - 1)])
        inb_t = ((cand[..., 0] >= h) & (cand[..., 0] < Z - h) &
                 (cand[..., 1] >= h) & (cand[..., 1] < Y - h) &
                 (cand[..., 2] >= h) & (cand[..., 2] < X - h))
        ok = (inb_t & (probs >= cfg.inf_move_threshold)).cpu().numpy()
        cand_np = cand.cpu().numpy()
        for i in range(len(batch)):
            for d6 in np.nonzero(ok[i])[0]:
                nc = tuple(cand_np[i, d6])
                if seen(nc):
                    continue
                if claimed is not None and claimed[nc]:
                    continue
                mark(nc)
                q.append(nc)
    if stats is not None:
        stats["steps"] = steps
        stats["truncated"] = bool(q) and steps >= cfg.fill_max_steps
    return commit_mask(pom, cfg, seed)
