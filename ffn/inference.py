"""Inference: single-object flood-fill, oversegmentation-consensus, FFN
agglomeration, and block-wise dual-GPU slab segmentation (DESIGN.md 6.7, 6.4).

The flood-fill logic is decoupled from the trained network via a `predict_fn`
callable (image_fov, pom_fov -> logit_fov), so every routine is unit-testable
with a ground-truth "oracle" predictor and identically drives the real model.

Efficiency: one object grows at a time (a single reused block POM canvas), but
its frontier is advanced in *batches* of up to K FOVs per forward, keeping the
GPU fed. Overlapping writes within a batch are applied sequentially (cheap) to
stay deterministic; only the expensive forward is batched.
"""
from __future__ import annotations

import numpy as np
import torch
from collections import deque

from .pom import logit, make_offset_grid, batched_crop, pom_to_input, neg_to_input
from .commit import commit_mask
from .movement import FACE_DIRS, face_offsets


def _seed_canvas(shape, seed, seed_pad, lo, hi, clamp, device):
    canvas = torch.full(shape, lo, dtype=torch.float32, device=device)
    z, y, x = seed
    r = seed_pad
    Z, Y, X = shape
    zz, yy, xx = np.mgrid[max(0, z - r):min(Z, z + r + 1),
                          max(0, y - r):min(Y, y + r + 1),
                          max(0, x - r):min(X, x + r + 1)]
    d2 = (zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2
    m = d2 <= r * r
    canvas[torch.as_tensor(zz[m], device=device),
           torch.as_tensor(yy[m], device=device),
           torch.as_tensor(xx[m], device=device)] = hi
    return canvas.clamp_(-clamp, clamp)


def flood_fill_object(predict_fn, image_gpu: torch.Tensor, seed, cfg,
                      K: int = 64, claimed: np.ndarray | None = None,
                      pom_buf: torch.Tensor | None = None,
                      visited_buf: np.ndarray | None = None,
                      visited_gen: int = 0,
                      stats: dict | None = None,
                      return_levels: bool = False):
    """Grow one object from `seed`. image_gpu: [Z,Y,X] normalized float on device.
    Returns a bool mask [Z,Y,X] of the grown object (POM prob >= cfg.commit_threshold,
    optionally restricted to the seed's connected component -- see ffn/commit.py).
    `pom_buf`: optional reusable canvas (avoids a full-volume alloc per seed).
    `claimed`: a full-volume array; a voxel is off-limits where `claimed[v]` is
    truthy (pass the int label volume directly -- nonzero == claimed).
    `visited_buf`/`visited_gen`: optional reusable int generation-counter buffer --
    a voxel counts as visited iff `visited_buf[v] == visited_gen`, so the caller
    increments the generation per fill and NEVER reallocates/clears a per-seed
    full-volume `visited` mask. On a ~200M-voxel slab this removes thousands of
    200MB allocations; semantics are identical to the per-seed boolean mask."""
    device = image_gpu.device
    Z, Y, X = image_gpu.shape
    fov = cfg.fov
    h = fov // 2
    delta = cfg.delta
    # `lo` is the canvas value for NOT-YET-VISITED voxels (the paper's pad_value), which is a
    # different quantity from the BCE background target tgt_lo; cfg.pom_init separates them.
    lo, hi = logit(getattr(cfg, "pom_init", cfg.tgt_lo)), logit(cfg.tgt_hi)
    fgrid = make_offset_grid(fov, device)
    gate = getattr(cfg, "inf_move_gate", "center")
    face_off = face_offsets(fov, delta, device) if gate == "facemax" else None
    move_lg = None                      # last forward's learned gate logits (gate == "learned")

    if pom_buf is not None:
        pom = pom_buf
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
    else:
        pom = _seed_canvas((Z, Y, X), seed, cfg.seed_pad, lo, hi, cfg.logit_clamp, device)
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

    wrote = (torch.zeros((Z, Y, X), dtype=torch.bool, device=device)
             if getattr(cfg, "pom_ratchet", False) else None)
    # NEGATIVE-EVIDENCE CANVAS at decode -- must exist wherever it exists in training, or the
    # network is fed a channel distribution it never saw (research/13 Lever #1).
    negc = (torch.zeros((Z, Y, X), dtype=torch.float32, device=device)
            if getattr(cfg, "neg_channel", False) else None)
    # freeze threshold in LOGIT space (cfg gives probability; 0.5 == the paper's coin-flip rule)
    _flz = logit(min(max(float(getattr(cfg, "pom_ratchet_freeze_p", 0.5)), 1e-4), 1 - 1e-4))
    q = deque()
    if inb(seed):
        q.append(tuple(seed)); mark((int(seed[0]), int(seed[1]), int(seed[2])))
    steps = 0
    fov_iters = getattr(cfg, "inf_fov_iters", 1)
    fov_tol = getattr(cfg, "inf_fov_tol", 0.05)
    while q and steps < cfg.fill_max_steps:
        batch = [q.popleft() for _ in range(min(K, len(q)))]
        steps += len(batch)
        centers = torch.tensor(batch, dtype=torch.long, device=device)
        zeros = torch.zeros(len(batch), dtype=torch.long, device=device)
        img_fov = batched_crop(image_gpu.unsqueeze(0), zeros, centers, fov, fgrid)
        # train/infer parity: re-forward this batch with its own updated POM until the
        # prediction converges (training lets a stationary FOV build confidence; a
        # single pass starves the movement gate and under-grows objects)
        prev = None
        slices = []
        for c in batch:
            z0, y0, x0 = c[0] - h, c[1] - h, c[2] - h
            slices.append((slice(z0, z0 + fov), slice(y0, y0 + fov), slice(x0, x0 + fov)))
        # SCOPE OF THE RATCHET: the paper freezes voxels "previously updated by the FFN", i.e. by an
        # EARLIER FOV POSITION -- not by an earlier refinement iteration at THIS position. Snapshot the
        # canvas and the written-mask on entry to this FOV and freeze against those, so the iterative
        # re-forward (inf_fov_iters, which lets a stationary FOV build confidence) still works.
        per_position = getattr(cfg, "pom_ratchet_scope", "step") == "position"
        snap = ([(pom[sl].clone(), wrote[sl].clone()) for sl in slices]
                if (wrote is not None and per_position) else None)
        for _ in range(max(1, fov_iters)):
            pom_fov = batched_crop(pom.unsqueeze(0), zeros, centers, fov, fgrid)
            chans = [img_fov, pom_to_input(pom_fov)]
            if negc is not None:
                chans.append(neg_to_input(batched_crop(negc.unsqueeze(0), zeros, centers,
                                                       fov, fgrid)))
            inp = torch.cat(chans, dim=1)
            _out = predict_fn(inp)
            if isinstance(_out, tuple):
                _out, move_lg = _out                    # learned movement gate (Lever #2)
            logits = _out.float()                       # [b,1,fov,fov,fov]
            logits = logits.clamp(-cfg.logit_clamp, cfg.logit_clamp)
            # sequential (deterministic) writes
            for i, sl in enumerate(slices):
                new = logits[i, 0]
                if wrote is None:
                    pom[sl] = new
                elif per_position:
                    prior0, w0 = snap[i]
                    frozen = w0 & (prior0 < _flz) & (new > prior0)
                    pom[sl] = torch.where(frozen, prior0, new)
                else:                       # "step": freeze against the LIVE canvas
                    prior = pom[sl]
                    frozen = wrote[sl] & (prior < _flz) & (new > prior)
                    pom[sl] = torch.where(frozen, prior, new)
                    wrote[sl] = True
            if negc is not None:
                # TRAIN/INFER PARITY: `apply_update` derives r from the value actually WRITTEN to
                # the canvas (post-ratchet `new`), not from the raw logits. With pom_ratchet on, a
                # frozen voxel keeps its prior, so reading `logits` here would feed the network a
                # different r than training ever produced -- in the very channel under test.
                for i, sl in enumerate(slices):
                    negc[sl] = torch.maximum(negc[sl], 1.0 - torch.sigmoid(pom[sl]))
            cur = torch.sigmoid(logits)
            if prev is not None and (cur - prev).abs().max().item() < fov_tol:
                break
            prev = cur
        if wrote is not None and per_position:   # position done refining -> mark it written
            for sl in slices:
                wrote[sl] = True
        # movement: expand frontier -- fully vectorized, ONE device->host sync per batch
        # (the per-neighbor .item() version cost thousands of GPU syncs per cube)
        dirs_t = torch.as_tensor(dirs, dtype=torch.long, device=device)      # [6,3]
        cand = centers.unsqueeze(1) + dirs_t.unsqueeze(0)                    # [b,6,3]
        cz = cand[..., 0].clamp(0, Z - 1)
        cy = cand[..., 1].clamp(0, Y - 1)
        cx = cand[..., 2].clamp(0, X - 1)
        if gate == "learned" and move_lg is not None:
            # LEARNED GATE: one pooled, calibratable decision per direction from the forward that
            # just ran at THIS fov -- instead of a max over 169 canvas voxels, most of which were
            # written by OTHER fov positions. Strictly cheaper (no gather) and it is the estimator
            # the 0.41%-tail arithmetic says the face-max cannot be.
            probs = torch.sigmoid(move_lg.float())                           # [b,6]
            thr = float(getattr(cfg, "move_threshold_learned", 0.5))
        elif face_off is not None:
            # FACE-MAX gate (google-ffn / Januszewski et al.): max over the (2*delta+1)^2
            # cuboid face at +/-delta (see movement.face_offsets -- NOT fov x fov, which would
            # read the neighbouring lamina), not the single candidate-centre voxel. Must match
            # training.
            pc = centers[:, None, None, :] + face_off[None]                  # [b,6,P,3]
            probs = torch.sigmoid(pom[pc[..., 0].clamp(0, Z - 1),
                                      pc[..., 1].clamp(0, Y - 1),
                                      pc[..., 2].clamp(0, X - 1)]).amax(dim=2)
            thr = cfg.inf_move_threshold
        else:
            probs = torch.sigmoid(pom[cz, cy, cx])                           # [b,6]
            thr = cfg.inf_move_threshold
        inb_t = ((cand[..., 0] >= h) & (cand[..., 0] < Z - h) &
                 (cand[..., 1] >= h) & (cand[..., 1] < Y - h) &
                 (cand[..., 2] >= h) & (cand[..., 2] < X - h))
        ok = (inb_t & (probs >= thr)).cpu().numpy()                          # sync
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
    return commit_mask(pom, cfg, seed, return_levels=return_levels)


def screen_seeds(predict_fn, image_gpu, seeds, cfg, batch: int = 256) -> np.ndarray:
    """Batched pre-screen: for every seed, run the iterated FOV update on an isolated
    per-seed mini-canvas and mark it viable if its fill could commit anything.

    EXACTNESS: a seed is non-viable only if, after the same iterated update the serial
    fill would perform, (a) no face candidate reaches the movement threshold (with no
    claims, movement can only be rarer in the serial pass) and (b) the grown voxels in
    its own FOV are below the commit threshold (serial `take` is a subset of this).
    Non-viable => the serial fill would commit nothing => skipping is exact. Viable
    seeds still run the full serial fill. Turns thousands of one-by-one dead fills
    into a few hundred batched forwards."""
    device = image_gpu.device
    fov = cfg.fov
    h = fov // 2
    lo, hi = logit(getattr(cfg, "pom_init", cfg.tgt_lo)), logit(cfg.tgt_hi)
    fgrid = make_offset_grid(fov, device)
    n = len(seeds)
    viable = np.zeros(n, dtype=bool)
    commit_thr = cfg.min_instance_size // 4
    fov_iters = max(1, getattr(cfg, "inf_fov_iters", 1))
    fov_tol = getattr(cfg, "inf_fov_tol", 0.05)
    face_off = (FACE_DIRS * cfg.delta + h).to(device)               # [6,3] FOV coords
    for s0 in range(0, n, batch):
        idx = np.arange(s0, min(n, s0 + batch))
        centers = torch.as_tensor(np.asarray(seeds)[idx], dtype=torch.long, device=device)
        zeros = torch.zeros(len(idx), dtype=torch.long, device=device)
        img_fov = batched_crop(image_gpu.unsqueeze(0), zeros, centers, fov, fgrid)
        pom = torch.full((len(idx), 1, fov, fov, fov), lo, device=device)
        r = cfg.seed_pad
        pom[:, :, h - r:h + r + 1, h - r:h + r + 1, h - r:h + r + 1] = hi
        # NEGATIVE-EVIDENCE CHANNEL on the isolated mini-canvas. This site was MISSED when the
        # channel was added and it crashed the first validation of the L1 arm ("expected input to
        # have 3 channels, but got 2"): the model is 3-channel, every other input builder was
        # updated, this one was not. Starts at 0 (= nothing visited yet, which is true for a fresh
        # per-seed canvas) and accumulates max(1-p) exactly as apply_update does.
        neg = torch.zeros_like(pom) if getattr(cfg, "neg_channel", False) else None
        prev = None
        for _ in range(fov_iters):
            chans = [img_fov, pom_to_input(pom)]
            if neg is not None:
                chans.append(neg_to_input(neg))
            _o = predict_fn(torch.cat(chans, 1))
            logits = (_o[0] if isinstance(_o, tuple) else _o).float()
            pom = logits.clamp(-cfg.logit_clamp, cfg.logit_clamp)
            if neg is not None:
                neg = torch.maximum(neg, 1.0 - torch.sigmoid(pom))
            cur = torch.sigmoid(pom)
            if prev is not None and (cur - prev).abs().max().item() < fov_tol:
                break
            prev = cur
        p = torch.sigmoid(pom[:, 0])                                # [b,fov,fov,fov]
        if getattr(cfg, "inf_move_gate", "center") == "facemax":
            fp = (face_offsets(fov, cfg.delta, device) + h).clamp(0, fov - 1)   # [6,P,3]
            can_move = (p[:, fp[..., 0], fp[..., 1], fp[..., 2]].amax(dim=2)
                        >= cfg.inf_move_threshold).any(dim=1)
        else:
            can_move = (p[:, face_off[:, 0], face_off[:, 1], face_off[:, 2]]
                        >= cfg.inf_move_threshold).any(dim=1)
        # SAME threshold as the serial fill, or the exactness argument above breaks: a seed is
        # only safe to skip if the serial fill would commit nothing, and "commit" is defined by
        # cfg.commit_threshold. (Raising it can only mark MORE seeds non-viable, which stays
        # exact -- the serial `take` is a subset of this isolated grow.)
        grown = (p >= cfg.commit_threshold).flatten(1).sum(1) >= commit_thr
        viable[idx] = (can_move | grown).cpu().numpy()
    return viable


def segment_block(predict_fn, image_gpu, seeds, cfg, K: int = 64,
                  reverse: bool = False, viable: np.ndarray | None = None,
                  step_log: list | None = None,
                  levels_out: np.ndarray | None = None) -> np.ndarray:
    # levels_out: optional uint8 [Z,Y,X] buffer; when given, each committed voxel receives its
    # quantized POM level (1..16 over [commit_threshold, 0.99]) -- the retained per-object
    # hierarchy (nested by construction; see commit.commit_mask).
    """Serial multi-seed flood-fill over a block. seeds: [M,3] descending-EDT
    order. Returns int32 label volume (0=bg). Dead seeds are skipped via an
    exactness-preserving batched pre-screen; the POM canvas is reused across fills.

    `viable`: optional precomputed per-seed viability (aligned to `seeds`). Seed
    viability is order-independent and depends only on (image, model), so the
    caller can compute it once and reuse it across the forward/reverse/seed-set
    decodes instead of re-running the pre-screen (its dominant fixed cost) each time.
    """
    Z, Y, X = image_gpu.shape
    committed = np.zeros((Z, Y, X), dtype=np.int32)
    order = list(range(len(seeds)))
    if reverse:
        order = order[::-1]
    if viable is None:
        viable = screen_seeds(predict_fn, image_gpu, seeds, cfg) if len(seeds) else \
            np.zeros(0, bool)
    rr = cfg.seed_reject_radius
    next_label = 1
    pom_buf = torch.empty((Z, Y, X), dtype=torch.float32, device=image_gpu.device)
    # reused generation-counter `visited` buffer: avoids a full-volume alloc per
    # fill. `committed` (int) is passed as `claimed` directly (nonzero == claimed),
    # avoiding a second per-fill `committed > 0` boolean allocation.
    visited_buf = np.zeros((Z, Y, X), dtype=np.int32)
    gen = 0                       # unique per fill ATTEMPT (buffer starts at 0)
    for si in order:
        if not viable[si]:
            continue
        z, y, x = [int(v) for v in seeds[si]]
        if committed[z, y, x] != 0:
            continue
        z0, z1 = max(0, z - rr), min(Z, z + rr + 1)
        y0, y1 = max(0, y - rr), min(Y, y + rr + 1)
        x0, x1 = max(0, x - rr), min(X, x + rr + 1)
        if committed[z0:z1, y0:y1, x0:x1].any():
            continue
        gen += 1
        st = {}
        out = flood_fill_object(predict_fn, image_gpu, (z, y, x), cfg, K=K,
                                claimed=committed, pom_buf=pom_buf,
                                visited_buf=visited_buf, visited_gen=gen, stats=st,
                                return_levels=levels_out is not None)
        mask, levels = out if levels_out is not None else (out, None)
        take = mask & (committed == 0)
        if take.sum() < cfg.min_instance_size // 4:
            continue
        # a fill that never moved is a single stationary window, not a traced object
        if st.get("steps", 1) < getattr(cfg, "min_fov_steps", 1):
            continue
        committed[take] = next_label
        if levels_out is not None:
            levels_out[take] = levels[take]
        next_label += 1
        if step_log is not None:
            step_log.append(int(st.get("steps", 1)))
    return committed


# ----------------------------------------------------------------------------
# Oversegmentation-consensus (DESIGN.md 6.7 step 2)
# ----------------------------------------------------------------------------
def _label_by_value(key: np.ndarray) -> np.ndarray:
    """Connected components of a *labeled* integer volume in ONE pass: two voxels
    join iff they are 6-neighbors AND carry the same nonzero key. 0 = background.

    Replaces a per-unique-key `ndi.label(key==k)` loop (O(N * #keys), the dominant
    consensus cost: thousands of full-volume passes on an over-split cube) with a
    single multi-label connected-components pass (cc3d, ndi fallback). Identical
    semantics -- same-key connected pieces get distinct ids, different keys never
    merge -- so the merge-safety guarantee of consensus is preserved exactly."""
    try:
        import cc3d
        # cc3d on an int (labeled) input performs multi-label CC: it splits by both
        # value and connectivity, exactly matching the old key-by-key ndi.label.
        return cc3d.connected_components(key, connectivity=6).astype(np.int64)
    except Exception:
        pass
    try:
        from skimage.measure import label as sklabel
        return sklabel(key, background=0, connectivity=1).astype(np.int64)
    except Exception:
        # last-resort exact fallback (slow): the original per-key loop
        from scipy import ndimage as ndi
        out = np.zeros(key.shape, np.int64); nxt = 1
        for k in np.unique(key):
            if k == 0:
                continue
            comp, n = ndi.label(key == k)
            for c in range(1, n + 1):
                out[comp == c] = nxt; nxt += 1
        return out


def consensus(segmentations, min_size: int = 50) -> np.ndarray:
    """Intersect multiple label volumes: two voxels share a consensus label only
    if they co-segment (same nonzero label) in *every* run. This over-splits but
    can never merge two objects that any single run kept apart."""
    segs = [s.astype(np.int64) for s in segmentations]
    shape = segs[0].shape
    fg = np.ones(shape, dtype=bool)
    for s in segs:
        fg &= (s > 0)
    # combined key = tuple of per-run labels, encoded
    key = np.zeros(shape, dtype=np.int64)
    for s in segs:
        key = key * (s.max() + 1) + s
    key[~fg] = 0
    # single-pass connected components of the labeled key volume (splits
    # disconnected same-key pieces by connectivity)
    comp = _label_by_value(key)
    # size-filter + contiguous relabel, fully vectorized (no per-component pass)
    ids, counts = np.unique(comp, return_counts=True)
    keep = counts >= min_size
    keep &= (ids != 0)
    kept_ids = ids[keep]
    if kept_ids.size == 0:
        return np.zeros(shape, dtype=np.int32)
    lut = np.zeros(int(comp.max()) + 1, dtype=np.int32)
    lut[kept_ids] = np.arange(1, kept_ids.size + 1, dtype=np.int32)
    return lut[comp]


# ----------------------------------------------------------------------------
# FFN agglomeration (DESIGN.md 6.7 step 3)
# ----------------------------------------------------------------------------
def _l1_offsets(radius: int):
    """Undirected nonzero integer offsets with L1 norm <= radius (first nonzero > 0)."""
    offs = []
    r = int(radius)
    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dz) + abs(dy) + abs(dx) == 0 or abs(dz) + abs(dy) + abs(dx) > r:
                    continue
                if (dz, dy, dx) > (0, 0, 0):
                    offs.append((dz, dy, dx))
    return offs


def _shift_slices(d: int, n: int):
    if d > 0:
        return slice(0, n - d), slice(d, n)
    if d < 0:
        return slice(-d, n), slice(0, n + d)
    return slice(0, n), slice(0, n)


def candidate_pairs(labels: np.ndarray, radius: int):
    """Pairs of labels whose voxels lie within L1 distance `radius` (== `radius` iterations of
    6-connectivity dilation, the historical semantics), with a decision-point voxel on their
    interface. SINGLE-PASS offset scan: the per-label `binary_dilation` loop was
    O(#labels x volume) -- ~190 full-volume dilations per cube -- and was the hard blocker on
    any over-segment-then-agglomerate design. ~2r^3 shifted comparisons replace it."""
    labels = np.asarray(labels)
    Z, Y, X = labels.shape
    P1 = int(labels.max()) + 1
    if P1 <= 1:
        return {}
    touch = {}                          # (a,b) -> list of up to 64 interface voxel coords (b side)
    for dz, dy, dx in _l1_offsets(radius):
        az, bz = _shift_slices(dz, Z)
        ay, by = _shift_slices(dy, Y)
        ax, bx = _shift_slices(dx, X)
        a = labels[az, ay, ax]
        b = labels[bz, by, bx]
        m = (a > 0) & (b > 0) & (a != b)
        if not m.any():
            continue
        ii, jj, kk = np.nonzero(m)
        aa = a[ii, jj, kk].astype(np.int64)
        bb = b[ii, jj, kk].astype(np.int64)
        # coords of the b-side voxel in volume frame
        bz0 = bz.start if bz.start else 0
        by0 = by.start if by.start else 0
        bx0 = bx.start if bx.start else 0
        for t in range(len(ii)):
            k = (min(aa[t], bb[t]), max(aa[t], bb[t]))
            lst = touch.setdefault(k, [])
            if len(lst) < 64:
                lst.append((int(ii[t] + bz0), int(jj[t] + by0), int(kk[t] + bx0)))
    return {k: v[len(v) // 2] for k, v in touch.items()}


def agglomerate(predict_fn, image_gpu, labels: np.ndarray, cfg, K: int = 64) -> np.ndarray:
    """Merge split pieces of the same wrap via decision-point reseeding.

    For each candidate pair, reseed a flood-fill at the interface decision point;
    merge only if the regrowth reclaims >= agglo_consistency of *both* segments
    (mutual consistency). A true inter-wrap gap fails this test -> stays split."""
    pairs = candidate_pairs(labels, cfg.agglo_radius)
    parent = {i: i for i in np.unique(labels) if i != 0}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    for (a, b), dp in pairs.items():
        A = labels == a
        Bm = labels == b
        grown = flood_fill_object(predict_fn, image_gpu, dp, cfg, K=K)
        fa = (grown & A).sum() / max(1, A.sum())
        fb = (grown & Bm).sum() / max(1, Bm.sum())
        if min(fa, fb) >= cfg.agglo_consistency:
            union(a, b)
    out = np.zeros_like(labels)
    remap = {}
    nxt = 1
    for i in np.unique(labels):
        if i == 0:
            continue
        r = find(i)
        if r not in remap:
            remap[r] = nxt; nxt += 1
        out[labels == i] = remap[r]
    return out


# ----------------------------------------------------------------------------
# Block-wise slab driver with cross-block stitching (DESIGN.md 6.4)
# ----------------------------------------------------------------------------
def block_grid(shape, block, halo):
    """Yield (z0,y0,x0, z1,y1,x1) core+halo block bounds tiling `shape`."""
    Z, Y, X = shape
    bz, by, bx = block
    for z0 in range(0, Z, bz):
        for y0 in range(0, Y, by):
            for x0 in range(0, X, bx):
                yield (max(0, z0 - halo), max(0, y0 - halo), max(0, x0 - halo),
                       min(Z, z0 + bz + halo), min(Y, y0 + by + halo), min(X, x0 + bx + halo))


def stitch_blocks(global_labels: np.ndarray, block_labels: np.ndarray,
                  origin, next_label: int, overlap_thr: float = 0.5):
    """Merge a block's labels into the global volume; union labels that overlap
    substantially in already-filled (halo) regions."""
    oz, oy, ox = origin
    Z, Y, X = block_labels.shape
    view = global_labels[oz:oz + Z, oy:oy + Y, ox:ox + X]
    parent = {}

    def new_global(bl):
        nonlocal next_label
        parent[bl] = next_label
        next_label += 1
        return parent[bl]

    for bl in np.unique(block_labels):
        if bl == 0:
            continue
        m = block_labels == bl
        existing = view[m & (view > 0)]
        if existing.size and existing.size >= overlap_thr * m.sum():
            vals, cnts = np.unique(existing, return_counts=True)
            gl = int(vals[cnts.argmax()])
        else:
            gl = new_global(bl)
        write = m & (view == 0)
        view[write] = gl
    global_labels[oz:oz + Z, oy:oy + Y, ox:ox + X] = view
    return next_label
