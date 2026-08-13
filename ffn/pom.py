"""Predicted-Object-Map (POM) state and batched FOV crop/paste primitives.

The recurrent FFN state is a per-example logit canvas (the POM). During a
training unroll the FOV center performs a bounded walk; the canvas is sized
`fov + 2*walk_radius` so every reachable FOV stays in bounds.

All primitives are batched and GPU-native: a whole batch of FOVs is cropped
from the GPU-resident volume stack with a single gather, so there are no host
round-trips inside the unroll (DESIGN.md 4.3).

Recurrence convention (faithful to Januszewski et al. / google-ffn):
    the network outputs *absolute* POM logits for the FOV and we *overwrite*
    the FOV region of the canvas (clamped). An additive-in-logit variant is
    available via `additive=True` for ablation (DESIGN.md 6.1).
"""
from __future__ import annotations

import math
import torch


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def make_offset_grid(fov: int, device) -> tuple:
    """Return (oz, oy, ox) each [fov,fov,fov] of centered offsets in [-h, h]."""
    h = fov // 2
    a = torch.arange(-h, h + 1, device=device)
    oz = a.view(fov, 1, 1).expand(fov, fov, fov)
    oy = a.view(1, fov, 1).expand(fov, fov, fov)
    ox = a.view(1, 1, fov).expand(fov, fov, fov)
    return oz, oy, ox


def batched_crop(stack: torch.Tensor, vol_idx: torch.Tensor,
                 centers: torch.Tensor, fov: int,
                 offset_grid: tuple | None = None) -> torch.Tensor:
    """Crop a batch of FOVs from a stack of volumes.

    stack:    [N, Z, Y, X]  (N volumes on device)
    vol_idx:  [B] long       (which volume each example draws from)
    centers:  [B, 3] long     (z,y,x center of each FOV, in stack coords)
    returns:  [B, 1, fov, fov, fov]
    """
    device = stack.device
    B = centers.shape[0]
    if offset_grid is None:
        offset_grid = make_offset_grid(fov, device)
    oz, oy, ox = offset_grid
    Z, Y, X = stack.shape[-3:]
    zz = (centers[:, 0].view(B, 1, 1, 1) + oz.unsqueeze(0)).clamp_(0, Z - 1)
    yy = (centers[:, 1].view(B, 1, 1, 1) + oy.unsqueeze(0)).clamp_(0, Y - 1)
    xx = (centers[:, 2].view(B, 1, 1, 1) + ox.unsqueeze(0)).clamp_(0, X - 1)
    vi = vol_idx.view(B, 1, 1, 1)
    out = stack[vi, zz, yy, xx]           # [B, fov, fov, fov]
    return out.unsqueeze(1)


def batched_write(canvas: torch.Tensor, centers: torch.Tensor, fov: int,
                  values: torch.Tensor, offset_grid: tuple | None = None) -> None:
    """Overwrite each example's FOV region of its canvas with `values`.

    canvas:  [B, 1, C, C, C]  (per-example; index b -> volume b)
    centers: [B, 3] long       (FOV center in canvas coords)
    values:  [B, 1, fov, fov, fov]
    """
    device = canvas.device
    B = centers.shape[0]
    if offset_grid is None:
        offset_grid = make_offset_grid(fov, device)
    oz, oy, ox = offset_grid
    C = canvas.shape[-1]
    zz = (centers[:, 0].view(B, 1, 1, 1) + oz.unsqueeze(0)).clamp_(0, C - 1)
    yy = (centers[:, 1].view(B, 1, 1, 1) + oy.unsqueeze(0)).clamp_(0, C - 1)
    xx = (centers[:, 2].view(B, 1, 1, 1) + ox.unsqueeze(0)).clamp_(0, C - 1)
    bi = torch.arange(B, device=device).view(B, 1, 1, 1).expand_as(zz)
    canvas[bi, 0, zz, yy, xx] = values[:, 0]


def init_pom_canvas(B: int, canvas_size: int, seed_pad: int,
                    tgt_lo: float, tgt_hi: float, clamp: float,
                    device) -> torch.Tensor:
    """Initialize a per-example POM logit canvas: lo everywhere, hi seed disc at center."""
    lo, hi = logit(tgt_lo), logit(tgt_hi)
    canvas = torch.full((B, 1, canvas_size, canvas_size, canvas_size),
                        lo, dtype=torch.float32, device=device)
    c = canvas_size // 2
    r = seed_pad
    if r <= 0:
        canvas[:, 0, c, c, c] = hi
    else:
        zz, yy, xx = torch.meshgrid(
            torch.arange(canvas_size, device=device),
            torch.arange(canvas_size, device=device),
            torch.arange(canvas_size, device=device), indexing="ij")
        ball = ((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2) <= r * r
        canvas[:, :, ball] = hi
    canvas.clamp_(-clamp, clamp)
    return canvas


def pom_to_input(pom_logit: torch.Tensor) -> torch.Tensor:
    """Map POM logits to the network's second input channel, O(1) scale [-1,1]
    (DESIGN.md 6.2: feed 2*sigmoid(POM)-1)."""
    return 2.0 * torch.sigmoid(pom_logit) - 1.0


def neg_to_input(neg: torch.Tensor) -> torch.Tensor:
    """Map the negative-evidence canvas [0,1] to the third input channel, same [-1,1] scale."""
    return 2.0 * neg - 1.0


def init_neg_canvas(B: int, canvas_size: int, device) -> torch.Tensor:
    """Negative-evidence canvas `r`: 0 = never visited. See `apply_update(neg=...)`.

    THE DEFECT THIS FIXES (research/13 S1.3, verified against this file). `cfg.pom_init` defaults to
    0.05 and `cfg.tgt_lo` is 0.05, and `apply_update` OVERWRITES the canvas with raw logits, so a
    voxel the network has LOOKED AT AND CONFIDENTLY REJECTED is written back at exactly the value of
    a voxel it has NEVER SEEN. Through `pom_to_input` both read -0.9. The state therefore has
    0.05->0.95 of dynamic range for positive evidence and ZERO for negative -- the exact inverse of
    our cost asymmetry, and the reason one bad FOV compounds into a permanent merge: every visit can
    raise the state and nothing can lower it.

    `inference.py` even documents that pom_init "separates" pad from target -- but the default VALUE
    collapses them again, and no constant can fix it: pom_init=0.5 (the paper's pad) is the most
    growth-permissive knob in the system and `as_tracer` reverted it for that reason. Separating the
    two states needs a second channel, not a different number.
    """
    return torch.zeros((B, 1, canvas_size, canvas_size, canvas_size),
                       dtype=torch.float32, device=device)


def apply_update(canvas: torch.Tensor, centers: torch.Tensor, fov: int,
                 logits: torch.Tensor, clamp: float, additive: bool = False,
                 offset_grid: tuple | None = None,
                 written: torch.Tensor | None = None,
                 freeze_logit: float = 0.0,
                 neg: torch.Tensor | None = None) -> None:
    """Write the network output into the canvas FOV (overwrite, or additive).

    `written` (optional, same shape as canvas): bool mask of voxels the FFN has already written. When
    given, the paper's split-biasing rule applies -- a voxel is left UNCHANGED if it was previously
    written AND its prior probability was < 0.5 AND the new value would be larger. This biases toward
    splits wherever successive FOVs disagree about background, which is precisely the blind-contact case.
    """
    B = canvas.shape[0]
    vi = torch.arange(B, device=canvas.device)
    cur = batched_crop(canvas.squeeze(1), vi, centers, fov, offset_grid)
    if additive:
        new = (cur + logits).clamp(-clamp, clamp)
    else:
        new = logits.clamp(-clamp, clamp)
    if written is not None:
        w_fov = batched_crop(written.squeeze(1).float(), vi, centers, fov, offset_grid) > 0.5
        # freeze_logit: the freeze applies where the prior is BELOW this logit (0.0 == p<0.5,
        # the paper's rule; cfg.pom_ratchet_freeze_p feeds it in probability units)
        frozen = w_fov & (cur < freeze_logit) & (new > cur)
        new = torch.where(frozen, cur, new)
        batched_write(written, centers, fov, torch.ones_like(w_fov, dtype=written.dtype), offset_grid)
    if neg is not None:
        # NEGATIVE EVIDENCE, accumulated as a running max of (1 - p) over every visit:
        #   never visited      -> 0.00        (channel -1.0)
        #   visited, claimed   -> ~0.05       (channel -0.9)
        #   visited, rejected  -> ~0.95       (channel +0.9)
        # `max` makes rejection STICKY while claiming stays revisable, which is the same asymmetry
        # as the cost function (a merge is unrecoverable, a split is interpolated downstream) and the
        # same asymmetry the external ratchet enforces blindly. Because the network can now SEE it,
        # merge-safety no longer has to be bought with global timidity -- it can learn "do not
        # re-claim where r is high, extend freely where r is 0". That is precisely the mechanism the
        # recorded kill of `pom_ratchet_train` lacked (Addendum 4: "the lock can only block growth"),
        # which held only because the lock was invisible to the network.
        cur_neg = batched_crop(neg.squeeze(1), vi, centers, fov, offset_grid)
        batched_write(neg, centers, fov,
                      torch.maximum(cur_neg, 1.0 - torch.sigmoid(new.detach())), offset_grid)
    batched_write(canvas, centers, fov, new, offset_grid)


def corrupt_canvas(canvas: torch.Tensor, inst: torch.Tensor, seed_inst: torch.Tensor,
                   *, kinds: torch.Tensor, leak_frac: float, hole_frac: float,
                   soften: float, tgt_lo: float, tgt_hi: float, pom_init: float,
                   clamp: float, coords: tuple, gen=None, present=None) -> dict:
    """Inject on-policy ERROR states into the POM canvas (research/11 #1) -- IN PLACE.

    WHY. A merge is not a single-step mistake; it is a leak the recurrence AMPLIFIES: the POM says
    foreground -> the net re-confirms it on the next FOV -> the movement gate opens -> the fill walks
    into the neighbouring wrap. Training has NEVER produced a canvas containing a false claim, so the
    network is never asked to retract one, and at inference the moment a leak starts it is both
    off-distribution AND untrained for the recovery. DAgger (Ross et al., AISTATS 2011): compounding
    error in sequential prediction is fixed only by training on the ERROR-state distribution. RITM
    (Sofiiuk et al., ICIP 2022): perturbing the previous mask was the single change that revived
    iterative segmentation. The FFN paper itself seeds fills from FOREIGN masks at agglomeration time
    -- a POM distribution our training never produces.

    The target is untouched (it is always the true same-instance mask), so "this POM voxel is wrong,
    predict low" becomes DIRECTLY supervised for the first time.

    `canvas` [B,1,P,P,P] logits, `inst` [B,P,P,P] ids, `seed_inst` [B]. `kinds` [B] int8:
    0 = none, 1 = leak, 2 = hole, 3 = soften. Canvas-level only -- no interpolation, so unlike any
    spatial augmentation this can neither collapse an inter-wrap gap nor drift a label.

    Returns per-kind painted voxel counts (diagnostics; cheap, already on device).
    """
    B = canvas.shape[0]
    P = canvas.shape[-1]
    lo, hi, mid = logit(tgt_lo), logit(tgt_hi), logit(pom_init)
    zz, yy, xx = coords                                    # each [P,P,P] long
    si = seed_inst.view(B, 1, 1, 1)
    seed_mask = inst == si
    other = (inst > 0) & (~seed_mask)

    def _ball(elig, cap_frac, ref_vol):
        """Random connected patch: a ball centred on a random ELIGIBLE voxel, intersected back with
        the eligible set -- so a leak lies ON the neighbouring lamina and never in air, and a hole
        lies strictly inside the true mask. Radius is capped so painted volume <= cap_frac*ref_vol,
        which is how the 'leak volume <= ~25% of FOV positive volume' bound is enforced by
        construction rather than by hoping."""
        w = elig.reshape(B, -1).float()
        has = w.sum(1) > 0
        idx = torch.multinomial(w + 1e-8, 1, generator=gen).squeeze(1)      # [B]
        cz = idx // (P * P)
        cy = (idx // P) % P
        cx = idx % P
        # r from the volume cap: (4/3)pi r^3 <= cap*ref  ->  r <= (3*cap*ref/(4pi))^(1/3)
        rmax = (3.0 * cap_frac * ref_vol.clamp(min=1.0) / (4.0 * math.pi)) ** (1.0 / 3.0)
        r = rmax.clamp(1.0, 6.0)
        d2 = ((zz.unsqueeze(0) - cz.view(B, 1, 1, 1)) ** 2
              + (yy.unsqueeze(0) - cy.view(B, 1, 1, 1)) ** 2
              + (xx.unsqueeze(0) - cx.view(B, 1, 1, 1)) ** 2)
        return (d2 <= (r * r).view(B, 1, 1, 1)) & elig & has.view(B, 1, 1, 1)

    # `present`: which kind codes occur this call, decided on the HOST by the caller. Testing
    # `kinds.any()` here would sync the device once per kind per step; the caller already knows.
    def _has(k):
        return (k in present) if present is not None else bool((kinds == k).any())

    seed_vol = seed_mask.reshape(B, -1).sum(1).float()
    cur = canvas[:, 0]
    out = {}

    m = (kinds == 1).view(B, 1, 1, 1)
    if _has(1):
        # LEAK: claim a patch of an ADJACENT instance. Restricted to other-instance voxels that
        # actually touch the seed sheet (6-neighbour dilation) -- a leak across a distant lamina is
        # not the failure mode and would just be noise.
        near = torch.nn.functional.max_pool3d(
            seed_mask.float().unsqueeze(1), 3, stride=1, padding=1).squeeze(1) > 0
        reg = _ball(other & near, leak_frac, seed_vol) & m
        cur.masked_fill_(reg, hi)
        out["leak"] = reg.reshape(B, -1).sum(1)

    m = (kinds == 2).view(B, 1, 1, 1)
    if _has(2):
        # HOLE: erase part of the TRUE mask. Only meaningful once the canvas has been filled, i.e.
        # at t>0 -- at t=0 the canvas is `lo` everywhere but the seed disc and this is a no-op.
        reg = _ball(seed_mask, hole_frac, seed_vol) & m
        cur.masked_fill_(reg, lo)
        out["hole"] = reg.reshape(B, -1).sum(1)

    m = (kinds == 3).view(B, 1, 1, 1)
    if _has(3):
        # SOFTEN: pull confident logits back toward pom_init, i.e. "you are less sure than you think".
        soft = cur + (mid - cur) * float(soften)
        canvas[:, 0] = torch.where(m.expand_as(cur), soft, cur)
        out["soften"] = m.expand_as(cur).reshape(B, -1).sum(1)

    canvas.clamp_(-clamp, clamp)
    return out
