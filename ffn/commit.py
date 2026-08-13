"""Turning a POM into a committed object mask.

This is the step where a probability becomes an irreversible assignment, and it was the least
principled line in the pipeline: a hard-coded `sigmoid(pom) >= 0.5` in two places.

Two things happen here, in order:

1. **Calibrated threshold.** `cfg.commit_threshold` is DERIVED from the training class balance and
   the merge:split cost ratio rather than picked -- see `FFNConfig.commit_threshold`. The old 0.5
   corresponded to a true posterior of 0.25 on the measured corpus.

2. **Seed-component restriction** (`cfg.commit_seed_cc`). Keep only the connected component that
   contains the seed. A fill that leaks across a blind contact typically reaches the far wrap through
   a thin bridge, so the far side survives thresholding but is not 26-connected to the seed once the
   bridge falls below threshold. Dropping it removes a merge without touching anything the fill
   legitimately reached -- measured a strict Pareto improvement (real merge -17%, NERL +16%, coverage
   up), which is unusual and worth stating: it is not a trade.

   Cost is kept off the hot path by labelling inside the mask's bounding box only.
"""
from __future__ import annotations

import numpy as np
import torch


def _label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """26-connected components. cc3d when available (fast); scipy otherwise."""
    try:
        import cc3d
        lab = cc3d.connected_components(mask, connectivity=26)
        return lab, int(lab.max())
    except Exception:
        from scipy import ndimage as ndi
        return ndi.label(mask, structure=np.ones((3, 3, 3), bool))


def seed_component(mask: np.ndarray, seed, miss: str = "keep",
                   snap_radius: int = 3) -> np.ndarray:
    """The connected component of `mask` containing `seed`.

    `miss` is the policy when the SEED VOXEL ITSELF is below threshold (the p~0.48 straddle
    case, exactly where merges are made):
      "keep"    -- historical: return `mask` unchanged. NOTE this silently disables the seed-CC
                   merge guard: a multi-component mask (possibly spanning wraps) commits whole.
      "nearest" -- snap to the nearest True voxel within `snap_radius` and keep ITS component;
                   reject (empty) if none is that close.
      "reject"  -- return an empty mask (maximally merge-safe; costs coverage)."""
    z, y, x = (int(v) for v in seed)
    if not mask[z, y, x]:
        if miss == "keep":
            return mask
        if miss == "nearest":
            r = int(snap_radius)
            Z, Y, X = mask.shape
            sub = mask[max(0, z - r):z + r + 1, max(0, y - r):y + r + 1,
                       max(0, x - r):x + r + 1]
            idx = np.argwhere(sub)
            if idx.size:
                off = np.array([max(0, z - r), max(0, y - r), max(0, x - r)])
                ctr = np.array([z, y, x]) - off
                near = idx[np.argmin(((idx - ctr) ** 2).sum(1))] + off
                return seed_component(mask, tuple(near), miss="keep")
        return np.zeros_like(mask)
    idx = np.argwhere(mask)
    if idx.size == 0:
        return mask
    lo = idx.min(0)
    hi = idx.max(0) + 1
    sub = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    lab, n = _label(sub)
    if n <= 1:
        return mask
    keep = lab[z - lo[0], y - lo[1], x - lo[2]]
    if keep == 0:
        return mask
    out = np.zeros_like(mask)
    out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = (lab == keep)
    return out


def commit_mask(pom: torch.Tensor, cfg, seed, return_levels: bool = False):
    """POM logits -> committed bool mask, calibrated and seed-restricted.

    `return_levels=True` additionally returns a uint8 volume quantizing the committed voxels'
    probability into 16 levels over [commit_threshold, 0.99] (0 outside the mask). Thresholding
    one monotone scalar gives a NESTED hierarchy by construction, so `{mask AND level >= k}` is
    an ultrametric family -- the split-side operating curve retained at zero model compute
    (the historical path threw this away for one boolean). Post-hoc levels can only SHRINK an
    object, never reassign voxels between objects: this is the split family, not a merge fix."""
    prob = torch.sigmoid(pom)
    mask = (prob >= cfg.commit_threshold).cpu().numpy()
    if getattr(cfg, "commit_seed_cc", False):
        mask = seed_component(mask, seed, miss=getattr(cfg, "commit_seed_miss", "keep"),
                              snap_radius=int(getattr(cfg, "seed_pad", 1)) + 2)
    if not return_levels:
        return mask
    lo = float(cfg.commit_threshold)
    hi = 0.99
    q = ((prob.cpu().numpy() - lo) / max(hi - lo, 1e-6) * 15.0)
    levels = (np.clip(q, 0, 15).astype(np.uint8) + 1) * mask.astype(np.uint8)
    return mask, levels
