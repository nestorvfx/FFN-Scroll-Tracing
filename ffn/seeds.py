"""Seed generation, adapted for thin laminar sheets (DESIGN.md 5.5).

Neuron-FFN seeds by EDT peaks (blob interiors). Our sheets are ~3-4 vox thick,
so an EDT peak is only ~1.5 vox from a boundary and can straddle two laminae.
Instead we seed on the 3-D medial axis (skeleton), which is one-lamina-deep by
construction.

  - Training seeds: skeleton of each instance mask -> candidate FOV centers that
    are unambiguously interior to a single sheet.
  - Inference seeds: from a predicted fiber probability -> threshold -> EDT ->
    skeleton -> candidate seeds in descending EDT order, with a reject radius
    against already-committed segments.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def instance_skeleton(mask: np.ndarray) -> np.ndarray:
    """3-D medial surface of a binary mask (bool array), as the EDT ridge.

    Replaces skimage.morphology.skeletonize, whose 3-D thinning silently returns
    an ALL-ZERO array on our 4-6 voxel sheets (scikit-image #3757, parity/thickness
    dependent). The medial surface is instead the Blum distance ridge: voxels whose
    Euclidean distance-to-boundary is a local maximum in their 3^3 neighborhood.
    These are maximally interior along thickness (one-lamina-deep by construction),
    and the global-max voxel is always a local max, so a non-empty mask NEVER yields
    an empty ridge. Robust and parity-independent."""
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return np.zeros(m.shape, dtype=bool)
    edt = ndi.distance_transform_edt(m)
    ridge = ndi.maximum_filter(edt, size=3)
    return m & (edt >= ridge - 1e-6) & (edt > 0)


def cube_skeletons(inst: np.ndarray, min_size: int = 0):
    """Skeletonize every instance in a cube.

    Returns (skel, labels): skel is a bool volume (union of per-instance
    skeletons); labels is an int volume carrying the instance id on skeleton
    voxels (0 elsewhere). Instance ids are read directly (may be non-contiguous).
    """
    skel = np.zeros(inst.shape, dtype=bool)
    lab = np.zeros(inst.shape, dtype=inst.dtype)
    ids = np.unique(inst)
    for lbl in ids:
        if lbl == 0:
            continue
        m = inst == lbl
        if m.sum() < min_size:
            continue
        s = instance_skeleton(m)
        skel |= s
        lab[s] = lbl
    return skel, lab


def cube_medial_seeds(inst: np.ndarray, min_size: int = 0, edt_q: float = 0.5):
    """Fast alternative to cube_skeletons for *training* candidate generation.

    For a thin sheet the high-EDT voxels ARE the medial (one-lamina-deep, maximally
    interior in thickness) seeds, and per-instance EDT is ~10x faster than
    skeletonize_3d. Returns (medial_bool, lab) with the same interface as
    cube_skeletons. Seeds = voxels with EDT >= max(1, q-quantile of the instance's
    EDT), guaranteeing they sit at least ~1 vox inside the sheet.
    """
    medial = np.zeros(inst.shape, dtype=bool)
    lab = np.zeros(inst.shape, dtype=inst.dtype)
    for lbl in np.unique(inst):
        if lbl == 0:
            continue
        m = inst == lbl
        if m.sum() < min_size:
            continue
        edt = ndi.distance_transform_edt(m)
        thr = max(1.0, float(np.quantile(edt[m], edt_q)))
        sel = m & (edt >= thr)
        medial |= sel
        lab[sel] = lbl
    return medial, lab


def edt_of_mask(mask: np.ndarray) -> np.ndarray:
    return ndi.distance_transform_edt(mask)


def instance_inference_seeds(inst: np.ndarray, min_edt: float = 1.0, spacing: int = 0):
    """inference_seeds semantics computed PER INSTANCE (labels known -- eval harness only).

    The union-mask path (`inference_seeds(fiber=(inst>0))`) has a structural defect the panel
    inherits: where two wraps touch, the union has no internal boundary, so its Blum ridge lies
    ON the contact plane and its EDT there is the HIGHEST in the volume -- descending-EDT order
    then consumes exactly the contact-straddling seeds first. Per-instance ridges are
    one-lamina-deep by construction and their EDT is the instance's own half-thickness.

    Returns (coords[K,3] int64 descending own-instance EDT, edt_vals[K] float32), spacing-thinned
    like inference_seeds (at most one seed per spacing^3 cell, highest EDT wins)."""
    coords_all = []
    vals_all = []
    for lbl in np.unique(inst):
        if lbl == 0:
            continue
        m = inst == lbl
        edt = ndi.distance_transform_edt(m)
        skel = instance_skeleton(m)
        cand = skel & (edt >= min_edt)
        if not cand.any():
            cand = skel if skel.any() else (m & (edt >= edt.max()))
        coords_all.append(np.argwhere(cand))
        vals_all.append(edt[cand])
    if not coords_all:
        return np.zeros((0, 3), np.int64), np.zeros((0,), np.float32)
    coords = np.concatenate(coords_all)
    vals = np.concatenate(vals_all)
    order = np.argsort(-vals)
    coords, vals = coords[order], vals[order]
    if spacing and spacing > 1 and len(coords):
        cells = coords // spacing
        seen = set()
        keep = np.zeros(len(coords), bool)
        for i, c in enumerate(map(tuple, cells)):
            if c not in seen:
                seen.add(c)
                keep[i] = True
        coords, vals = coords[keep], vals[keep]
    return coords.astype(np.int64), vals.astype(np.float32)


def inference_seeds(fiber_prob: np.ndarray, thr: float = 0.5,
                    min_edt: float = 1.0, spacing: int = 0):
    """Produce ordered candidate seeds from a fiber probability volume.

    Returns coords [K,3] (z,y,x) sorted by descending EDT (thickest-interior
    first) and the EDT value at each. The reject-radius rule against committed
    segments is applied by the caller during serial flood-fill (DESIGN.md 6.7).

    `spacing`>0 grid-thins the seeds: at most one (highest-EDT) seed per
    spacing^3 cell, so a dense skeleton does not produce a 100k-seed serial loop.
    """
    mask = fiber_prob >= thr
    if mask.sum() == 0:
        return np.zeros((0, 3), np.int64), np.zeros((0,), np.float32)
    edt = ndi.distance_transform_edt(mask)
    skel = instance_skeleton(mask)
    cand = skel & (edt >= min_edt)
    if not cand.any():            # fiber thinner than min_edt -> NEVER return 0 seeds
        cand = skel if skel.any() else (mask & (edt >= edt.max()))
    coords = np.argwhere(cand)
    vals = edt[cand]
    order = np.argsort(-vals)
    coords, vals = coords[order], vals[order]
    if spacing and spacing > 1 and len(coords):
        cells = coords // spacing
        seen = set()
        keep = np.zeros(len(coords), bool)
        for i, c in enumerate(map(tuple, cells)):
            if c not in seen:
                seen.add(c); keep[i] = True
        coords, vals = coords[keep], vals[keep]
    return coords.astype(np.int64), vals.astype(np.float32)
