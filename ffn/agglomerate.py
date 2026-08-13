"""FFN agglomeration by RESEGMENTATION -- the 2018 Online Methods criterion.

WHY THIS FILE EXISTS. `inference.agglomerate` implements something the paper does not describe: it
seeds ONE flood-fill at the A/B interface point and merges on mutual reclaim. A seed placed on a
blind interface is *guaranteed* to grow into both wraps, so that test false-merges exactly the case
it must refuse -- which is why `cfg.agglo_enabled` was set False and the pipeline has shipped with
NO assembly stage. FINDINGS 3.3: "the finding that FFN agglomeration is unsafe here is not
supported by the implementation that produced it."

THE ACTUAL CRITERION (Januszewski et al. 2018, Online Methods; DESIGN 6.7):
  1. REMOVE both segments A and B from the subvolume (they become free space; every OTHER segment
     stays claimed, so a regrowth cannot wander into a third object).
  2. Seed at the EDT MAXIMUM INSIDE each fragment -- never on the seam. This is the same principle
     the seed audit confirmed: a seed on a contact plane starts from ambiguous evidence.
  3. Run TWO INDEPENDENT regrowths, one from inside A, one from inside B.
  4. Accept the merge only if ALL of:
        iou(regrowA, regrowB)      > agglo_iou          (the two regrowths agree)
        reclaim(regrowA -> B)      > agglo_consistency  (A's regrowth takes B)
        reclaim(regrowB -> A)      > agglo_consistency  (B's regrowth takes A)
        deleted / |A u B|          < agglo_deleted_frac (little of the pair went unclaimed)
  5. Retry under exclusion (up to agglo_retries): if a regrowth is degenerate, zero a cuboid around
     the seed and take the next EDT maximum.

WHY IT IS MERGE-SAFE WHERE THE OLD ONE IS NOT. If A and B are two different wraps, a fill seeded
deep inside A refuses to cross the contact (that is the one thing the network is trained to do), so
regrowA ~ A and regrowB ~ B, giving iou ~ 0 -> REJECT. If A and B are two fragments of ONE sheet,
both regrowths flow through the whole sheet, giving iou ~ 1 and high mutual reclaim -> ACCEPT. The
old interface-seeded test cannot make this distinction even in principle.

Splits are our cheap currency and merges are catastrophic, so this stage is the intended way to buy
merge-safety: over-segment hard at decode (e.g. a high ratchet freeze), then recover the splits here.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage as ndi

from .inference import candidate_pairs, flood_fill_object


def _edt_max_seed(mask: np.ndarray, excluded: np.ndarray | None = None, border: int = 0):
    """Voxel of `mask` maximally interior to it (EDT argmax), optionally excluding a region.

    `border` drops candidates within that many voxels of the SUBVOLUME face. This is not cosmetic:
    `flood_fill_object` refuses to enqueue a centre closer than fov//2 to a face (`inb`), so a seed
    there yields steps=0 and a 5-voxel mask -- a silent degenerate regrowth. distance_transform_edt
    does not treat the array boundary as background, so an EDT argmax lands ON the face whenever a
    fragment runs off the edge (measured: seed (0,0,17) on a full-span slab), which is the common
    case for fragments touching a block face."""
    m = mask if excluded is None else (mask & ~excluded)
    if border > 0:
        keep = np.zeros_like(m)
        b = int(border)
        keep[b:m.shape[0] - b, b:m.shape[1] - b, b:m.shape[2] - b] = True
        m = m & keep
    if not m.any():
        return None
    edt = ndi.distance_transform_edt(m)
    return tuple(int(v) for v in np.unravel_index(int(np.argmax(edt)), edt.shape))


def _pair_subvolume(labels: np.ndarray, a: int, b: int, margin: int):
    """Tight bbox around A u B expanded by `margin`, as slices + the cropped arrays."""
    both = (labels == a) | (labels == b)
    idx = np.argwhere(both)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + 1 + margin, np.array(labels.shape))
    sl = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
    return sl, both[sl]


def resegment_pair(predict_fn, image_gpu, labels, a, b, cfg, K=64, stats=None):
    """Run the paper's two-regrowth test on one candidate pair. Returns (merge: bool, info: dict)."""
    margin = cfg.fov + cfg.delta
    sl, both = _pair_subvolume(labels, a, b, margin)
    sub_lab = labels[sl]
    A = sub_lab == a
    B = sub_lab == b
    n_both = int(both.sum())
    info = dict(pair=(int(a), int(b)), n_both=n_both, iou=0.0, rec_a=0.0, rec_b=0.0,
                deleted=1.0, reason="")
    if n_both == 0 or not A.any() or not B.any():
        info["reason"] = "empty"
        return False, info
    # every OTHER segment stays claimed; A and B are freed so a regrowth may span them
    claimed = sub_lab.copy()
    claimed[A | B] = 0
    img_sub = image_gpu[sl].contiguous()

    def regrow(seed_mask, other_mask):
        excl = np.zeros_like(seed_mask)
        for _ in range(max(1, int(getattr(cfg, "agglo_retries", 8)))):
            s = _edt_max_seed(seed_mask, excl if excl.any() else None,
                              border=cfg.fov // 2 + 1)
            if s is None:
                return None
            st = {}
            m = flood_fill_object(predict_fn, img_sub, s, cfg, K=K, claimed=claimed, stats=st)
            # degenerate: never moved, or claimed almost nothing of its own fragment
            if st.get("steps", 0) >= getattr(cfg, "min_fov_steps", 1) and \
               (m & seed_mask).sum() >= 0.2 * seed_mask.sum():
                return m
            r = max(2, cfg.delta)
            z, y, x = s
            excl[max(0, z - r):z + r + 1, max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = True
        return None

    with torch.no_grad():
        ra = regrow(A, B)
        rb = regrow(B, A)
    if ra is None or rb is None:
        info["reason"] = "degenerate_regrowth"
        return False, info
    inter = int((ra & rb).sum())
    union = int((ra | rb).sum())
    info["iou"] = inter / max(union, 1)
    info["rec_a"] = int((ra & B).sum()) / max(int(B.sum()), 1)      # A's regrowth reclaims B
    info["rec_b"] = int((rb & A).sum()) / max(int(A.sum()), 1)      # B's regrowth reclaims A
    info["deleted"] = int((both & ~(ra | rb)).sum()) / max(n_both, 1)
    ok = (info["iou"] > getattr(cfg, "agglo_iou", 0.8)
          and info["rec_a"] > cfg.agglo_consistency
          and info["rec_b"] > cfg.agglo_consistency
          and info["deleted"] < getattr(cfg, "agglo_deleted_frac", 0.02))
    if not ok:
        fails = []
        if info["iou"] <= getattr(cfg, "agglo_iou", 0.8):
            fails.append("iou")
        if info["rec_a"] <= cfg.agglo_consistency or info["rec_b"] <= cfg.agglo_consistency:
            fails.append("consistency")
        if info["deleted"] >= getattr(cfg, "agglo_deleted_frac", 0.02):
            fails.append("deleted")
        info["reason"] = "+".join(fails)
    if stats is not None:
        stats.setdefault("pairs", []).append(info)
    return bool(ok), info


def agglomerate_resegment(predict_fn, image_gpu, labels: np.ndarray, cfg, K: int = 64,
                          stats: dict | None = None, max_pairs: int = 0) -> np.ndarray:
    """Merge same-wrap fragments via the 2018 resegmentation criterion. Merge-safe by design."""
    pairs = candidate_pairs(labels, cfg.agglo_radius)
    ids = [int(i) for i in np.unique(labels) if i]
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    order = sorted(pairs.keys())
    if max_pairs:
        order = order[:max_pairs]
    n_ok = 0
    for (a, b) in order:
        if find(int(a)) == find(int(b)):
            continue                       # already joined transitively
        ok, _ = resegment_pair(predict_fn, image_gpu, labels, int(a), int(b), cfg, K=K,
                               stats=stats)
        if ok:
            parent[find(int(a))] = find(int(b))
            n_ok += 1
    out = np.zeros_like(labels)
    remap, nxt = {}, 1
    for i in ids:
        r = find(i)
        if r not in remap:
            remap[r] = nxt
            nxt += 1
        out[labels == i] = remap[r]
    if stats is not None:
        stats["n_pairs"] = len(order)
        stats["n_merged"] = n_ok
    return out
