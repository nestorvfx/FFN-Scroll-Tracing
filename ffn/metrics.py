"""Evaluation metrics (DESIGN.md 7), self-contained and unit-tested against
known-answer phantoms (no graph_tool / funlib dependency).

  - adjacent_wrap_merge_rate  : PRIMARY north-star (maps 1:1 to the winding
                                catastrophe), with a blind-contact subset.
  - erl / nerl                : expected run length (punishes merges).
  - voi_split_merge           : VOI decomposed; VOI_merge is the 2nd north-star.
  - adapted_rand              : SNEMI3D over/under-seg summary.
  - bridge_count              : the pipeline's real failure signal (Guo-Hall
                                thinning bridges between distinct instances).

All operate on integer label volumes (0 = background). GT ids may be
non-contiguous; every routine reads ids via np.unique.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

try:
    import cc3d
except Exception:  # pragma: no cover
    cc3d = None


# A predicted label that claims >= this fraction of a GT wrap counts as
# "covering" it. A label covering >=2 wraps (each above this floor) is a MERGE
# and contributes 0 run length to every wrap it touches. The floor replaces the
# denoising the skeleton used to give for free: it rejects 1-voxel boundary
# bleed while keeping any real cross-wrap merge catastrophic.
_MERGE_COVER_FRAC = 0.05


def _label_cc(vol, connectivity=26):
    """26-connected components of an integer label volume: each maximal
    26-connected region of a single non-zero value becomes one component
    (background 0 is ignored). Uses cc3d (C++, fast) when available, else scipy."""
    if cc3d is not None:
        return cc3d.connected_components(np.ascontiguousarray(vol),
                                         connectivity=connectivity)
    out = np.zeros(vol.shape, np.int64)          # scipy fallback (cc3d missing)
    struct = ndi.generate_binary_structure(vol.ndim, vol.ndim)
    nxt = 0
    for v in np.unique(vol):
        if v == 0:
            continue
        lab, n = ndi.label(vol == v, structure=struct)
        m = lab > 0
        out[m] = lab[m] + nxt
        nxt += n
    return out


# ----------------------------------------------------------------------------
# Contingency helpers
# ----------------------------------------------------------------------------
def _contingency(gt, pred):
    """Return (pairs[K,2] int64, counts[K]) of co-occurring (gt,pred) labels
    over foreground voxels (gt>0 AND pred>0)."""
    fg = (gt > 0) & (pred > 0)
    g = gt[fg].astype(np.int64)
    p = pred[fg].astype(np.int64)
    if g.size == 0:
        return np.zeros((0, 2), np.int64), np.zeros(0, np.int64)
    key = g * (p.max() + 1) + p
    uk, cnt = np.unique(key, return_counts=True)
    gi = uk // (p.max() + 1)
    pi = uk % (p.max() + 1)
    return np.stack([gi, pi], 1), cnt


# ----------------------------------------------------------------------------
# 1. Adjacent-wrap pairwise merge rate (PRIMARY)
# ----------------------------------------------------------------------------
def _adjacency_pairs_dilate(gt: np.ndarray, radius: int = 3):
    """Historical per-label dilation implementation, kept as the equivalence oracle for tests.
    O(#labels x volume) -- do not call on real cubes."""
    ids = [i for i in np.unique(gt) if i != 0]
    struct = ndi.generate_binary_structure(3, 1)
    pairs = set()
    for i in ids:
        m = gt == i
        dil = ndi.binary_dilation(m, structure=struct, iterations=radius)
        neigh = gt[dil & (gt != i) & (gt != 0)]
        for j in np.unique(neigh):
            pairs.add((min(i, int(j)), max(i, int(j))))
    return pairs


def adjacency_pairs(gt: np.ndarray, radius: int = 3):
    """Set of GT wrap pairs (i,j) within L1 distance `radius` of each other.

    Semantics identical to `radius` iterations of 6-connectivity dilation (the historical
    implementation, retained above as the test oracle), but computed as a SINGLE PASS of
    ~2*radius^3 shifted comparisons instead of one full-volume dilation per label -- the
    per-label loop was O(#labels x volume) and dominated eval setup on many-instance cubes."""
    gt = np.asarray(gt)
    Z, Y, X = gt.shape
    pairs = set()
    r = int(radius)
    offs = []
    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                l1 = abs(dz) + abs(dy) + abs(dx)
                if l1 == 0 or l1 > r:
                    continue
                if (dz, dy, dx) > (0, 0, 0):
                    offs.append((dz, dy, dx))

    def _sl(d, n):
        if d > 0:
            return slice(0, n - d), slice(d, n)
        if d < 0:
            return slice(-d, n), slice(0, n + d)
        return slice(0, n), slice(0, n)

    for dz, dy, dx in offs:
        az, bz = _sl(dz, Z)
        ay, by = _sl(dy, Y)
        ax, bx = _sl(dx, X)
        a = gt[az, ay, ax]
        b = gt[bz, by, bx]
        m = (a > 0) & (b > 0) & (a != b)
        if not m.any():
            continue
        aa = a[m].astype(np.int64)
        bb = b[m].astype(np.int64)
        lo = np.minimum(aa, bb)
        hi = np.maximum(aa, bb)
        P1 = int(gt.max()) + 1
        for k in np.unique(lo * P1 + hi).tolist():
            pairs.add((int(k // P1), int(k % P1)))
    return pairs


def _pair_merged(gt, pred, i, j, theta):
    """A predicted label L merges (i,j) if it claims >= theta of *both* wraps."""
    mi = gt == i
    mj = gt == j
    ni, nj = mi.sum(), mj.sum()
    if ni == 0 or nj == 0:
        return False
    pi = pred[mi]
    pj = pred[mj]
    vi, ci = np.unique(pi[pi > 0], return_counts=True)
    vj, cj = np.unique(pj[pj > 0], return_counts=True)
    fi = {int(v): c / ni for v, c in zip(vi, ci)}
    fj = {int(v): c / nj for v, c in zip(vj, cj)}
    for L in set(fi) & set(fj):
        if fi[L] >= theta and fj[L] >= theta:
            return True
    return False


def _pair_is_blind(gt, image, i, j, radius, gap_frac_thr=0.7, gap_intensity=None):
    """A contact is 'blind' if most interface voxels stay bright (no gap).
    gap_intensity default = Otsu-ish midpoint of the image foreground."""
    struct = ndi.generate_binary_structure(3, 1)
    di = ndi.binary_dilation(gt == i, struct, radius)
    dj = ndi.binary_dilation(gt == j, struct, radius)
    interface = di & dj
    if interface.sum() == 0:
        return False
    if gap_intensity is None:
        gap_intensity = float(image[image > 0].mean()) * 0.5
    bright = (image[interface] > gap_intensity).mean()
    return bright >= gap_frac_thr


def adjacent_wrap_merge_rate(gt, pred, radius=3, theta=0.10, image=None):
    """Returns dict: merge_rate over all adjacent pairs, and (if `image` given)
    the blind-contact subset rate. Lower is better; ~0.80 baseline, target <0.05."""
    pairs = adjacency_pairs(gt, radius)
    if not pairs:
        return dict(merge_rate=0.0, n_pairs=0, n_merged=0,
                    blind_merge_rate=float("nan"), n_blind=0, n_blind_merged=0)
    merged = 0
    blind_pairs = 0
    blind_merged = 0
    for (i, j) in pairs:
        m = _pair_merged(gt, pred, i, j, theta)
        merged += int(m)
        if image is not None and _pair_is_blind(gt, image, i, j, radius):
            blind_pairs += 1
            blind_merged += int(m)
    return dict(
        merge_rate=merged / len(pairs), n_pairs=len(pairs), n_merged=merged,
        blind_merge_rate=(blind_merged / blind_pairs) if blind_pairs else float("nan"),
        n_blind=blind_pairs, n_blind_merged=blind_merged)


# ----------------------------------------------------------------------------
# 2. ERL / NERL
# ----------------------------------------------------------------------------
def erl(gt, pred, skeletons=None):
    """Expected Run Length for thin 2-D sheets (papyrus wraps) -- VOLUMETRIC form.

    ERL was defined (Januszewski et al. 2018) for 1-D neurites as the
    length-weighted mean error-free *skeleton path length*. A papyrus wrap is a
    3-D solid, not a curve: its 1-voxel skeleton is a medial *surface*, and
    skimage's 3-D thinning silently returns an EMPTY array on 4-6 voxel sheets
    (scikit-image #3757, thickness/parity dependent) -- which zeroed this metric
    for every prediction. A robust 1-voxel medial surface does not exist for a
    sheet either (a distance ridge shatters into hundreds of components; a TEASAR
    curve skeleton, e.g. kimimaro, collapses one lateral dimension and measures a
    spanning path, not extent). So we keep the SAME ERL estimator but swap the
    object measure from path length to the natural measure of a solid: VOLUME.

    Nodes of a wrap = its foreground voxels. An error-free RUN is a 26-connected
    component of (wrap AND one predicted label that is NOT a merge); run size =
    its voxel count. ERL = sum(run^2)/sum(node), NERL = sum(run^2)/sum(node^2).
    A solid wrap is always its own connected component, so this is robust,
    parity-independent, and needs no skeletonization.

    Every property that made NERL the right selection metric is preserved:
      (a) coverage floor       -- an uncovered voxel (pred==0) joins no run -> 0;
      (b) merge = catastrophic -- a predicted label spanning >=2 GT wraps
          contributes 0 to every wrap it touches;
      (c) size^2 weighting     -- runs contribute size^2 (concentrates on the
          large dense stretches);
      (d) NERL is a single scalar in [0,1];
      (e) not gameable         -- empty/collapse -> 0, a split costs only mildly,
          a merge costs the whole wrap.

    `skeletons`, if given as {wrap_id: coords[N,3]}, is used as the per-wrap node
    set instead of the full volume (used by the analytic phantom tests and lets a
    caller pass a precomputed node set); default None -> volumetric nodes = gt>0.

    Returns dict(erl, nerl, erl_perfect, sum_runsq, sum_perfsq, L). Panels
    aggregate EXACTLY as (Sum sum_runsq)/(Sum sum_perfsq), NOT a mean of NERLs.
    """
    _ZERO = dict(erl=0.0, nerl=0.0, erl_perfect=0.0,
                 sum_runsq=0.0, sum_perfsq=0.0, L=0.0)
    pred = np.asarray(pred)
    # --- per-wrap node volume + node counts n_w, cropped to the node bbox ---
    if skeletons is None:
        node = np.asarray(gt)
        nz = node > 0
        if not nz.any():
            return _ZERO
        sl = ndi.find_objects(nz.astype(np.uint8))[0]
        node = np.ascontiguousarray(node[sl].astype(np.int64, copy=False))
        pred = np.ascontiguousarray(pred[sl])
        ids, cnts = np.unique(node[node > 0], return_counts=True)
        n_w = {int(i): int(c) for i, c in zip(ids, cnts)}
    else:
        node = np.zeros(np.asarray(gt).shape, np.int64)
        n_w = {}
        for wid, coords in skeletons.items():
            if coords.shape[0] == 0:
                continue
            node[coords[:, 0], coords[:, 1], coords[:, 2]] = int(wid)
            n_w[int(wid)] = int(coords.shape[0])
        nz = node > 0
        if not nz.any():
            return _ZERO
        sl = ndi.find_objects(nz.astype(np.uint8))[0]
        node = np.ascontiguousarray(node[sl])
        pred = np.ascontiguousarray(pred[sl])

    L = float(sum(n_w.values()))
    sum_perfsq = float(sum(v * v for v in n_w.values()))
    if L == 0:
        return _ZERO

    P = int(pred.max()) if pred.size else 0
    if P == 0:                                   # empty prediction -> no runs
        return dict(erl=0.0, nerl=0.0, erl_perfect=sum_perfsq / L,
                    sum_runsq=0.0, sum_perfsq=sum_perfsq, L=L)

    # --- merge detection: pred label -> set of wraps it covers >= floor ---
    fg = (node > 0) & (pred > 0)
    g = node[fg].astype(np.int64)
    p = pred[fg].astype(np.int64)
    key = g * (P + 1) + p
    uk, cnt = np.unique(key, return_counts=True)
    gg = uk // (P + 1)
    pp = uk % (P + 1)
    seg_wraps = {}
    for gi, pi, c in zip(gg.tolist(), pp.tolist(), cnt.tolist()):
        if c >= max(1.0, _MERGE_COVER_FRAC * n_w[gi]):
            seg_wraps.setdefault(pi, set()).add(gi)
    is_merged = np.zeros(P + 1, dtype=bool)
    for pi, w in seg_wraps.items():
        if len(w) >= 2:
            is_merged[pi] = True

    # --- runs: 26-connected components of (wrap id, non-merged pred label) ---
    valid = fg & ~is_merged[pred]
    combo = np.zeros(node.shape, np.int64)
    combo[valid] = node[valid] * (P + 1) + pred[valid]
    lab = _label_cc(combo, connectivity=26)
    sizes = np.bincount(lab.reshape(-1))
    if sizes.size:
        sizes[0] = 0                             # drop the background component
    sum_runsq = float((sizes.astype(np.float64) ** 2).sum())

    # sum_runsq/sum_perfsq/L are exposed so multiple cubes/panels aggregate EXACTLY:
    # panel NERL = (Σ_c sum_runsq_c) / (Σ_c sum_perfsq_c), not a mean of per-cube NERLs.
    return dict(erl=sum_runsq / L, nerl=(sum_runsq / sum_perfsq) if sum_perfsq else 0.0,
                erl_perfect=sum_perfsq / L, sum_runsq=sum_runsq,
                sum_perfsq=sum_perfsq, L=L)


# ----------------------------------------------------------------------------
# 3. VOI split / merge
# ----------------------------------------------------------------------------
def voi_split_merge(gt, pred):
    """Return (voi_split, voi_merge). split=H(PRED|GT) (over-seg),
    merge=H(GT|PRED) (under-seg, our 2nd north-star). Lower is better."""
    pairs, cnt = _contingency(gt, pred)
    N = cnt.sum()
    if N == 0:
        return 0.0, 0.0
    pij = cnt / N
    gi, pi = pairs[:, 0], pairs[:, 1]
    # marginals
    gmarg = {}
    pmarg = {}
    for g_, p_, c in zip(gi, pi, pij):
        gmarg[g_] = gmarg.get(g_, 0.0) + c
        pmarg[p_] = pmarg.get(p_, 0.0) + c
    voi_split = 0.0  # H(PRED|GT)
    voi_merge = 0.0  # H(GT|PRED)
    for g_, p_, c in zip(gi, pi, pij):
        voi_split -= c * np.log(c / gmarg[g_])
        voi_merge -= c * np.log(c / pmarg[p_])
    return float(voi_split), float(voi_merge)


# ----------------------------------------------------------------------------
# 4. Adapted-Rand (SNEMI3D)
# ----------------------------------------------------------------------------
def adapted_rand(gt, pred):
    """Return dict(are, precision, recall). are = 1 - F1 of the Rand index over
    foreground. Lower error is better."""
    pairs, cnt = _contingency(gt, pred)
    N = cnt.sum()
    if N == 0:
        return dict(are=0.0, precision=1.0, recall=1.0)
    pij = cnt / N
    gmarg, pmarg = {}, {}
    for (g_, p_), c in zip(pairs, pij):
        gmarg[g_] = gmarg.get(g_, 0.0) + c
        pmarg[p_] = pmarg.get(p_, 0.0) + c
    sum_pij2 = float((pij ** 2).sum())
    sum_a2 = float(sum(v * v for v in gmarg.values()))
    sum_b2 = float(sum(v * v for v in pmarg.values()))
    precision = sum_pij2 / sum_b2 if sum_b2 > 0 else 0.0
    recall = sum_pij2 / sum_a2 if sum_a2 > 0 else 0.0
    f = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return dict(are=1.0 - f, precision=precision, recall=recall)


# ----------------------------------------------------------------------------
# 5. Bridge count (Guo-Hall thinning failure signal)
# ----------------------------------------------------------------------------
def bridge_count(pred: np.ndarray) -> int:
    """Number of distinct predicted-instance pairs that TOUCH (26-adjacent with no
    background gap between them) -- a direct merge bridge in the instance labeling,
    exactly the Guo-Hall failure signal (two blobs joined through the foreground
    show as touching labels). 0 = clean separation (Gate A target).

    Skeleton-free by construction: the old skimage-thinning route returned an
    empty skeleton on our thin sheets (scikit-image #3757) and so falsely reported
    0 bridges. Adjacency is robust and equivalent -- a thinned foreground bridges
    two labels iff those labels are connected through the foreground, i.e. touch."""
    pred = np.asarray(pred)
    if pred.ndim != 3 or pred.max() <= 0:
        return 0
    P1 = int(pred.max()) + 1
    pairs = set()
    # 13 undirected offsets spanning the 26-neighborhood (first nonzero = +1)
    offs = [o for o in ((dz, dy, dx) for dz in (-1, 0, 1)
                        for dy in (-1, 0, 1) for dx in (-1, 0, 1)) if o > (0, 0, 0)]

    def _sl(d, n):                               # (base slice, shifted-by-d slice)
        if d > 0:
            return slice(0, n - d), slice(d, n)
        if d < 0:
            return slice(-d, n), slice(0, n + d)
        return slice(0, n), slice(0, n)

    Z, Y, X = pred.shape
    for dz, dy, dx in offs:
        az, bz = _sl(dz, Z); ay, by = _sl(dy, Y); ax, bx = _sl(dx, X)
        a = pred[az, ay, ax]; b = pred[bz, by, bx]
        m = (a > 0) & (b > 0) & (a != b)
        if not m.any():
            continue
        aa = a[m].astype(np.int64); bb = b[m].astype(np.int64)
        lo = np.minimum(aa, bb); hi = np.maximum(aa, bb)
        for k in np.unique(lo * P1 + hi).tolist():
            pairs.add(k)
    return len(pairs)
