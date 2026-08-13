"""Cross-block identity: mid-plane mutual-maximal-overlap matching + union-find.

Each block claims exactly its core, so no voxel is written twice. Identity across a
block face is decided on the SINGLE plane where the two cores meet: fragment a (block A)
and fragment b (block B) are united iff each is the other's maximal overlap on that
plane and the overlap clears a floor.

Mutual-max is a MATCHING, not a vote. This is deliberate and non-negotiable: label-space
voting was measured at -42% NERL, and a one-sided majority rule (the training-eval path's
stitch) lets a large fragment absorb every small one it touches -- the merge direction.
Under mutual-max, ambiguity (no reciprocal winner) leaves fragments split, which is the
recoverable error.
"""
from __future__ import annotations

import numpy as np

from .blocks import Block, core_in_canvas


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def _plane_matches(pa: np.ndarray, pb: np.ndarray, min_overlap: int):
    """Mutual-maximal-overlap pairs between two label planes (0 = background)."""
    m = (pa > 0) & (pb > 0)
    if not m.any():
        return []
    a, b = pa[m].astype(np.int64), pb[m].astype(np.int64)
    key = a * (b.max() + 1) + b
    uniq, cnt = np.unique(key, return_counts=True)
    ua, ub = uniq // (b.max() + 1), uniq % (b.max() + 1)
    best_a, best_b = {}, {}
    for x, y, c in zip(ua, ub, cnt):
        if c > best_a.get(x, (0, 0))[1]:
            best_a[x] = (y, c)
        if c > best_b.get(y, (0, 0))[1]:
            best_b[y] = (x, c)
    out = []
    for x, (y, c) in best_a.items():
        if c >= min_overlap and best_b.get(y, (None, 0))[0] == x:
            out.append((int(x), int(y), int(c)))
    return out


def assemble(shape, blocks: list[Block], labels: list[np.ndarray],
             min_overlap: int = 40, out: np.ndarray | None = None):
    """Blocks' canvas labels -> one global int32 volume.

    `labels[i]` is canvas-shaped for `blocks[i]`. Returns (global_labels, n_objects,
    n_matches). `out`: optional preallocated int32 volume (memmap/zarr view at scale).
    """
    glob = out if out is not None else np.zeros(shape, np.int32)
    offs = []
    nxt = 1
    for lab in labels:
        offs.append(nxt - 1)
        nxt += int(lab.max())
    uf = UnionFind()

    # write cores (offset labels; background stays 0)
    for b, lab, off in zip(blocks, labels, offs):
        sl = core_in_canvas(b)
        core = lab[sl]
        gz0, gy0, gx0, gz1, gy1, gx1 = b.core
        dst = glob[gz0:gz1, gy0:gy1, gx0:gx1]
        dst[...] = np.where(core > 0, core + off, 0)

    # match on the shared plane of every adjacent core pair
    n_match = 0
    index = {tuple(b.core[:3]): i for i, b in enumerate(blocks)}
    for i, b in enumerate(blocks):
        z0, y0, x0, z1, y1, x1 = b.core
        for ax, (nz, ny, nx) in enumerate(((z1, y0, x0), (z0, y1, x0), (z0, y0, x1))):
            j = index.get((nz, ny, nx))
            if j is None:
                continue
            a_can, b_can = blocks[i].canvas, blocks[j].canvas
            plane = (z1, y1, x1)[ax]                      # global coord of shared plane
            if plane >= shape[ax]:
                continue

            def plane_of(block_idx, canvas):
                cz0, cy0, cx0, cz1, cy1, cx1 = canvas
                lo = [cz0, cy0, cx0]
                sl = [slice(None)] * 3
                sl[ax] = plane - lo[ax]
                if not (0 <= sl[ax] < labels[block_idx].shape[ax]):
                    return None
                return labels[block_idx][tuple(sl)]

            pa, pb = plane_of(i, a_can), plane_of(j, b_can)
            if pa is None or pb is None:
                continue
            # restrict both planes to the overlap of the two canvases in-plane
            for x_lab, y_lab, _ in _plane_matches(pa, pb, min_overlap):
                uf.union(x_lab + offs[i], y_lab + offs[j])
                n_match += 1

    # apply union-find via LUT, relabel contiguous
    lut = np.arange(nxt, dtype=np.int32)
    for lab_id in list(uf.p):
        lut[lab_id] = uf.find(lab_id)
    roots, inv = np.unique(lut, return_inverse=True)
    lut = inv.astype(np.int32)          # contiguous; 0 maps to 0 because root 0 is first
    glob[...] = lut[glob]
    return glob, int(glob.max()), n_match
