"""Core+halo tiling of an arbitrary volume."""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Block:
    core: tuple      # (z0,y0,x0,z1,y1,x1) -- the region this block CLAIMS
    canvas: tuple    # (z0,y0,x0,z1,y1,x1) -- core +/- halo, clipped to volume


def tile(shape, block: int, halo: int):
    """Blocks whose cores partition `shape` exactly; canvases overlap by 2*halo."""
    Z, Y, X = shape
    out = []
    for z0 in range(0, Z, block):
        for y0 in range(0, Y, block):
            for x0 in range(0, X, block):
                z1, y1, x1 = min(Z, z0 + block), min(Y, y0 + block), min(X, x0 + block)
                cz0, cy0, cx0 = max(0, z0 - halo), max(0, y0 - halo), max(0, x0 - halo)
                cz1, cy1, cx1 = min(Z, z1 + halo), min(Y, y1 + halo), min(X, x1 + halo)
                out.append(Block(core=(z0, y0, x0, z1, y1, x1),
                                 canvas=(cz0, cy0, cx0, cz1, cy1, cx1)))
    return out


def core_in_canvas(b: Block):
    """Slice of the core inside the canvas array."""
    (z0, y0, x0, z1, y1, x1) = b.core
    (cz0, cy0, cx0, *_ ) = b.canvas
    return (slice(z0 - cz0, z1 - cz0), slice(y0 - cy0, y1 - cy0), slice(x0 - cx0, x1 - cx0))
