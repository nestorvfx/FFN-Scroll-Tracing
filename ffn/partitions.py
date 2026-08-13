"""Partition (fill-fraction) computation and balanced coordinate generation.

Analogues of google/ffn's `compute_partitions.py` + `build_coordinates.py`,
reimplemented GPU-native (integral-image box sums instead of scipy per-voxel).

compute_partition_volume: for each foreground voxel, the fraction of the
    (2*lom_radius+1)^3 box that shares the voxel's instance label, quantized
    into the 17 paper bins (DESIGN.md 5.4). Faithful to google/ffn: the
    denominator is the full box volume, so the fraction equals 1.0 only for an
    object larger than the box (solid blob); our thin sheets legitimately
    cluster at low fill, which is exactly why all 17 bins are kept and balanced.

build_coordinates: from skeleton candidate centers, emit a per-cube index of
    training locations balanced across the represented bins, respecting the
    edge margin, with a hard-negative flag for centers near inter-sheet
    contacts (DESIGN.md 6.3 Phase C).
"""
from __future__ import annotations

import numpy as np
import torch


def integral_box_sum(vol: torch.Tensor, r: int) -> torch.Tensor:
    """Sum over a (2r+1)^3 box centered at each voxel, via a 3-D integral image.

    vol: [Z,Y,X] float32 (on device). Border windows are clipped to the volume
    (fewer voxels), matching a summed-area-table with edge clamping. Returns the
    box *counts* [Z,Y,X]; divide by box volume for the fraction.
    """
    Z, Y, X = vol.shape
    # integral image with a zero pad on the low side: I[i,j,k] = sum vol[:i,:j,:k]
    ii = torch.zeros((Z + 1, Y + 1, X + 1), dtype=torch.float64, device=vol.device)
    ii[1:, 1:, 1:] = vol.to(torch.float64).cumsum(0).cumsum(1).cumsum(2)

    z0 = torch.clamp(torch.arange(Z, device=vol.device) - r, min=0)
    z1 = torch.clamp(torch.arange(Z, device=vol.device) + r + 1, max=Z)
    y0 = torch.clamp(torch.arange(Y, device=vol.device) - r, min=0)
    y1 = torch.clamp(torch.arange(Y, device=vol.device) + r + 1, max=Y)
    x0 = torch.clamp(torch.arange(X, device=vol.device) - r, min=0)
    x1 = torch.clamp(torch.arange(X, device=vol.device) + r + 1, max=X)

    def gather3(iz, iy, ix):
        return ii[iz.view(Z, 1, 1), iy.view(1, Y, 1), ix.view(1, 1, X)]

    s = (gather3(z1, y1, x1) - gather3(z0, y1, x1)
         - gather3(z1, y0, x1) - gather3(z1, y1, x0)
         + gather3(z0, y0, x1) + gather3(z0, y1, x0) + gather3(z1, y0, x0)
         - gather3(z0, y0, x0))
    return s.to(torch.float32)


def compute_partition_volume(inst: np.ndarray, lom_radius: int,
                             bins, min_size: int = 0, device="cuda") -> np.ndarray:
    """Return an int8 volume of partition-bin indices; -1 for background/dropped.

    bins: the 18 thresholds (len 18 -> 17 bins) from FFNConfig.partition_bins.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    inst_t = torch.from_numpy(inst.astype(np.int32)).to(dev)
    frac = torch.zeros(inst.shape, dtype=torch.float32, device=dev)
    box_vol = float((2 * lom_radius + 1) ** 3)
    ids = torch.unique(inst_t)
    for lbl in ids.tolist():
        if lbl == 0:
            continue
        m = (inst_t == lbl)
        if int(m.sum()) < min_size:
            continue
        cnt = integral_box_sum(m.to(torch.float32), lom_radius)
        frac[m] = cnt[m] / box_vol
    thr = torch.tensor(bins[1:-1], device=dev)   # interior thresholds -> 17 bins
    binvol = torch.bucketize(frac, thr).to(torch.int16)   # 0..17
    fg = frac > 0
    out = torch.full(inst.shape, -1, dtype=torch.int16, device=dev)
    out[fg] = binvol[fg]
    return out.to(torch.int8).cpu().numpy()


def interior_seeds_gpu(inst_t: "torch.Tensor") -> "torch.Tensor":
    """One-shot GPU medial/interior seeds: voxels whose 3^3 neighborhood is a
    single instance (>=1 vox inside a lamina, away from gap/other sheet). For our
    ~3-vox sheets this is the medial core -- equivalent to skeleton seeding for
    the one-lamina-deep guarantee, but a single max/min-pool instead of per-
    instance EDT/skeletonize. Returns a bool [Z,Y,X] tensor."""
    import torch.nn.functional as F
    idf = inst_t.float()[None, None]
    idmax = F.max_pool3d(idf, 3, 1, 1)
    idpos = torch.where(inst_t > 0, inst_t.float(),
                        torch.full_like(inst_t.float(), 1e9))[None, None]
    idmin = -F.max_pool3d(-idpos, 3, 1, 1)
    interior = (idmax[0, 0] == idmin[0, 0]) & (inst_t > 0)
    return interior


def contact_map(inst: np.ndarray, radius: int, device="cuda") -> np.ndarray:
    """Bool volume: foreground voxels within `radius` of a different instance.

    These are the blind-contact decision points for hard-negative mining
    (DESIGN.md 6.3). Computed by dilating each instance and marking overlaps.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    inst_t = torch.from_numpy(inst.astype(np.int32)).to(dev)
    # max-pool the label id and min (via -maxpool of -id over fg) to detect a
    # neighborhood containing >1 distinct nonzero label.
    import torch.nn.functional as F
    k = 2 * radius + 1
    fg = (inst_t > 0).float()[None, None]
    idf = inst_t.float()[None, None]
    big = -1.0
    idmax = F.max_pool3d(idf, k, 1, radius)
    # min over nonzero labels: set background to +inf then -maxpool(-x)
    idpos = torch.where(inst_t > 0, inst_t.float(), torch.full_like(idf[0, 0], 1e9))[None, None]
    idmin = -F.max_pool3d(-idpos, k, 1, radius)
    has_two = (idmax > idmin + 0.5) & (idmax > 0)  # two distinct labels in window
    out = (has_two[0, 0] & (inst_t > 0)).cpu().numpy()
    return out


def blind_contact_map(inst: np.ndarray, image: np.ndarray, radius: int,
                      gap_intensity: float, device="cuda") -> np.ndarray:
    """Bool volume of BLIND inter-sheet contact voxels (AUDIT FIX 3).

    A voxel is a contact if >=2 instances lie within `radius` (see contact_map). It is
    *blind* if the interface carries NO CT-intensity gap: the local minimum intensity over
    the (2*radius+1)^3 window stays >= `gap_intensity` (there is no dark gap/air voxel
    between the sheets). Gapped seams -- which the tracer separates for free -- have a dark
    voxel in the window and are excluded. Only blind contacts are the metric-relevant merge
    cases, so this is what hard-negative mining should oversample. Mirrors the blindness
    test in metrics._pair_is_blind, applied densely on the GPU."""
    import torch.nn.functional as F
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    contact = contact_map(inst, radius, device=str(dev))          # bool [Z,Y,X]
    img_t = torch.from_numpy(image.astype(np.float32)).to(dev)[None, None]
    k = 2 * radius + 1
    img_min = (-F.max_pool3d(-img_t, k, 1, radius))[0, 0].cpu().numpy()   # local min
    return contact & (img_min >= gap_intensity)


def build_coordinates(skel_lab: np.ndarray, part_vol: np.ndarray,
                      contact: np.ndarray, margin: int,
                      hardneg_radius: int, device="cuda") -> dict:
    """Assemble per-cube candidate centers on the skeleton.

    Returns arrays: z,y,x (int16), inst (int32), pbin (int8), hard (bool).
    Only skeleton voxels at least `margin` from every edge are kept (so a full
    FOV+delta fits). `hard` marks centers within `hardneg_radius` of a contact.
    """
    Z, Y, X = skel_lab.shape
    interior = np.zeros(skel_lab.shape, dtype=bool)
    interior[margin:Z - margin, margin:Y - margin, margin:X - margin] = True
    cand = (skel_lab > 0) & interior
    coords = np.argwhere(cand)
    if coords.shape[0] == 0:
        return dict(z=np.zeros(0, np.int16), y=np.zeros(0, np.int16),
                    x=np.zeros(0, np.int16), inst=np.zeros(0, np.int32),
                    pbin=np.zeros(0, np.int8), hard=np.zeros(0, bool))
    z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]
    inst = skel_lab[z, y, x].astype(np.int32)
    pbin = part_vol[z, y, x].astype(np.int8)

    # hard-neg: distance to nearest contact <= hardneg_radius
    if contact.any():
        cd = ndi_distance_to(contact)
        hard = cd[z, y, x] <= hardneg_radius
    else:
        hard = np.zeros(coords.shape[0], dtype=bool)
    return dict(z=z.astype(np.int16), y=y.astype(np.int16), x=x.astype(np.int16),
                inst=inst, pbin=pbin, hard=hard)


def ndi_distance_to(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi
    return ndi.distance_transform_edt(~mask)
