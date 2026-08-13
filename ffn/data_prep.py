"""Per-cube preprocessing: turn a synth cube's instance labels into an FFN
candidate-center index (interior/medial seeds + partition bins + hard-neg flags).

GPU-native and point-based: one max/min-pool gives the one-lamina-deep interior
seeds, the partition SAT and contact dilation run on the GPU, and we only keep a
capped, edge-margined subsample of candidates. ~1-2 s/cube on a 5090 vs ~100 s
for the CPU EDT/skeletonize path. See DESIGN.md 5.4, 5.5, 6.3.
"""
from __future__ import annotations

import os
import numpy as np
import tifffile
import torch
import torch.nn.functional as F

from .config import FFNConfig
from . import partitions as PT


def process_cube(corpus_dir: str, cube_id: str, cfg: FFNConfig,
                 device: str = "cuda", max_cands: int = 20000, seed: int = 0) -> dict:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    inst_np = tifffile.imread(os.path.join(corpus_dir, "labelsTr_inst",
                                           f"{cube_id}.tif")).astype(np.int32)
    Z, Y, X = inst_np.shape
    inst_t = torch.from_numpy(inst_np).to(dev)
    # image (uint8) is needed for blind-contact hard-negative targeting (AUDIT FIX 3)
    img_np = tifffile.imread(os.path.join(corpus_dir, "imagesTr",
                                          f"{cube_id}_0000.tif")).astype(np.float32)

    # instance sizes -> drop small instances
    ids = torch.unique(inst_t)
    counts = torch.bincount(inst_t.reshape(-1).clamp(min=0),
                            minlength=int(inst_t.max()) + 1)
    keep_lbl = (counts >= cfg.min_instance_size)
    keep_lbl[0] = False

    interior = PT.interior_seeds_gpu(inst_t)                     # bool [Z,Y,X]
    # edge margin so a full FOV+delta fits
    m = cfg.margin
    mask_margin = torch.zeros_like(interior)
    mask_margin[m:Z - m, m:Y - m, m:X - m] = True
    cand = interior & mask_margin & keep_lbl[inst_t.clamp(min=0)]

    coords = torch.nonzero(cand, as_tuple=False)                 # [M,3]
    if coords.shape[0] == 0:
        return _empty(inst_np.shape, int(keep_lbl.sum()))
    # subsample
    if coords.shape[0] > max_cands:
        g = torch.Generator(device=dev); g.manual_seed(seed)
        sel = torch.randperm(coords.shape[0], generator=g, device=dev)[:max_cands]
        coords = coords[sel]
    labels = inst_t[coords[:, 0], coords[:, 1], coords[:, 2]]

    # partition bin at candidate points (per-instance SAT point-query)
    pbin = _partition_at(inst_t, coords, labels, cfg, dev)

    # hard-neg: candidate within hardneg_radius of an inter-sheet contact.
    # AUDIT FIX 3: target BLIND contacts (no CT gap) -- the metric-relevant merge cases --
    # instead of every contact (which the dense synth corpus flags on ~91% of seeds).
    if getattr(cfg, "blind_hardneg", True):
        fg_mean = float(img_np[inst_np > 0].mean()) if (inst_np > 0).any() else 0.0
        gap_intensity = cfg.gap_intensity_frac * fg_mean
        contact = PT.blind_contact_map(inst_np, img_np, cfg.contact_radius,
                                       gap_intensity, device=str(dev))
    else:
        contact = PT.contact_map(inst_np, cfg.contact_radius, device=str(dev))
    n_contact = int(contact.sum())
    contact_t = torch.from_numpy(contact).to(dev).float()[None, None]
    r = cfg.hardneg_radius
    dil = F.max_pool3d(contact_t, 2 * r + 1, 1, r)[0, 0] > 0
    hard = dil[coords[:, 0], coords[:, 1], coords[:, 2]]

    return dict(
        z=coords[:, 0].to(torch.int16).cpu().numpy(),
        y=coords[:, 1].to(torch.int16).cpu().numpy(),
        x=coords[:, 2].to(torch.int16).cpu().numpy(),
        inst=labels.to(torch.int32).cpu().numpy(),
        pbin=pbin.to(torch.int8).cpu().numpy(),
        hard=hard.cpu().numpy(),
        n_inst=int(keep_lbl.sum()), shape=inst_np.shape, n_contact=n_contact)


def _partition_at(inst_t, coords, labels, cfg: FFNConfig, dev) -> torch.Tensor:
    """Fill-fraction partition bin at each candidate: per instance, box-count of
    same-label voxels within lom_radius / box volume, quantized into 17 bins."""
    r = cfg.lom_radius
    box_vol = float((2 * r + 1) ** 3)
    frac = torch.zeros(coords.shape[0], device=dev)
    for lbl in torch.unique(labels).tolist():
        m = labels == lbl
        cnt = PT.integral_box_sum((inst_t == lbl).float(), r)
        c = coords[m]
        frac[m] = cnt[c[:, 0], c[:, 1], c[:, 2]] / box_vol
    thr = torch.tensor(cfg.partition_bins[1:-1], device=dev)
    return torch.bucketize(frac, thr)


def _empty(shape, n_inst):
    z = np.zeros(0, np.int16)
    return dict(z=z, y=z.copy(), x=z.copy(), inst=np.zeros(0, np.int32),
               pbin=np.zeros(0, np.int8), hard=np.zeros(0, bool),
               n_inst=n_inst, shape=shape, n_contact=0)
