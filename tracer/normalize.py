"""Input normalization: make ANY volume look like the training distribution.

The training corpus was not raw CT. Every real crop went through ingest_real.harmonize():
a linear map sending (its bg-median, its fg-p95) onto the anchors of the Scroll-1 training
slab, then uint8, then the model-space map (x/255 - 0.5)/0.5. A volume decoded WITHOUT that
harmonization (e.g. the official 320^3 cubes with the fixed /255 map) is the only material
the model ever sees off-distribution -- so production applies the exact same construction,
made mask-free:

  fg  := voxels >= EM air/papyrus threshold  (ffn.ctstats.air_papyrus_threshold; the
         ingest used the GT mask here, which production does not have)
  map := linear (bg_med, fg_p95) -> (slab_bg, slab_fg95), clip to [0,255]

`calibrate()` computes the slab anchors ONCE from the actual training slab and persists
them, so the deployment constant has provenance instead of being a magic number.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, "/root/surface_detection/FFN")
from ffn.ctstats import air_papyrus_threshold  # noqa: E402


def calibrate(slab_dir: str, out_path: str) -> dict:
    """Anchors from the training slab (same definition as ingest_real.slab_anchors)."""
    import nrrd
    vol, _ = nrrd.read(os.path.join(slab_dir, "volume.nrrd"))
    truth, _ = nrrd.read(os.path.join(slab_dir, "truth.nrrd"))
    fg = truth > 0
    a = {"bg": float(np.median(vol[~fg])), "fg95": float(np.percentile(vol[fg], 95)),
         "slab": slab_dir}
    json.dump(a, open(out_path, "w"))
    return a


def harmonize(vol: np.ndarray, slab_bg: float, slab_fg95: float) -> np.ndarray:
    """Mask-free ingest_real.harmonize: raw CT (uint8/16) -> uint8 in slab units."""
    v = vol.astype(np.float32)
    thr = air_papyrus_threshold(vol)
    fg = vol >= thr
    if not fg.any() or fg.all():          # degenerate: identity in slab units
        return np.clip(v, 0, 255).astype(np.uint8)
    bg_med = float(np.median(v[~fg]))
    fg95 = float(np.percentile(v[fg], 95))
    a = (slab_fg95 - slab_bg) / max(1.0, fg95 - bg_med)
    out = (v - bg_med) * a + slab_bg
    return np.clip(out, 0, 255).astype(np.uint8)


def to_model(vol_u8: np.ndarray) -> np.ndarray:
    """Slab-unit uint8 -> model input float32. The one true map (matches training)."""
    return (vol_u8.astype(np.float32) / 255.0 - 0.5) / 0.5
