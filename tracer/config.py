"""Production tracer configuration.

Everything the decode needs beyond the model's own FFNConfig (which travels inside the
checkpoint and is authoritative for fov/delta/thresholds/gates -- we never override the
model's trained operating point here, only the DEPLOYMENT of it).
"""
from __future__ import annotations

import dataclasses
import json
import os


@dataclasses.dataclass
class TracerConfig:
    # ---- checkpoint --------------------------------------------------------
    ckpt: str = "/root/surf/ffn_run5/ckpt_last.pt"   # latest, by explicit decision

    # ---- blocking ----------------------------------------------------------
    # core+halo canvases. halo must cover fov/2 (=16) plus room for a fill to walk
    # back in from the face so mid-plane overlap evidence exists: 2 fov faces = 32,
    # plus delta-quantized approach slop. 48 keeps the canvas at core+96.
    block: int = 192
    halo: int = 48

    # ---- fills (production values; the 4000-step cap in the training-eval path is
    # an inline-eval economy that TRUNCATES sheets at 320^3 -- measured) ----------
    fill_max_steps: int = 20_000
    fill_batch_k: int = 64

    # ---- seeds -------------------------------------------------------------
    # NO rank cap. An EDT-ordered cap is not a subsample -- it deletes whole thin
    # regions (measured at 320^3: seed density 15/Mvox vs 28 at 192^3, NERL x0.13).
    # Density is controlled by grid thinning only. The spacing itself comes from the model's
    # FFNConfig (one source of truth); rescale/lam-rel modes scale it so PHYSICAL density is
    # constant across decode modes.
    seed_min_edt: float = 1.0

    # ---- committed-neighbour barrier --------------------------------------
    barrier: bool = True

    # ---- GEOMETRIC normalisation (tracer/rescale.py) ----------------------
    # The network is a single-scale filter bank tuned to the training wrap period
    # (lambda* ~ 15.5 vox); the S3/S4 cores sit at lambda ~ 10.3 with ZERO overlap with the
    # training distribution, and at that scale the model assigns the NEIGHBOURING wrap p=0.73.
    # Per-block: measure lambda from the image (structure-tensor normal + run-length; model-free,
    # ~2 s), upsample by s = lambda*/lambda (clamped), decode, map labels back. Measured on 2
    # Scroll-4 core cubes against the released winding meshes: recall 19.6->34.9% / 17.1->23.1%,
    # 0 merges introduced. The two grid-dependent scalars (fill budget x s^2, commit floor x s^3)
    # travel with the resample automatically.
    rescale: bool = True
    lam_star: float = 15.5            # target apparent period = the measured tuning-curve peak
    rescale_max_s: float = 2.0        # upper clamp; never downsample (lower clamp 1.0)
    rescale_min_s: float = 1.08       # below this, skip the resample (not worth s^3 compute)
    # Blocks whose s lands ABOVE rescale_max_s, or where the resample is skipped for compute, still
    # get LAMBDA-RELATIVE decode constants (delta <= lam/2, spacing/floor/budget ratios) -- the
    # geometry-only fallback. Blocks with unmeasurable lambda decode with stock constants.
    lam_relative_fallback: bool = True

    # ---- stitching ---------------------------------------------------------
    stitch_min_overlap: int = 40      # plane voxels; below this a match is noise
    # mutual-max only. There is deliberately NO "majority" or fractional-vote rule
    # here: label-space voting measured -42% NERL.

    # ---- normalization -----------------------------------------------------
    # Anchors of the TRAINING distribution (bg-median, fg-p95 of the Scroll-1 slab
    # the whole corpus was harmonized to). Filled from anchors.json at runtime;
    # values here are only a fallback and are overwritten by calibrate().
    anchors_path: str = "/root/tracer/anchors.json"
    slab_bg: float = float("nan")
    slab_fg95: float = float("nan")

    # ---- test-time augmentation (rot90 logit averaging). Implemented, default
    # off: single-checkpoint single-pass is the v1 baseline we measure first.
    tta_rot: bool = False

    # ---- runtime -----------------------------------------------------------
    device: str = "cuda:1"            # training owns priority; we ride GPU 1
    out_dtype: str = "int32"

    def load_anchors(self):
        if os.path.exists(self.anchors_path):
            a = json.load(open(self.anchors_path))
            self.slab_bg = float(a["bg"])
            self.slab_fg95 = float(a["fg95"])
        return self
