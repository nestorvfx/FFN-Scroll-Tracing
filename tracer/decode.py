"""Per-block decode: lambda-normalise -> seeds -> viability screen -> serial barrier fills.

The unit of work is one core+halo canvas that fits on GPU whole. Everything full-volume here is
CANVAS-sized, so scroll scale is bounded by block size, not volume size.

GEOMETRIC NORMALISATION (the fix for compact scrolls). The network is a single-scale filter bank
tuned to the training wrap period (~15.5 vox); the S3/S4 cores sit at ~10.3 with zero overlap with
the training distribution. Each canvas therefore:

  1. measures its own wrap period lambda from the image (model-free, ~2 s);
  2. if lambda is off-target, UPSAMPLES by s = lam_star/lambda (clamped) so the material appears at
     the scale every voxel-unit constant -- ERF, delta, gate face, seed spacing -- was built for,
     decodes there, and maps labels back to the native grid;
  3. if the resample is skipped (unmeasurable, marginal, or clamped), falls back to LAMBDA-RELATIVE
     decode constants (delta <= lambda/2 etc.) -- fixing the decode-geometry half only.

Measured (2 Scroll-4 core cubes, released winding meshes as GT): recall 19.6 -> 34.9% and
17.1 -> 23.1%, zero merges introduced. The same mechanism, capped-seed variant, took a controlled
compact-Scroll-1 experiment from merge 0.317 back to 0.024 (native reads 0.073).
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "/root/surface_detection/FFN")
from ffn import inference as I     # noqa: E402  (screen_seeds -- exactness argument holds
#                                    with a barrier: it can only shrink growth, so a seed
#                                    non-viable without it is non-viable with it)
from ffn import seeds as S         # noqa: E402
from ffn.ctstats import air_papyrus_threshold  # noqa: E402

from .fill import flood_fill, tta_wrap
from .normalize import to_model
from .rescale import (measure_lambda, scale_for, resample, labels_back,
                      scaled_cfg, lambda_relative_cfg)


def _plan_scale(canvas_u8: np.ndarray, cfg, tc):
    """Decide how this canvas is decoded: (work_volume, cfg', s, lam, mode).

    mode is one of 'rescale' / 'lam-rel' / 'native'. The seed geometry (spacing, min_edt) scales
    with s so PHYSICAL seed density is held constant -- otherwise a rescaled block would be seeded
    s^3 more densely than a native one and the two regimes would not be comparable.
    """
    if not getattr(tc, "rescale", False):
        return canvas_u8, cfg, 1.0, 0.0, "native"
    lam = measure_lambda(canvas_u8)
    if lam <= 0:
        return canvas_u8, cfg, 1.0, lam, "native"
    s = scale_for(lam, tc.lam_star, lo=1.0, hi=tc.rescale_max_s)
    if s >= tc.rescale_min_s:
        return resample(canvas_u8, s), scaled_cfg(cfg, s), s, lam, "rescale"
    if getattr(tc, "lam_relative_fallback", True) and lam < tc.lam_star / tc.rescale_min_s:
        return canvas_u8, lambda_relative_cfg(cfg, lam, tc.lam_star), 1.0, lam, "lam-rel"
    return canvas_u8, cfg, 1.0, lam, "native"


def decode_block(predict_fn, canvas_u8: np.ndarray, cfg, tc, fiber: np.ndarray | None = None):
    """Segment one canvas (uint8, already in slab units).

    Returns (labels int32 on the NATIVE canvas grid, stats dict). `fiber`: optional externally
    supplied foreground probability on the NATIVE grid (e.g. GT footprint in controlled tests);
    default is the mask-free EM threshold, matching production.
    """
    dev = torch.device(tc.device)
    pf = tta_wrap(predict_fn, tc.tta_rot)

    work, wcfg, s, lam, mode = _plan_scale(canvas_u8, cfg, tc)
    image_gpu = torch.from_numpy(to_model(work)).to(dev)
    Z, Y, X = work.shape

    if fiber is None:
        fib = (work >= air_papyrus_threshold(work)).astype(np.float32)
    else:                                   # externally supplied, native grid -> follow the plan
        fib = resample((fiber > 0.5).astype(np.uint8) * 255, s).astype(np.float32) / 255.0 \
            if s > 1.0 else fiber
    coords, _ = S.inference_seeds(fib, thr=0.5,
                                  min_edt=tc.seed_min_edt * s,
                                  spacing=max(2, int(round(wcfg.seed_spacing * s))))
    # NO rank cap -- grid thinning is the only density control (a rank cap deletes whole thin
    # regions; measured at 320^3).
    viable = I.screen_seeds(pf, image_gpu, coords, wcfg) if len(coords) else \
        np.zeros(0, bool)

    committed = np.zeros((Z, Y, X), dtype=np.int32)
    barrier_t = torch.zeros((Z, Y, X), dtype=torch.bool, device=dev) if tc.barrier else None
    pom_buf = torch.empty((Z, Y, X), dtype=torch.float32, device=dev)
    visited_buf = np.zeros((Z, Y, X), dtype=np.int32)
    gen = 0
    next_label = 1
    rr = max(1, int(round(wcfg.seed_reject_radius * s)))
    fills = []
    for si in range(len(coords)):
        if not viable[si]:
            continue
        z, y, x = (int(v) for v in coords[si])
        if committed[z, y, x] != 0:
            continue
        z0, z1 = max(0, z - rr), min(Z, z + rr + 1)
        y0, y1 = max(0, y - rr), min(Y, y + rr + 1)
        x0, x1 = max(0, x - rr), min(X, x + rr + 1)
        if committed[z0:z1, y0:y1, x0:x1].any():
            continue
        gen += 1
        st = {}
        mask = flood_fill(pf, image_gpu, (z, y, x), wcfg, K=tc.fill_batch_k,
                          claimed=committed, barrier=barrier_t,
                          pom_buf=pom_buf, visited_buf=visited_buf, visited_gen=gen,
                          stats=st)
        take = mask & (committed == 0)
        if take.sum() < wcfg.min_instance_size // 4:
            continue
        if st.get("steps", 1) < getattr(wcfg, "min_fov_steps", 1):
            continue
        committed[take] = next_label
        if barrier_t is not None:
            idx = np.argwhere(take)
            barrier_t[torch.as_tensor(idx[:, 0], device=dev),
                      torch.as_tensor(idx[:, 1], device=dev),
                      torch.as_tensor(idx[:, 2], device=dev)] = True
        fills.append(dict(label=next_label, seed=[z, y, x],
                          vox=int(take.sum()), steps=int(st.get("steps", 0)),
                          truncated=bool(st.get("truncated", False))))
        next_label += 1
        # PROGRESS. A block emits nothing until it finishes, so a long decode is
        # indistinguishable from a hung one from the outside -- and on a slow or
        # off-distribution volume that is exactly when you need to know. Cheap:
        # one line per 25 committed objects.
        if next_label % 25 == 0:
            done_vox = int((committed != 0).sum())
            print(f"    ...{next_label - 1} objects, {done_vox} vox committed, "
                  f"seed {si + 1}/{len(coords)}, "
                  f"trunc {sum(1 for f in fills if f['truncated'])}", flush=True)
    del image_gpu, pom_buf, barrier_t
    torch.cuda.empty_cache()
    labels = labels_back(committed, s, canvas_u8.shape) if s > 1.0 else committed
    stats = dict(seeds=int(len(coords)), viable=int(viable.sum()),
                 objects=next_label - 1,
                 truncated=sum(1 for f in fills if f["truncated"]),
                 lam=round(float(lam), 2), s=round(float(s), 2), mode=mode,
                 fills=fills)
    return labels, stats
