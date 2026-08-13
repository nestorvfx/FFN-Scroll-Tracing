#!/usr/bin/env python
"""Build a DEPLOYMENT-REGIME eval panel: winding labels for the Scroll-4 core cubes.

WHY. Every merge number this project has ever reported comes from Scroll-1-derived data at
lambda 14.5-18.8 and fg 0.32-0.36. The cubes we actually deploy on sit at lambda 9.4-11.9 and
fg ~0.62 -- ZERO overlap in wrap period. We have never measured an adjacent-wrap merge rate where
the model is used. Today's seed audit showed the seeding defect is density-gated, so the panel we
select checkpoints on is structurally blind to the regime that matters.

HOW (cheap route). The Scroll-4 eval cubes are already CT crops with known L1 coordinates
(manifest.json: z0,y0,x0 at 4.798um, 317^3 -> downsampled to 192^3 at 7.91um). The merged, traced
core surface (`meshes.zip`) supplies the WINDING reference. So we do not need the 178 GB volume:
for each cube, rasterise the merged surface inside its window, tag each surface voxel with its
column's winding number, and hand every papyrus voxel to its nearest surface voxel's winding
(the `voxelize/winding_assign.py` method -- validated on Scroll-1 GT at 0.996/0.999/0.990 agreement,
100% adjacent, 0 non-adjacent merges).

HONEST LIMITS, state them with any number this produces:
  * SUCCESS BIAS. Traced windings exist where the official pipeline SUCCEEDED, and the Scroll-4
    paper names compressed contacts as its own bottleneck #1. A merge rate measured here reads
    OPTIMISTIC relative to the untraced regions.
  * These are mesh-derived reference labels, not human annotation.
  * Cubes outside the merged surface's coverage get no labels and are skipped (reported).

Usage:
  python -m scripts.phase0_crossscroll build --cubes DIR --meshes GLOB --out DIR [--n 20]
  python -m scripts.phase0_crossscroll eval  --panel DIR --work DIR --ckpt PATH [--device cuda:0]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import tifffile
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

CUBE_L1 = 317          # source cube side at 4.798um that downsamples to 192 at 7.91um
CUBE_OUT = 192

_MERGED = {}


def merged_grid_and_winding(mesh_glob):
    """(Pg [H,W,3] in L1 x,y,z with NaN invalid, wind_col[W] winding per column).

    Ported from voxelize/winding_assign.py (the validated implementation): winding = unwrapped
    angle of each column's mean position about the umbilicus, divided by 2*pi.
    """
    if "P" in _MERGED:
        return _MERGED["P"], _MERGED["w"]
    ds = sorted(glob.glob(mesh_glob))
    if not ds:
        raise SystemExit(f"no merged surface matching {mesh_glob}")
    d = ds[0]
    X = tifffile.imread(os.path.join(d, "x.tif"))
    Y = tifffile.imread(os.path.join(d, "y.tif"))
    Z = tifffile.imread(os.path.join(d, "z.tif"))
    valid = (X > 0) & (Y > 0) & (Z > 0)
    Pg = np.stack([X, Y, Z], -1).astype(np.float32) / 2.0        # L0 (2.399um) -> L1 (4.798um)
    Pg[~valid] = np.nan
    Ys, Xs = Pg[..., 1], Pg[..., 0]
    inner = valid[:, :400]
    uy = float(np.nanmean(np.where(inner, Ys[:, :400], np.nan)))
    ux = float(np.nanmean(np.where(inner, Xs[:, :400], np.nan)))
    cnt = valid.sum(0).astype(np.float32)
    my = np.where(cnt > 0, np.where(valid, Ys, 0).sum(0) / np.maximum(cnt, 1), np.nan)
    mx = np.where(cnt > 0, np.where(valid, Xs, 0).sum(0) / np.maximum(cnt, 1), np.nan)
    ang = np.arctan2(my - uy, mx - ux)
    good = np.isfinite(ang)
    angf = np.interp(np.arange(Pg.shape[1]), np.where(good)[0], ang[good])
    theta = np.unwrap(angf)
    wind_col = ((theta - theta.min()) / (2 * np.pi)).astype(np.float32)
    _MERGED["P"], _MERGED["w"] = Pg, wind_col
    print(f"[mesh] {os.path.basename(d)} grid {Pg.shape[:2]} windings "
          f"{wind_col.min():.2f}..{wind_col.max():.2f}", flush=True)
    return Pg, wind_col


def _upsample_grid(P, factor):
    H, W, C = P.shape
    valid = np.isfinite(P).all(-1)
    ys = np.linspace(0, H - 1, max(1, int((H - 1) * factor) + 1))
    xs = np.linspace(0, W - 1, max(1, int((W - 1) * factor) + 1))
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    y0 = np.floor(gy).astype(int); x0 = np.floor(gx).astype(int)
    y1 = np.minimum(y0 + 1, H - 1); x1 = np.minimum(x0 + 1, W - 1)
    fy = (gy - y0)[..., None]; fx = (gx - x0)[..., None]
    ok = valid[y0, x0] & valid[y1, x0] & valid[y0, x1] & valid[y1, x1]
    return ((P[y0, x0] * (1 - fy) + P[y1, x0] * fy) * (1 - fx)
            + (P[y0, x1] * (1 - fy) + P[y1, x1] * fy) * fx)[ok]


def label_cube(ct192, z0, y0, x0, mesh_glob, max_grow=13, up=6.0, fg_pct=None):
    """Winding labels for ONE eval cube. Returns (labels192 int32, stats) or None if uncovered.

    The cube is a 317^3 L1 window area-averaged to 192^3. We rasterise the surface in the L1
    window, then scale coordinates by 192/317 into the output frame -- so labels land on the same
    grid as the CT the FFN sees.
    """
    from ffn.ctstats import air_papyrus_threshold
    Pg, wind_col = merged_grid_and_winding(mesh_glob)
    S = CUBE_L1
    z, y, x = Pg[..., 2], Pg[..., 1], Pg[..., 0]
    inw = ((z >= z0 - 6) & (z < z0 + S + 6) & (y >= y0 - 6) & (y < y0 + S + 6)
           & (x >= x0 - 6) & (x < x0 + S + 6))
    if not inw.any():
        return None
    rmin, rmax = np.where(inw.any(1))[0][[0, -1]]
    cmin, cmax = np.where(inw.any(0))[0][[0, -1]]
    sub = Pg[rmin:rmax + 1, cmin:cmax + 1]
    subw = np.broadcast_to(wind_col[cmin:cmax + 1][None, :], sub.shape[:2]).astype(np.float32)
    up4 = _upsample_grid(np.concatenate([sub, subw[..., None]], -1), up)
    Ps, Wc = up4[:, :3], up4[:, 3]
    sc = CUBE_OUT / float(S)                       # L1 window -> 192 output grid
    qx = np.rint((Ps[:, 0] - x0) * sc).astype(np.int64)
    qy = np.rint((Ps[:, 1] - y0) * sc).astype(np.int64)
    qz = np.rint((Ps[:, 2] - z0) * sc).astype(np.int64)
    inb = ((qz >= 0) & (qz < CUBE_OUT) & (qy >= 0) & (qy < CUBE_OUT)
           & (qx >= 0) & (qx < CUBE_OUT))
    if inb.sum() < 200:
        return None
    wmark = np.full((CUBE_OUT,) * 3, -1.0, np.float32)
    wmark[qz[inb], qy[inb], qx[inb]] = Wc[inb]
    surf = wmark >= 0
    thr = air_papyrus_threshold(ct192)
    fg = ct192 >= thr
    grow = max_grow * sc                            # max_grow is in L1 voxels
    d, idx = ndi.distance_transform_edt(~surf, return_indices=True)
    w = np.where(fg & (d <= grow), wmark[idx[0], idx[1], idx[2]], -1.0)
    lab = np.where(w >= 0, np.rint(w).astype(np.int32) + 1, 0).astype(np.int32)
    ids, cnts = np.unique(lab[lab > 0], return_counts=True)
    for i, c in zip(ids, cnts):                     # drop specks
        if c < 500:
            lab[lab == i] = 0
    ids = [int(i) for i in np.unique(lab) if i]
    stats = dict(n_wraps=len(ids), labelled_frac=float((lab > 0).sum()) / max(int(fg.sum()), 1),
                 fg_frac=float(fg.mean()), thr=float(thr))
    return lab, stats


def cmd_build(a):
    man = json.load(open(os.path.join(a.cubes, "manifest.json")))
    os.makedirs(os.path.join(a.out, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "labelsTr_inst"), exist_ok=True)
    kept, skipped = [], 0
    for rec in man:
        if len(kept) >= a.n:
            break
        name = rec["name"]
        f = os.path.join(a.cubes, "imagesTr", f"{name}_0000.tif")
        if not os.path.exists(f):
            continue
        ct = tifffile.imread(f)
        out = label_cube(ct, rec["z0"], rec["y0"], rec["x0"], a.meshes)
        if out is None:
            skipped += 1
            continue
        lab, st = out
        if st["n_wraps"] < 2 or st["labelled_frac"] < 0.2:
            skipped += 1
            continue
        tifffile.imwrite(os.path.join(a.out, "imagesTr", f"{name}_0000.tif"), ct)
        tifffile.imwrite(os.path.join(a.out, "labelsTr_inst", f"{name}.tif"),
                         lab.astype(np.uint16))
        kept.append(dict(name=name, **st))
        print(f"  {name:38s} wraps {st['n_wraps']:3d}  labelled {st['labelled_frac']:.2f} "
              f"fg {st['fg_frac']:.2f}", flush=True)
    json.dump(kept, open(os.path.join(a.out, "panel.json"), "w"), indent=1)
    print(f"\n[build] kept {len(kept)} cubes, skipped {skipped} (no surface coverage / too few "
          f"wraps). Panel -> {a.out}")
    if kept:
        print(f"[build] median wraps/cube {np.median([k['n_wraps'] for k in kept]):.0f}, "
              f"median labelled fraction {np.median([k['labelled_frac'] for k in kept]):.2f}")


def cmd_eval(a):
    import dataclasses
    import torch
    from ffn.config import FFNConfig
    from ffn import inference as I
    from ffn import seeds as S
    from ffn.metrics import adjacency_pairs, erl as erl_metric
    from ffn.inline_eval import fair_seeds
    from scripts.phase0_decode import make_predict
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    cfg0 = FFNConfig.load(os.path.join(a.work, "config.json"))
    pf, step = make_predict(cfg0, a.ckpt, dev)
    files = sorted(glob.glob(os.path.join(a.panel, "labelsTr_inst", "*.tif")))[:a.n]
    print(f"ckpt {step} | DEPLOYMENT-REGIME panel: {len(files)} Scroll-4 core cubes", flush=True)
    print(f"{'freeze':>7s} {'nerl':>7s} {'merge':>7s} {'pairs':>6s} {'cov':>6s} {'inst':>5s}",
          flush=True)
    for fp in [float(x) for x in a.freeze.split(",")]:
        cfg = dataclasses.replace(cfg0, fill_max_steps=cfg0.eval_fill_max_steps,
                                  agglo_enabled=False, pom_ratchet_freeze_p=fp)
        rq = rp = 0.0
        nm = npair = cov_n = cov_d = ninst = 0
        for f in files:
            gt = tifffile.imread(f).astype(np.int32)
            ct = tifffile.imread(f.replace("labelsTr_inst", "imagesTr").replace(".tif",
                                                                                "_0000.tif"))
            c = cfg.eval_crop
            o = [(s - c) // 2 for s in gt.shape]
            gt = gt[o[0]:o[0] + c, o[1]:o[1] + c, o[2]:o[2] + c]
            ct = ct[o[0]:o[0] + c, o[1]:o[1] + c, o[2]:o[2] + c]
            co, _ = S.instance_inference_seeds(gt, min_edt=cfg.inf_seed_min_edt,
                                               spacing=cfg.eval_seed_spacing)
            co = fair_seeds(co, gt, cfg.eval_seed_cap, getattr(cfg, "eval_seed_per_inst", 0))
            g = torch.from_numpy(((ct.astype(np.float32) / 255.0 - 0.5) / 0.5)).to(dev)
            with torch.no_grad():
                pred = I.segment_block(pf, g, co, cfg, K=128)
            e = erl_metric(gt, pred)
            rq += e["sum_runsq"]; rp += e["sum_perfsq"]
            sizes = {int(i): int((gt == i).sum()) for i in np.unique(gt) if i}
            for (i, j) in adjacency_pairs(gt, cfg.adjacency_radius):
                npair += 1
                for L in np.unique(pred[(gt == i) & (pred > 0)]):
                    if L and (pred[gt == i] == L).sum() / max(sizes[i], 1) >= cfg.merge_theta \
                       and (pred[gt == j] == L).sum() / max(sizes[j], 1) >= cfg.merge_theta:
                        nm += 1
                        break
            cov_n += int(((gt > 0) & (pred > 0)).sum()); cov_d += int((gt > 0).sum())
            ninst += len(np.unique(pred[pred > 0]))
        print(f"{fp:7.2f} {rq/max(rp,1e-9):7.4f} {nm/max(npair,1):7.4f} {npair:6d} "
              f"{cov_n/max(cov_d,1):6.3f} {ninst:5d}", flush=True)
    print("\nSUCCESS BIAS: these labels exist where the official tracing SUCCEEDED; the Scroll-4 "
          "paper names compressed contacts as its own bottleneck. Read this merge rate as a "
          "LOWER BOUND on the deployment regime.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--cubes", required=True)
    b.add_argument("--meshes", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--n", type=int, default=20)
    e = sub.add_parser("eval")
    e.add_argument("--panel", required=True)
    e.add_argument("--work", required=True)
    e.add_argument("--ckpt", required=True)
    e.add_argument("--n", type=int, default=20)
    e.add_argument("--freeze", default="0.5,0.7")
    e.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    (cmd_build if a.cmd == "build" else cmd_eval)(a)


if __name__ == "__main__":
    main()
