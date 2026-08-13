"""Arms test on labelled data: what do production caps and the barrier actually buy?

  EVAL   the training-eval path exactly as previously reported (fill cap 4000,
         eval seed spacing/cap) -- continuity with every number in the dossier
  PROD   production decode, barrier OFF  (fill 20000, uncapped grid-thinned seeds)
  BARR   production decode, barrier ON   (the committed-neighbour barrier)

Two datasets:
  * 6 labelled real 192^3 crops (GT instance labels; fiber = GT footprint so the
    arms measure SEPARATION, matching all prior numbers)
  * official 320^3 surfaces cubes (thread->body GT, band-restricted, mask-free
    harmonized input -- the production regime end to end)

Scored on merges FIRST (acceptance gate), then NERL, coverage, fragments,
truncated fills. A NERL gain that adds a merge is a rejection.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/root/ds_probe/pylibs")     # imagecodecs BEFORE tifffile

import dataclasses                # noqa: E402
import glob                       # noqa: E402
import json                       # noqa: E402
import os                         # noqa: E402

import numpy as np                # noqa: E402
import tifffile                   # noqa: E402
import torch                      # noqa: E402

sys.path.insert(0, "/root/surface_detection/FFN")
sys.path.insert(0, "/root")
from ffn import inference as I    # noqa: E402
from ffn import seeds as S        # noqa: E402
from ffn.metrics import erl as erl_metric                  # noqa: E402
from ffn.ctstats import air_papyrus_threshold              # noqa: E402
from scripts.evaluate import load_model, make_predict_fn   # noqa: E402
from inference.v1_separation import adjacent_pairs         # noqa: E402

from tracer.config import TracerConfig    # noqa: E402
from tracer.decode import decode_block    # noqa: E402
from tracer.normalize import harmonize    # noqa: E402

CORPUS = "/root/surf/data/synthfuse_corpus"
KAGGLE = "/root/ds_probe/kaggle"


def score(inst, out, pairs):
    e = erl_metric(inst, out)
    sizes = {g: int((inst == g).sum()) for g in np.unique(inst) if g}
    merged = 0
    for (a, b) in pairs:
        va, vb = out[inst == a], out[inst == b]
        sa = {int(x) for x, c in zip(*np.unique(va[va > 0], return_counts=True))
              if c >= 0.10 * sizes[a]}
        sb = {int(x) for x, c in zip(*np.unique(vb[vb > 0], return_counts=True))
              if c >= 0.10 * sizes[b]}
        merged += bool(sa & sb)
    fg = inst > 0
    cov = float((out[fg] > 0).mean()) if fg.any() else 0.0
    return dict(nerl=round(e["nerl"], 4), merges=merged, pairs=len(pairs),
                cov=round(cov, 3), frags=int(len(np.unique(out[out > 0]))))


def eval_arm(pf, cfg, ct, fiber):
    """The training-eval decode exactly as previously reported."""
    c = dataclasses.replace(cfg, consensus_scales=[1],
                            fill_max_steps=cfg.eval_fill_max_steps)
    dev = next(iter([torch.device("cuda:1" if torch.cuda.device_count() > 1
                                  else "cuda:0")]))
    gpu = torch.from_numpy(((ct.astype(np.float32) / 255.0 - 0.5) / 0.5)).to(dev)
    coords, _ = S.inference_seeds(fiber, thr=0.5, min_edt=c.inf_seed_min_edt,
                                  spacing=c.eval_seed_spacing)
    coords = coords[:c.eval_seed_cap]
    viable = I.screen_seeds(pf, gpu, coords, c) if len(coords) else np.zeros(0, bool)
    return I.segment_block(pf, gpu, coords, c, viable=viable)


def thread_to_body(ct, lab_raw):
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed
    surf = np.isin(lab_raw, [1, 255])
    band = np.isin(lab_raw, [0, 1, 255])
    pap = (ct >= air_papyrus_threshold(ct)) & band
    seeds, _ = ndi.label(surf, structure=np.ones((3, 3, 3), bool))
    cost = (255 - ndi.gaussian_filter(ct.astype(np.float32), 1.2)).astype(np.uint16)
    body = watershed(cost, markers=seeds, mask=pap)
    ids, cnts = np.unique(body[body > 0], return_counts=True)
    keep = ids[cnts >= 20000]
    remap = np.zeros(int(body.max()) + 1, np.int32)
    remap[keep] = np.arange(1, keep.size + 1)
    return remap[body], band


def main():
    tc = TracerConfig().load_anchors()
    dev = torch.device(tc.device)
    model, cfg = load_model(tc.ckpt, dev)
    prod_cfg = dataclasses.replace(cfg, fill_max_steps=tc.fill_max_steps,
                                   consensus_scales=[1])
    pf = make_predict_fn(model, prod_cfg, dev)
    print(f"ckpt={os.path.basename(tc.ckpt)}", flush=True)
    results = {}

    # ---------------- 192^3 labelled crops ----------------
    print("\n== 192^3 labelled crops (fiber = GT footprint) ==", flush=True)
    rows = {"EVAL": [], "PROD": [], "BARR": []}
    C = cfg.eval_crop
    for p in sorted(glob.glob(os.path.join(CORPUS, "labelsTr_inst", "real_*.tif")))[:6]:
        cid = os.path.basename(p)[:-4]
        inst = tifffile.imread(p).astype(np.int32)
        ct = tifffile.imread(p.replace("labelsTr_inst", "imagesTr")
                             .replace(".tif", "_0000.tif"))
        o = [(s - C) // 2 for s in inst.shape]
        ct = ct[o[0]:o[0] + C, o[1]:o[1] + C, o[2]:o[2] + C]
        inst = inst[o[0]:o[0] + C, o[1]:o[1] + C, o[2]:o[2] + C]
        fiber = (inst > 0).astype(np.float32)
        pairs = adjacent_pairs(inst)

        lab = eval_arm(pf, cfg, ct, fiber)
        rows["EVAL"].append(score(inst, lab, pairs))
        for arm, bar in (("PROD", False), ("BARR", True)):
            tc.barrier = bar
            lab, st = decode_block(pf, ct, prod_cfg, tc, fiber=fiber)
            r = score(inst, lab, pairs)
            r["trunc"] = st["truncated"]
            rows[arm].append(r)
        line = f"{cid[:26]:26s}"
        for arm in ("EVAL", "PROD", "BARR"):
            r = rows[arm][-1]
            line += (f" | {arm} n={r['nerl']:.3f} m={r['merges']}/{r['pairs']}"
                     f" f={r['frags']}")
        print(line, flush=True)
    results["gt192"] = rows

    # ---------------- official 320^3 cubes ----------------
    print("\n== official 320^3 cubes (mask-free harmonized, band-scored) ==", flush=True)
    rows3 = {"PROD": [], "BARR": []}
    for f in sorted(glob.glob(os.path.join(KAGGLE, "images", "*.tif"))):
        cid = os.path.basename(f)[:-4]
        ct_raw = tifffile.imread(f)
        inst, band = thread_to_body(ct_raw, tifffile.imread(f.replace("images", "labels")))
        if len(np.unique(inst[inst > 0])) < 4:
            continue
        pairs = adjacent_pairs(inst)
        ct = harmonize(ct_raw, tc.slab_bg, tc.slab_fg95) \
            if np.isfinite(tc.slab_bg) else ct_raw
        line = f"{cid[:26]:26s}"
        for arm, bar in (("PROD", False), ("BARR", True)):
            tc.barrier = bar
            lab, st = decode_block(pf, ct, prod_cfg, tc)   # mask-free fiber (production)
            lab[~band] = 0
            r = score(inst, lab, pairs)
            r["trunc"] = st["truncated"]
            rows3[arm].append(r)
            line += (f" | {arm} n={r['nerl']:.3f} m={r['merges']}/{r['pairs']}"
                     f" f={r['frags']} tr={r['trunc']}")
        print(line, flush=True)
    results["off320"] = rows3

    json.dump(results, open("/root/tracer_test_gt.json", "w"), indent=1)
    print("\n== POOLED ==")
    for ds, rr in results.items():
        for arm, lst in rr.items():
            if not lst:
                continue
            n = float(np.mean([r["nerl"] for r in lst]))
            m = sum(r["merges"] for r in lst)
            p = sum(r["pairs"] for r in lst)
            cv = float(np.mean([r["cov"] for r in lst]))
            fr = sum(r["frags"] for r in lst)
            tr = sum(r.get("trunc", 0) for r in lst)
            print(f"{ds:6s} {arm:4s}  NERL {n:.3f}  merges {m}/{p}  cov {cv:.3f}  "
                  f"frags {fr}  truncated {tr}")


if __name__ == "__main__":
    main()
