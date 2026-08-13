#!/usr/bin/env python3
"""TSTR (Train-on-Synthetic, Test-on-Real) DUAL-METRIC harness. Does training on the synthetic fused corpus make
the model SEPARATE densely-packed sheets in REAL data WITHOUT over-splitting distinct ones? Evaluate a model on the
held-out REAL instance cubes (never trained on) and report BOTH failure directions plus boundary quality:

  MERGER RATE  : a GT sheet whose dominant pred-fiber component also dominates >=1 OTHER GT sheet (sheets fused).  v
  SPLIT  RATE  : a GT sheet broken into >=2 substantial pred-fiber components (one sheet fragmented).             ^
  VOI(merge)   : H(GT | pred)  -- high when pred merges GT sheets (under-segmentation).
  VOI(split)   : H(pred | GT)  -- high when pred fragments GT sheets (over-segmentation).
  SURF-DICE@t  : boundary-tolerant surface Dice at tolerance t voxels (regime-agnostic boundary quality).
  COMPOSITE    : interim panoptic-ish score (higher=better); Betti TopoScore is deferred to an on-demand build.

Merger and split are the two halves of the SAME tradeoff -- reading either alone is misleading, which is the whole
point of this harness. Compare medial_059 (baseline) vs the Dataset201-trained model; pass --vs <baseline.json> to
print a delta table.

Usage: python tstr_eval.py --model_dir <.../TRAINER__plans__3d_fullres> --fold 0 --ckpt checkpoint_best.pth \
                           --test_list <test_cubes.txt> --cubes_root <instance_cubes_dir> --out result.json \
                           [--gpu 0] [--label medial059] [--vs baseline.json]
"""
import os, glob, json, argparse, numpy as np
from scipy import ndimage as ndi

CONN = np.ones((3, 3, 3))           # 26-connectivity for instance recovery


def load_cube(d):
    import nrrd
    vol = np.asarray(nrrd.read(glob.glob(d + '/*_volume.nrrd')[0])[0]).astype(np.float32)
    msk = np.asarray(nrrd.read(glob.glob(d + '/*_mask.nrrd')[0])[0])
    lo, hi = np.percentile(vol, 0.5), np.percentile(vol, 99.5)   # SAME normalization as training
    vol = np.clip((vol - lo) / (hi - lo + 1e-6) * 255.0, 0, 255)
    return vol, msk


def build_predictor(model_dir, fold, ckpt, gpu=0):
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                        device=torch.device(f'cuda:{gpu}'), verbose=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(model_dir, use_folds=(fold,), checkpoint_name=ckpt)
    return p


def predict_fiber(pred, vol):
    ret = pred.predict_single_npy_array(vol[None].astype(np.float32), {'spacing': (1, 1, 1)}, None, None, False)
    seg = ret[0] if isinstance(ret, (list, tuple)) else ret
    return np.asarray(seg) == 1


def instance_stats(pred_fiber, gt_inst, min_sheet=2000, frag_frac=0.15, frag_min=500, pred_cc=None):
    """Both halves of the tradeoff in one pass over pred connected-components.
    MERGER: pred CC (the one each sheet maps to) that covers >=2 GT sheets.
    SPLIT : a GT sheet covered by >=2 pred CCs each holding a substantial share of the sheet's predicted fiber."""
    if pred_cc is None:                                          # reuse a precomputed CC labeling when given (eval dedup)
        pred_cc, _ = ndi.label(pred_fiber, structure=CONN)
    gt_ids = [int(i) for i in np.unique(gt_inst) if i != 0 and (gt_inst == i).sum() >= min_sheet]
    comp_to_gts, split_sheets, covered = {}, 0, 0
    for g in gt_ids:
        comps = pred_cc[(gt_inst == g) & pred_fiber]
        comps = comps[comps > 0]
        if comps.size == 0:
            continue                                            # sheet missed entirely (not a merge or split)
        covered += 1
        bc = np.bincount(comps)
        dom = int(bc.argmax())
        comp_to_gts.setdefault(dom, set()).add(g)               # the pred component this sheet maps to (merge side)
        thr = max(frag_min, frag_frac * comps.size)             # substantial fragment of THIS sheet's pred fiber
        if int((bc >= thr).sum()) >= 2:                         # >=2 substantial pred pieces -> sheet fragmented
            split_sheets += 1
    merged_sheets = sum(len(s) for s in comp_to_gts.values() if len(s) >= 2)
    return dict(n_sheets=len(gt_ids), n_covered=covered,
                merged_sheets=merged_sheets, merger_rate=round(merged_sheets / max(1, covered), 4),
                merged_groups=sum(1 for s in comp_to_gts.values() if len(s) >= 2),
                split_sheets=split_sheets, split_rate=round(split_sheets / max(1, covered), 4))


def voi(gt_inst, pred_fiber, min_sheet=2000, pred_cc=None):
    """Variation of information over voxels in (GT sheets) U (pred fiber). Returns (voi_merge, voi_split) in bits:
    voi_merge=H(GT|pred) (pred under-segments/merges), voi_split=H(pred|GT) (pred over-segments/splits)."""
    if pred_cc is None:
        pred_cc, _ = ndi.label(pred_fiber, structure=CONN)
    keep = np.zeros_like(gt_inst)
    for i in np.unique(gt_inst):
        if i != 0 and (gt_inst == i).sum() >= min_sheet:
            keep[gt_inst == i] = i
    mask = (keep > 0) | pred_fiber
    a = keep[mask].astype(np.int64)                             # GT labels (0 = bg)
    b = pred_cc[mask].astype(np.int64)                          # pred labels (0 = bg)
    n = a.size
    if n == 0:
        return 0.0, 0.0
    key = a * (b.max() + 1) + b
    _, jc = np.unique(key, return_counts=True)
    pj = jc / n
    _, ac = np.unique(a, return_counts=True); pa = ac / n
    _, bc = np.unique(b, return_counts=True); pb = bc / n
    Hj = -(pj * np.log2(pj)).sum(); Ha = -(pa * np.log2(pa)).sum(); Hb = -(pb * np.log2(pb)).sum()
    return round(float(Hj - Hb), 4), round(float(Hj - Ha), 4)   # H(A|B)=merge, H(B|A)=split


def fg_dice(gt_inst, pred_fiber):
    """Binary foreground (fiber) Dice -- the SAME metric family as medial_059's official pseudo-dice (~0.60), so
    our model is comparable to SOTA on identical data. NOTE: GT instance masks can be SPARSE (unlabeled real fiber
    exists), which biases the ABSOLUTE value low for BOTH models equally -> read the head-to-head delta, not the
    absolute vs 0.60. The 0.60 anchor is medial_059's published pseudo-dice on its own Dataset059 val."""
    gt = gt_inst > 0
    inter = float((gt & pred_fiber).sum()); s = float(gt.sum() + pred_fiber.sum())
    return round(2.0 * inter / s, 4) if s > 0 else 0.0


def surface_dice(gt_inst, pred_fiber, tau=2):
    gt = gt_inst > 0
    if gt.sum() == 0 or pred_fiber.sum() == 0:
        return 0.0
    gs = gt ^ ndi.binary_erosion(gt)
    ps = pred_fiber ^ ndi.binary_erosion(pred_fiber)
    if gs.sum() == 0 or ps.sum() == 0:
        return 0.0
    d_to_p = ndi.distance_transform_edt(~ps)
    d_to_g = ndi.distance_transform_edt(~gs)
    cov = (d_to_p[gs] <= tau).sum() + (d_to_g[ps] <= tau).sum()
    return round(float(cov) / float(gs.sum() + ps.sum()), 4)


def composite(merger_rate, split_rate, surf_dice, voi_total):
    """Interim higher=better score. Boundary quality + topology (low VOI) + both failure directions explicit.
    Betti TopoScore deferred to on-demand; reweight when it lands."""
    return round(0.35 * surf_dice + 0.35 * (1.0 / (1.0 + 0.3 * voi_total))
                 + 0.15 * (1.0 - merger_rate) + 0.15 * (1.0 - split_rate), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True); ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--ckpt", default="checkpoint_best.pth")     # best is what exists mid-run; final only at the end
    ap.add_argument("--test_list", required=True); ap.add_argument("--cubes_root", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tau", type=int, default=2); ap.add_argument("--label", default=None)
    ap.add_argument("--vs", default=None, help="path to a previous result.json to print a delta table")
    a = ap.parse_args()
    cubes = [c.strip() for c in open(a.test_list) if c.strip()]
    pred = build_predictor(a.model_dir, a.fold, a.ckpt, a.gpu)
    per_cube = []
    agg = dict(merged=0, split=0, covered=0, vm=0.0, vs=0.0, sd=0.0, dsc=0.0, n=0)
    for cid in cubes:
        d = os.path.join(a.cubes_root, cid)
        if not glob.glob(d + '/*_volume.nrrd'):
            continue
        vol, gt = load_cube(d)
        pf = predict_fiber(pred, vol)
        pred_cc, _ = ndi.label(pf, structure=CONN)              # label ONCE; share across instance_stats + voi
        s = instance_stats(pf, gt, pred_cc=pred_cc)
        vm, vs = voi(gt, pf, pred_cc=pred_cc); sd = surface_dice(gt, pf, a.tau); dsc = fg_dice(gt, pf)
        s.update(voi_merge=vm, voi_split=vs, surf_dice=sd, fg_dice=dsc,
                 composite=composite(s['merger_rate'], s['split_rate'], sd, vm + vs), cube=cid)
        per_cube.append(s)
        agg['merged'] += s['merged_sheets']; agg['split'] += s['split_sheets']; agg['covered'] += s['n_covered']
        agg['vm'] += vm; agg['vs'] += vs; agg['sd'] += sd; agg['dsc'] += dsc; agg['n'] += 1
        print(f"{cid}: sheets {s['n_sheets']} cov {s['n_covered']} | fgDice {dsc} | MERGE {s['merger_rate']} "
              f"SPLIT {s['split_rate']} | VOI(m/s) {vm}/{vs} | sDice {sd} | C {s['composite']}", flush=True)
    n = max(1, agg['n']); cov = max(1, agg['covered'])
    O = dict(fg_dice=round(agg['dsc'] / n, 4),
             merger_rate=round(agg['merged'] / cov, 4), split_rate=round(agg['split'] / cov, 4),
             voi_merge=round(agg['vm'] / n, 4), voi_split=round(agg['vs'] / n, 4), surf_dice=round(agg['sd'] / n, 4))
    O['composite'] = composite(O['merger_rate'], O['split_rate'], O['surf_dice'], O['voi_merge'] + O['voi_split'])
    res = dict(label=a.label or os.path.basename(a.model_dir), model_dir=a.model_dir, ckpt=a.ckpt,
               n_cubes=agg['n'], total_merged=agg['merged'], total_split=agg['split'], total_covered=agg['covered'],
               overall=O, per_cube=per_cube)
    json.dump(res, open(a.out, 'w'), indent=2)
    print(f"\n=== {res['label']} / {a.ckpt}  ({agg['n']} real cubes) ===")
    print(f"  FG DICE     : {O['fg_dice']}   (official-family metric; medial_059 published pseudo-dice ~0.60 on its own val)")
    print(f"  MERGER RATE : {O['merger_rate']}   (lower=better; baseline medial_059 ~0.42)")
    print(f"  SPLIT  RATE : {O['split_rate']}   (lower=better; the dual failure)")
    print(f"  VOI merge/split: {O['voi_merge']} / {O['voi_split']}   SURF-DICE@{a.tau}: {O['surf_dice']}")
    print(f"  COMPOSITE   : {O['composite']}   (higher=better)")
    if a.vs and os.path.isfile(a.vs):
        b = json.load(open(a.vs)); B = b['overall']
        print(f"\n  vs {b.get('label','baseline')}:")
        for k, better_low in [('fg_dice', False), ('merger_rate', True), ('split_rate', True), ('voi_merge', True),
                              ('voi_split', True), ('surf_dice', False), ('composite', False)]:
            dv = round(O[k] - B[k], 4); arrow = '' if dv == 0 else ('↓' if dv < 0 else '↑')
            good = (dv < 0) if better_low else (dv > 0)
            tag = '  GOOD' if (dv != 0 and good) else ('  WORSE' if dv != 0 else '')
            print(f"    {k:11s}: {B[k]:>8} -> {O[k]:>8}  ({dv:+}) {arrow}{tag}")


if __name__ == "__main__":
    main()
