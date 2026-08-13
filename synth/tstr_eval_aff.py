#!/usr/bin/env python3
"""Instance-eval harness for the AFFINITY + Mutex-Watershed arm. Same held-out real cubes, same merger/split/VOI
definitions as tstr_eval.py -- but the instance labeling (pred_cc) comes from MWS over predicted affinities instead
of connected-components of the binary fiber. This makes the affinity merger number DIRECTLY comparable to control's
0.307 (the binary-CC merger on these same cubes).

For each cube it reports THREE labelings off the SAME affinity-model forward pass:
  cc   : connected-components of the model's fiber head        -> sanity-checks the fiber head didn't regress vs control
  mws  : Mutex-Watershed over signed affinities, masked to fiber  -> the actual proposal (the de-merge mechanism)
The cc->mws merger delta isolates the agglomeration benefit from any fiber-head drift.

nnU-Net's predictor hardcodes the logit volume to num_segmentation_heads channels, so we run a self-contained
Gaussian sliding window over the dual-head net (returns cat([seg_logits, aff_logits]) when deep_supervision is off).
Normalization is the predictor's own preprocessor (identical to tstr_eval / control) so the fiber head is apples-to-apples.

MWS bg-confinement: every affinity edge with a background endpoint (or out-of-bounds) is forced to -1 (max repulsive)
so MWS cannot bridge two sheets through unsupervised background; instances are then intersected with the fiber mask.

Usage: python tstr_eval_aff.py --model_dir <.../nnUNetTrainerMedialFinetuneAff__plans__3d_fullres> --fold 0 \
                               --ckpt checkpoint_best.pth --test_list <cubes.txt> --cubes_root <dir> --out aff.json \
                               [--gpu 0] [--label aff_ep15] [--vs control.json]
"""
import os, sys, glob, json, argparse, numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tstr_eval import (load_cube, instance_stats, voi, surface_dice, fg_dice, composite, CONN)

# offsets MUST match the trained arm's AFF_RANGES exactly (GT offsets == inference offsets); set via --ranges
def build_offsets(ranges):
    """MUST stay bit-identical to custom/nnUNetTrainerAff.build_offsets -- the GT offsets and the inference offsets
    are the same list, and a mismatch silently reinterprets every affinity channel. AFF_OFFSETS (JSON list of
    [dz,dy,dx]) overrides the ladder; see the trainer for the measured cross-rate table motivating it."""
    explicit = os.environ.get('AFF_OFFSETS', '').strip()
    if explicit:
        # env WINS over --ranges here (unlike the trainer, whose unit tests pass ranges explicitly): --ranges has an
        # argparse default, so honouring it would silently fall back to the ladder and reinterpret every channel.
        offs = json.loads(explicit)
        assert all(len(o) == 3 for o in offs), f"AFF_OFFSETS must be a list of [dz,dy,dx], got {offs[:3]}"
        return [[int(c) for c in o] for o in offs]
    rs = [int(r) for r in str(ranges).split(',') if str(r).strip()]
    offs = []
    for r in rs:
        offs += [[r, 0, 0], [0, r, 0], [0, 0, r]]
    return offs


def build_aff_network(model_dir, fold, ckpt, device):
    """Build the dual-head arch via the trainer named in the plans, load the fold's weights, DS off (inference mode)."""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                        device=device, verbose=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(model_dir, use_folds=(fold,), checkpoint_name=ckpt)
    net = p.network
    net.load_state_dict(p.list_of_parameters[0])
    net.decoder.deep_supervision = False                 # external inference: forward returns concatenated logits
    net.to(device).eval()
    return p, net


def preprocess(p, vol):
    """Exact training-time normalization (predictor's own preprocessor) -> (C,Z,Y,X) float32."""
    prep = p.configuration_manager.preprocessor_class(verbose=False)
    out = prep.run_case_npy(vol[None].astype(np.float32), None, {'spacing': (1, 1, 1)},
                            p.plans_manager, p.configuration_manager, p.dataset_json)
    return np.asarray(out[0], dtype=np.float32)


def sliding_window_logits(net, data, patch_size, n_out, device, step=0.5, devices=None, nets=None):
    """Gaussian-weighted sliding window; aggregates the n_out concatenated logit channels. Returns (n_out,Z,Y,X).

    CPU-accumulated (the 14-channel slab logit volume OOMs a 32GB GPU), transfer-optimized by cropping each tile
    on the GPU to the UNPADDED z-range before download: the slab pads z 64->192, so ~2/3 of every naive transfer
    was padding -> ~3x less host transfer. Verified BIT-EXACT vs the full-volume accumulation by the --swtest gate
    (diag_sw.py). NOTE: the download stays fp32 -- an earlier fp16 cast of (out*g) UNDERFLOWED the gaussian-tail
    products (g spans ~1 at center to ~1e-8 at edges; fp16 subnormal floor ~6e-8), corrupting the weighted average
    of edge-fed voxels by up to 0.22 in probability. fp16 is NOT safe here; fp32 crop gives the speed without it.

    MULTI-GPU (devices=[dev0,dev1,...] + nets=[net0,net1,...]): the y-rows of the SAME tile grid are dealt
    round-robin to one worker thread per device; each worker accumulates partial (acc,wacc); partials are summed
    and divided ONCE. Identical tiles + identical gaussian weights -> same result up to fp32 summation order
    (~1e-7, far below every downstream decision: 0.1 dip resolution, 0.5 threshold). ~linear wall-clock speedup.
    NOTE: nets must be INDEPENDENTLY BUILT per device (build_aff_network) -- NetWithAffinityHead's forward_pre_hook
    lambda closes over its own instance, so deepcopy would write features into the ORIGINAL net (races/breaks)."""
    import torch
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    data, slicer = pad_nd_image(data, patch_size, 'constant', {'constant_values': 0}, True)
    data = torch.from_numpy(data).float()
    Z, Y, X = data.shape[1:]
    zs = slicer[1]                                        # unpadded z-range within the padded volume
    z_lo, z_hi = zs.start or 0, zs.stop if zs.stop is not None else Z
    steps = compute_steps_for_sliding_window((Z, Y, X), patch_size, step)
    pz, py, px = patch_size
    if devices is None:
        devices = [device]
    if nets is None:
        nets = [net]
    if len(nets) != len(devices):
        raise ValueError(f"need one independently-built net per device (got {len(nets)} nets, {len(devices)} devices)")
    errs = {}

    def run_shard(worker, dev, model, out):
        try:
            g = compute_gaussian(tuple(patch_size), sigma_scale=1. / 8, device=dev, dtype=torch.float32)
            gc = g.cpu()
            acc = torch.zeros((n_out, z_hi - z_lo, Y, X), dtype=torch.float32)
            wacc = torch.zeros((1, z_hi - z_lo, Y, X), dtype=torch.float32)
            with torch.no_grad():
                for sz in steps[0]:
                    a, b = max(sz, z_lo), min(sz + pz, z_hi)  # tile overlap with the unpadded z-window
                    if b <= a:
                        continue
                    for yi, sy in enumerate(steps[1]):
                        if yi % len(devices) != worker:    # round-robin y-rows of the SAME grid across devices
                            continue
                        for sx in steps[2]:
                            tile = data[:, sz:sz + pz, sy:sy + py, sx:sx + px][None].to(dev)
                            with torch.autocast(dev.type, enabled=(dev.type == 'cuda')):
                                o = model(tile)[0].float()  # (n_out, pz,py,px) concatenated logits
                            piece = (o[:, a - sz:b - sz] * g[a - sz:b - sz]).cpu()  # fp32 (fp16 underflows g)
                            acc[:, a - z_lo:b - z_lo, sy:sy + py, sx:sx + px] += piece
                            wacc[:, a - z_lo:b - z_lo, sy:sy + py, sx:sx + px] += gc[a - sz:b - sz]
            out[worker] = (acc, wacc)
        except Exception as e:                             # surface thread failures (else parts[w] stays None)
            errs[worker] = e

    parts = [None] * len(devices)
    if len(devices) == 1:
        run_shard(0, devices[0], nets[0], parts)
    else:
        import threading
        ths = [threading.Thread(target=run_shard, args=(w, d, m, parts)) for w, (d, m) in enumerate(zip(devices, nets))]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
    if errs:
        raise RuntimeError(f"sliding-window shard worker(s) failed: { {k: repr(v) for k, v in errs.items()} }")
    acc = parts[0][0]
    wacc = parts[0][1]
    for p in parts[1:]:
        acc += p[0]
        wacc += p[1]
    acc /= wacc
    # spatial unpad: z already cropped during accumulation; apply the slicer's y/x crops only. (The slicer's
    # channel entry is for the 1-channel input and must NOT be applied to the n_out accumulation.)
    acc = acc.numpy()[(slice(None), slice(None)) + tuple(slicer[2:])]
    return acc


def shift_fiber(fiber, off):
    """g[v] = fiber[v+off] for in-bounds v+off, else False (no wrap)."""
    g = np.zeros_like(fiber)
    dz, dy, dx = off
    Z, Y, X = fiber.shape
    z0, z1 = max(0, -dz), Z - max(0, dz)
    y0, y1 = max(0, -dy), Y - max(0, dy)
    x0, x1 = max(0, -dx), X - max(0, dx)
    g[z0:z1, y0:y1, x0:x1] = fiber[z0 + dz:z1 + dz, y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    return g


def mws_instances(aff, fiber, offsets, stride_scale=0, long_repulsive_only=False, bias=0.5):
    """Mutex-Watershed decode. Signed edge weights (p - bias); any bg-endpoint edge forced to -1 (mwatershed:
    sign => attractive/mutex, sorted by |w|, so -1 bg edges outrank everything and isolate the fiber mask).

    long_repulsive_only=True (slab decode): offsets with |off|_1 > 1 contribute ONLY their repulsive part
    (min(0, p-bias)). Rationale for the merge-averse objective: a long-range ATTRACTIVE edge can fuse two wraps
    straight across a thin air gap (unbounded downstream cost), while the fragmentation it would have prevented
    is cheap and recovered by affinity-aware absorption. This is the Wolf et al. MWS setup (short attractive,
    long-range repulsive).

    stride_scale>0: grid-subsample edges per offset (stride ~ max(1, |off|//stride_scale)) so the slab-scale
    edge list fits in RAM. Deterministic grid strides (randomized_strides samples via an UNSEEDED rng in the
    Rust source -- non-reproducible runs)."""
    import mwatershed
    signed = (aff.astype(np.float64) - bias)
    if long_repulsive_only:
        for i, off in enumerate(offsets):
            if sum(abs(c) for c in off) > 1:
                np.minimum(signed[i], 0.0, out=signed[i])
    strides = None
    if stride_scale > 0:
        strides = [[max(1, abs(c) // stride_scale) if c else 1 for c in off] for off in offsets]
    for i, off in enumerate(offsets):
        bg_edge = (~fiber) | (~shift_fiber(fiber, off))   # source bg OR target bg/OOB
        signed[i][bg_edge] = -1.0
    labels = mwatershed.agglom(signed, offsets, strides=strides, randomized_strides=False)
    labels = np.asarray(labels).astype(np.int32)
    labels[~fiber] = 0                                     # confine to fiber (redundant w/ bg-edge masking, but safe)
    return labels


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True); ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--ckpt", default="checkpoint_best.pth")
    ap.add_argument("--test_list", required=True); ap.add_argument("--cubes_root", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tau", type=int, default=2); ap.add_argument("--label", default=None)
    ap.add_argument("--ranges", default="1,3,9", help="offset magnitudes; MUST match the arm's AFF_RANGES")
    ap.add_argument("--vs", default=None, help="path to a control result.json to print a delta table")
    a = ap.parse_args()
    offsets = build_offsets(a.ranges); n_aff = len(offsets)
    print(f"offsets ({n_aff}): {offsets}")
    device = torch.device(f'cuda:{a.gpu}')
    cubes = [c.strip() for c in open(a.test_list) if c.strip()]
    p, net = build_aff_network(a.model_dir, a.fold, a.ckpt, device)
    patch_size = list(p.configuration_manager.patch_size)
    per_cube = []
    agg = {m: dict(merged=0, split=0, covered=0) for m in ('cc', 'mws')}
    aggf = dict(vm=0.0, vs=0.0, sd=0.0, dsc=0.0, n=0)
    for cid in cubes:
        d = os.path.join(a.cubes_root, cid)
        if not glob.glob(d + '/*_volume.nrrd'):
            continue
        vol, gt = load_cube(d)
        data = preprocess(p, vol)
        logits = sliding_window_logits(net, data, patch_size, 2 + n_aff, device)
        fiber = logits[:2].argmax(0) == 1
        aff = 1.0 / (1.0 + np.exp(-logits[2:]))           # sigmoid -> P(same sheet) per offset
        cc, _ = ndi.label(fiber, structure=CONN)
        mws = mws_instances(aff, fiber, offsets)
        rec = dict(cube=cid)
        for name, pred_cc in (('cc', cc), ('mws', mws)):
            s = instance_stats(fiber, gt, pred_cc=pred_cc)
            rec[name] = dict(merger_rate=s['merger_rate'], split_rate=s['split_rate'],
                             merged_sheets=s['merged_sheets'], split_sheets=s['split_sheets'],
                             n_sheets=s['n_sheets'], n_covered=s['n_covered'])
            agg[name]['merged'] += s['merged_sheets']; agg[name]['split'] += s['split_sheets']
            agg[name]['covered'] += s['n_covered']
        vm, vs = voi(gt, fiber, pred_cc=mws)
        sd = surface_dice(gt, fiber, a.tau); dsc = fg_dice(gt, fiber)
        rec.update(voi_merge=vm, voi_split=vs, surf_dice=sd, fg_dice=dsc)
        per_cube.append(rec)
        aggf['vm'] += vm; aggf['vs'] += vs; aggf['sd'] += sd; aggf['dsc'] += dsc; aggf['n'] += 1
        print(f"{cid}: cov {rec['cc']['n_covered']} | fgDice {dsc} | "
              f"CC merge {rec['cc']['merger_rate']} split {rec['cc']['split_rate']}  ->  "
              f"MWS merge {rec['mws']['merger_rate']} split {rec['mws']['split_rate']} | sDice {sd}", flush=True)
    n = max(1, aggf['n'])
    O = {}
    for name in ('cc', 'mws'):
        cov = max(1, agg[name]['covered'])
        O[name] = dict(merger_rate=round(agg[name]['merged'] / cov, 4),
                       split_rate=round(agg[name]['split'] / cov, 4))
    O['fg_dice'] = round(aggf['dsc'] / n, 4); O['surf_dice'] = round(aggf['sd'] / n, 4)
    O['voi_merge'] = round(aggf['vm'] / n, 4); O['voi_split'] = round(aggf['vs'] / n, 4)
    O['composite'] = composite(O['mws']['merger_rate'], O['mws']['split_rate'], O['surf_dice'],
                               O['voi_merge'] + O['voi_split'])
    res = dict(label=a.label or os.path.basename(a.model_dir), model_dir=a.model_dir, ckpt=a.ckpt,
               n_cubes=aggf['n'], overall=O, per_cube=per_cube)
    json.dump(res, open(a.out, 'w'), indent=2)
    print(f"\n=== {res['label']} / {a.ckpt}  ({aggf['n']} real cubes) ===")
    print(f"  FG DICE       : {O['fg_dice']}")
    print(f"  CC  MERGER    : {O['cc']['merger_rate']}   split {O['cc']['split_rate']}   (fiber head + connected-comp; vs control 0.307)")
    print(f"  MWS MERGER    : {O['mws']['merger_rate']}   split {O['mws']['split_rate']}   (affinity + Mutex-Watershed; THE proposal)")
    print(f"  VOI merge/split: {O['voi_merge']} / {O['voi_split']}   SURF-DICE@{a.tau}: {O['surf_dice']}")
    print(f"  COMPOSITE     : {O['composite']}")
    if a.vs and os.path.isfile(a.vs):
        b = json.load(open(a.vs)); B = b['overall']
        bm = B['merger_rate'] if 'merger_rate' in B else B.get('mws', {}).get('merger_rate')
        bs = B['split_rate'] if 'split_rate' in B else B.get('mws', {}).get('split_rate')
        dm = round(O['mws']['merger_rate'] - bm, 4); dsr = round(O['mws']['split_rate'] - bs, 4)
        print(f"\n  vs {b.get('label','control')}: MWS merger {bm} -> {O['mws']['merger_rate']} ({dm:+}) "
              f"{'GOOD' if dm < 0 else ('WORSE' if dm > 0 else '')} | split {bs} -> {O['mws']['split_rate']} ({dsr:+})")


if __name__ == "__main__":
    main()
