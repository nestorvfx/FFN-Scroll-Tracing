#!/usr/bin/env python3
"""OVERFIT-ONE-BATCH: the decisive bug-vs-research test. Run this BEFORE any further training.

Rationale: the affinity head scores AUC 0.52 on its OWN synth training domain. A network of this capacity can fit
even RANDOM labels to near-zero training error (Zhang et al.), so chance-level performance ON TRAINING DATA cannot
be an information/learnability problem -- it is a bug or an optimization failure. This isolates which:

  fit AUC -> ~1.0  : no plumbing bug. The affinity target IS learnable and the failure is the loss/schedule/scale.
  fit AUC -> ~0.5  : a real BUG (offset/crop/mask misalignment, target built from the wrong array, gradient not
                     reaching the head). Stop and hunt it; no amount of training or loss tuning will help.

Deliberately minimal: a handful of patches, PLAIN per-offset class-balanced BCE (a proper scoring rule), NO
Focal-Tversky, NO MALIS -- so nothing but the core mapping is under test. Also reports the per-offset same-rate
pi_k (needed for prior-bias init and per-offset balancing) and the head's logit range.

Usage:
  python overfit_test.py --model_dir <...__3d_fullres> --ckpt checkpoint_final.pth \
      --corpus /root/surf/data/synthfuse_corpus --ranges 1,3,9,27 --n 4 --steps 300 [--reinit] [--gpu 0]
"""
import os, sys, glob, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tstr_eval_aff import build_offsets, build_aff_network, preprocess
from diag_affinity import auc_mannwhitney


def aff_targets(inst_t, offsets):
    """target=1 iff same nonzero instance; valid = fiber->fiber (matches what the decode consumes)."""
    import torch
    Z, Y, X = inst_t.shape[-3:]
    N = len(offsets)
    tgt = inst_t.new_zeros((N, Z, Y, X), dtype=torch.float32)
    val = inst_t.new_zeros((N, Z, Y, X), dtype=torch.float32)
    for i, (dz, dy, dx) in enumerate(offsets):
        z0, z1 = max(0, -dz), Z - max(0, dz)
        y0, y1 = max(0, -dy), Y - max(0, dy)
        x0, x1 = max(0, -dx), X - max(0, dx)
        a = inst_t[z0:z1, y0:y1, x0:x1]
        b = inst_t[z0 + dz:z1 + dz, y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        tgt[i, z0:z1, y0:y1, x0:x1] = ((a == b) & (a > 0)).float()
        val[i, z0:z1, y0:y1, x0:x1] = ((a > 0) & (b > 0)).float()
    return tgt, val


def measure_auc(logits, tgt, val):
    p = 1.0 / (1.0 + np.exp(-logits))
    m = val > 0
    pos = p[m & (tgt > 0)]
    neg = p[m & (tgt == 0)]
    return auc_mannwhitney(pos, neg) if (pos.size and neg.size) else float('nan')


def build_fresh(base_dir, ckpt, n_aff, hidden, device):
    """Build the dual-head net with a FRESH affinity head on the PRETRAINED backbone -- i.e. exactly the state the
    real run starts from. Needs no affinity checkpoint, so the test can run before any affinity training exists.
    Backbone weights load non-strictly: encoder/decoder keys match medial_059, aff_head keys are absent and stay
    at init (which is the point -- we are testing whether a fresh head CAN fit, not whether the old one recovers)."""
    import torch
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "custom"))
    import nnUNetTrainerAff as _A
    _A.N_AFF = n_aff                                     # NetWithAffinityHead reads the module-level default
    from nnUNetTrainerAff import nnUNetTrainerMedialFinetuneAff
    from nnUNetTrainerAffG import DeepAffinityHead
    plans = json.load(open(os.path.join(base_dir, "plans.json")))
    dj = json.load(open(os.path.join(base_dir, "dataset.json")))
    pm = PlansManager(plans)
    cm = pm.get_configuration("3d_fullres")
    lm = pm.get_label_manager(dj)
    net = nnUNetTrainerMedialFinetuneAff.build_network_architecture(
        cm.network_arch_class_name, cm.network_arch_init_kwargs, cm.network_arch_init_kwargs_req_import,
        len(dj["channel_names"]), lm.num_segmentation_heads, True)
    feat = net.decoder.seg_layers[-1].in_channels
    net.aff_head = DeepAffinityHead(feat, n_aff, hidden=hidden)     # same head the V2 arm trains
    sd = torch.load(os.path.join(base_dir, "fold_0", ckpt), map_location="cpu", weights_only=False)
    sd = sd.get("network_weights", sd)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(f"warm-start: {len(sd)} keys; missing {len(missing)} (aff_head, expected), unexpected {len(unexpected)}")
    assert all('aff_head' in m for m in missing), f"backbone keys failed to load: {[m for m in missing][:5]}"
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    net.decoder.deep_supervision = False
    return pm, cm, lm, dj, net.to(device)


class _P:                                                # duck-type of what preprocess()/n_seg need
    def __init__(self, pm, cm, lm, dj):
        self.plans_manager, self.configuration_manager, self.label_manager, self.dataset_json = pm, cm, lm, dj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir"); ap.add_argument("--ckpt", default="checkpoint_final.pth")
    ap.add_argument("--base_dir", help="pretrained backbone folder (plans.json+fold_0/) -> FRESH aff head on it")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--corpus", required=True); ap.add_argument("--ranges", default="1,3,9,27")
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--reinit", action="store_true", help="reinit aff_head w/ per-channel prior bias first")
    ap.add_argument("--crop", type=int, default=128, help="cube crop to fit backward in memory")
    a = ap.parse_args()
    import torch, tifffile
    import torch.nn.functional as F
    dev = torch.device(f"cuda:{a.gpu}")
    offsets = build_offsets(a.ranges)
    if a.base_dir:
        pm, cm, lm, dj, net = build_fresh(a.base_dir, a.ckpt, len(offsets), a.hidden, dev)
        p = _P(pm, cm, lm, dj)
        a.reinit = True                                  # a fresh head must get the prior-bias init to be a fair test
    else:
        p, net = build_aff_network(a.model_dir, a.fold, a.ckpt, dev)
    n_seg = int(p.label_manager.num_segmentation_heads)

    # ---- load a few patches
    xs, ts, vs = [], [], []
    for f in sorted(glob.glob(f"{a.corpus}/labelsTr_inst/*.tif"))[:a.n]:
        cid = os.path.basename(f)[:-4]
        img = glob.glob(f"{a.corpus}/imagesTr/{cid}_0000.tif")
        if not img:
            continue
        c = a.crop
        vol = tifffile.imread(img[0]).astype(np.float32)[:c, :c, :c]
        inst = tifffile.imread(f).astype(np.int32)[:c, :c, :c]
        data = preprocess(p, vol)
        it = torch.from_numpy(inst.astype(np.int64)).to(dev)
        tgt, val = aff_targets(it, offsets)
        xs.append(torch.from_numpy(data)[None].to(dev)); ts.append(tgt); vs.append(val)
    print(f"patches={len(xs)} crop={a.crop} offsets={len(offsets)}", flush=True)

    # ---- per-offset same-rate pi_k (needed for prior-bias init + per-offset balancing)
    pis = []
    for i in range(len(offsets)):
        tot = sum(float(v[i].sum()) for v in vs)
        pos = sum(float((t[i] * v[i]).sum()) for t, v in zip(ts, vs))
        pis.append(pos / max(tot, 1.0))
    print("per-offset same-rate pi_k:", [round(x, 4) for x in pis], flush=True)

    if a.reinit:
        # RetinaNet-style prior-bias init: b_k = logit(pi_k); small weights. Cannot fine-tune OUT of a collapsed
        # basin, so the head must be reset for the test to be about learnability rather than the old attractor.
        with torch.no_grad():
            last = None
            for mod in net.aff_head.modules() if hasattr(net.aff_head, 'modules') else []:
                if isinstance(mod, torch.nn.Conv3d):
                    last = mod
            if last is None and isinstance(net.aff_head, torch.nn.Conv3d):
                last = net.aff_head
            assert last is not None and last.weight.shape[0] == len(offsets), \
                f"picked the wrong conv as the head output: {None if last is None else tuple(last.weight.shape)}"
            last.weight.normal_(0, 0.01)
            for k, pk in enumerate(pis):
                pk = min(max(pk, 1e-4), 1 - 1e-4)
                last.bias[k] = float(np.log(pk / (1 - pk)))
            print(f"reinit aff_head last layer: bias set to logit(pi_k) = "
                  f"{[round(float(last.bias[k]), 3) for k in range(len(pis))]}", flush=True)

    # ---- baseline AUC before any fitting
    net.eval()
    with torch.no_grad():
        o = net(xs[0])
        o = (o[0] if isinstance(o, (tuple, list)) else o).float()[0]
    print(f"BEFORE: logit min {float(o[n_seg:].min()):+.3f} max {float(o[n_seg:].max()):+.3f}  "
          f"AUC {measure_auc(o[n_seg:].cpu().numpy(), ts[0].cpu().numpy(), vs[0].cpu().numpy()):.4f}", flush=True)

    # ---- overfit: plain per-offset class-balanced BCE, nothing else
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    pos_w = torch.tensor([(1 - q) / max(q, 1e-4) for q in pis], device=dev, dtype=torch.float32)
    net.train()
    for step in range(a.steps):
        i = step % len(xs)
        opt.zero_grad(set_to_none=True)
        o = net(xs[i])
        o = o[0] if isinstance(o, (tuple, list)) else o
        if isinstance(o, (tuple, list)):
            o = o[0]
        al = o[0, n_seg:].float()
        t, v = ts[i], vs[i]
        w = pos_w.view(-1, 1, 1, 1)
        bce = F.binary_cross_entropy_with_logits(al, t, reduction='none')
        bce = bce * (t * w + (1 - t))           # per-offset class balancing
        loss = (bce * v).sum() / v.sum().clamp(min=1)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == a.steps - 1:
            net.eval()
            with torch.no_grad():
                oo = net(xs[0])
                oo = (oo[0] if isinstance(oo, (tuple, list)) else oo).float()[0]
            au = measure_auc(oo[n_seg:].cpu().numpy(), ts[0].cpu().numpy(), vs[0].cpu().numpy())
            print(f"  step {step:4d}  loss {float(loss):.4f}  fitAUC {au:.4f}  "
                  f"logit[{float(oo[n_seg:].min()):+.2f},{float(oo[n_seg:].max()):+.2f}]", flush=True)
            net.train()
    net.eval()
    with torch.no_grad():
        oo = net(xs[0])
        oo = (oo[0] if isinstance(oo, (tuple, list)) else oo).float()[0]
    final = measure_auc(oo[n_seg:].cpu().numpy(), ts[0].cpu().numpy(), vs[0].cpu().numpy())
    print(f"\nFINAL fit AUC {final:.4f}")
    print("OVERFIT_OK -- no plumbing bug; the target IS learnable, failure is loss/schedule/scale"
          if final > 0.9 else
          "OVERFIT_FAIL -- cannot fit even a few patches => REAL BUG (offset/crop/mask alignment or gradient path)")
    print("OVERFIT_TEST_DONE")


if __name__ == "__main__":
    main()
