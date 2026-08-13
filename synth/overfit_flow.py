#!/usr/bin/env python3
"""OVERFIT-ONE-BATCH for the FLOW head -- the go/no-go gate before any 20-epoch flow-arm run (SOTA_PLAN U9).

Mirrors overfit_test.py (which proved the affinity target learnable at AUC 0.987), but for the interior-distance +
flow field. A fresh flow head on the pretrained medial_059 backbone must, on a handful of synth patches with plain
L1+cosine field loss and nothing else:
  (1) drive the distance L1 down, and
  (2) reach a high SEPARATION AUC = predicted distance ranks wrap-CORE voxels above wrap-SEAM voxels (fiber voxels
      whose neighborhood holds a different wrap id). This is the threshold-free analog of the affinity AUC gate and
      is what the watershed decode relies on. Secondarily, the watershed decode of the predicted distance should
      recover the wraps (adjusted-Rand vs GT instances).
If separation AUC does not climb well above 0.5 here -- ON DATA THE HEAD HAS SEEN -- the field target is not being
learned and no 20-epoch run will help; stop and diagnose. If it does, the representation + plumbing are sound.

Usage:
  python overfit_flow.py --base_dir /root/hf_ckpt/04_resources/checkpoints/medialBW_synth_100ep \
      --corpus /root/surf/data/synthfuse_corpus --n 4 --steps 400 --crop 128 --gpu 0
"""
import os, sys, glob, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'custom'))
from tstr_eval_aff import preprocess
from diag_affinity import auc_mannwhitney
from flow_field import flow_target_fast, flow_decode


def build_fresh_flow(base_dir, ckpt, device):
    """Fresh NetWithFlowHead on the pretrained backbone (encoder/decoder loaded; flow_head at init)."""
    import torch, json
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    import nnUNetTrainerFlow as FL
    plans = json.load(open(os.path.join(base_dir, "plans.json")))
    dj = json.load(open(os.path.join(base_dir, "dataset.json")))
    pm = PlansManager(plans); cm = pm.get_configuration("3d_fullres"); lm = pm.get_label_manager(dj)
    net = FL._FlowGtMixin.build_network_architecture(
        cm.network_arch_class_name, cm.network_arch_init_kwargs, cm.network_arch_init_kwargs_req_import,
        len(dj["channel_names"]), lm.num_segmentation_heads, True)
    sd = torch.load(os.path.join(base_dir, "fold_0", ckpt), map_location="cpu", weights_only=False)
    sd = sd.get("network_weights", sd)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    assert all('flow_head' in m for m in missing), f"backbone keys failed to load: {[m for m in missing][:5]}"
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    net.decoder.deep_supervision = False
    return pm, cm, lm, dj, net.to(device), FL


def sep_auc(D_pred, inst):
    """separation AUC: predicted distance on wrap CORES (high) vs SEAMS (low). Seam = fiber voxel whose 3x3x3
    neighborhood holds a different wrap id; core = the rest of the fiber."""
    from scipy import ndimage as ndi
    fg = inst > 0
    BIG = np.iinfo(np.int32).max
    big = np.where(fg, inst, BIG)
    mn = ndi.minimum_filter(big, 3); mx = ndi.maximum_filter(np.where(fg, inst, 0), 3)
    seam = fg & (mn != BIG) & (mn != mx)
    core = fg & ~seam
    if not seam.any() or not core.any():
        return float('nan')
    return auc_mannwhitney(D_pred[core], D_pred[seam])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True); ap.add_argument("--ckpt", default="checkpoint_final.pth")
    ap.add_argument("--corpus", required=True); ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--steps", type=int, default=400); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--crop", type=int, default=128); ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    import torch, tifffile
    import torch.nn.functional as F
    from nnUNetTrainerFlow import FieldLoss
    dev = torch.device(f"cuda:{a.gpu}")
    pm, cm, lm, dj, net, FL = build_fresh_flow(a.base_dir, a.ckpt, dev)
    n_seg = int(lm.num_segmentation_heads)

    class _P:
        def __init__(s): s.plans_manager, s.configuration_manager, s.label_manager, s.dataset_json = pm, cm, lm, dj
    P = _P()

    xs, Ds, Fs, insts = [], [], [], []
    for f in sorted(glob.glob(f"{a.corpus}/labelsTr_inst/*.tif"))[:a.n]:
        cid = os.path.basename(f)[:-4]
        img = glob.glob(f"{a.corpus}/imagesTr/{cid}_0000.tif")
        if not img:
            continue
        c = a.crop
        vol = tifffile.imread(img[0]).astype(np.float32)[:c, :c, :c]
        inst = tifffile.imread(f).astype(np.int32)[:c, :c, :c]
        D, flow = flow_target_fast(inst)
        data = preprocess(P, vol)
        xs.append(torch.from_numpy(data)[None].to(dev))
        Ds.append(torch.from_numpy(D[None, None]).to(dev))
        Fs.append(torch.from_numpy(flow[None]).to(dev))
        insts.append(inst)
    print(f"patches={len(xs)} crop={a.crop}", flush=True)
    floss = FieldLoss(flow_w=float(os.environ.get('FLOW_W', '1.0')))

    def evalsep(i):
        net.eval()
        with torch.no_grad():
            o = net(xs[i]); o = (o[0] if isinstance(o, (tuple, list)) else o).float()[0]
        Dp = F.softplus(o[n_seg]).cpu().numpy()
        return Dp, sep_auc(Dp, insts[i])

    Dp0, au0 = evalsep(0)
    print(f"BEFORE: sep-AUC {au0:.4f}  (D_pred range {Dp0.min():.2f}..{Dp0.max():.2f})", flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    net.train()
    for step in range(a.steps):
        i = step % len(xs)
        opt.zero_grad(set_to_none=True)
        o = net(xs[i]); o = o[0] if isinstance(o, (tuple, list)) else o
        fiber = (Ds[i] > 0) | (Fs[i].abs().sum(1, keepdim=True) > 0)
        fiber = torch.from_numpy(insts[i][None, None] > 0).to(dev)
        l, l_d, l_f = floss(o[:, n_seg:], Ds[i], Fs[i], fiber)
        l.backward(); opt.step()
        if step % 50 == 0 or step == a.steps - 1:
            Dp, au = evalsep(0)
            print(f"  step {step:4d}  l {float(l):.4f} (l_d {float(l_d):.4f} l_f {float(l_f):.4f})  sep-AUC {au:.4f}",
                  flush=True)
            net.train()

    # final: separation AUC across all patches + a decode adjusted-Rand check on patch 0
    aucs = [evalsep(i)[1] for i in range(len(xs))]
    final = float(np.nanmean(aucs))
    Dp, _ = evalsep(0)
    labels = flow_decode(Dp, insts[0] > 0, core_frac=0.5)
    try:
        from sklearn.metrics import adjusted_rand_score
        m = insts[0] > 0
        ari = adjusted_rand_score(insts[0][m], labels[m])
    except Exception:
        ari = float('nan')
    print(f"\nFINAL sep-AUC {final:.4f} (per-patch {[round(x,3) for x in aucs]}) | decode ARI vs GT {ari:.3f}")
    print("OVERFIT_FLOW_OK -- flow target learnable + plumbed; proceed to the 20ep flow run"
          if final > 0.85 else
          "OVERFIT_FLOW_FAIL -- distance field not separating cores from seams even in-domain; diagnose before training")
    print("OVERFIT_FLOW_DONE")


if __name__ == "__main__":
    main()
