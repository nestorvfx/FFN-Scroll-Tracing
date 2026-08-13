#!/usr/bin/env python3
"""Emit the FLOW arm on the GT slab and decode to a carved binary (the tracer-facing artifact), for CTF scoring.

Runs the dual-head flow net over the slab (sliding window), takes the fiber prob from the seg head and the interior
distance D=softplus(flow_ch0) + flow=flow_ch1..3 from the flow head, then marker-controlled watershed on D within
the fiber (synth/flow_field.flow_decode) -> instances -> carve 26-adjacent seams -> carved binary. Caches the raw
network pass (fiber.npy + field_f16.npy) so decode params iterate for free. Mirrors slab_emit_carved.py.

Usage:
  python slab_emit_flow.py --model_dir <...Flow__...3d_fullres> --ckpt checkpoint_final.pth --tag flow --gpu 0,1
                           [--core_frac 0.5] [--min_core 30]
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tstr_eval_aff import build_aff_network, preprocess, sliding_window_logits
from flow_field import flow_decode, carve_from_labels


def main():
    import torch, nrrd
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True); ap.add_argument("--ckpt", default="checkpoint_final.pth")
    ap.add_argument("--tag", default="flow"); ap.add_argument("--gpu", default="0")
    ap.add_argument("--fold", type=int, default=0); ap.add_argument("--fiber_thr", type=float, default=0.5)
    ap.add_argument("--core_frac", type=float, default=0.5); ap.add_argument("--min_core", type=int, default=30)
    ap.add_argument("--n_flow", type=int, default=4)
    a = ap.parse_args()
    gpu_ids = [int(g) for g in str(a.gpu).split(",") if str(g).strip() != ""]
    devices = [torch.device(f"cuda:{g}") for g in gpu_ids]; device = devices[0]
    out_dir = f"/root/slab_pred_{a.tag}"; os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(f"{out_dir}/field_f16.npy") and os.path.exists(f"{out_dir}/fiber.npy"):
        fiber_prob = np.load(f"{out_dir}/fiber.npy")
        field = np.load(f"{out_dir}/field_f16.npy").astype(np.float32)
        print("reusing cached fiber/field", field.shape, flush=True)
    else:
        vol = nrrd.read("/root/data/slab/volume.nrrd")[0].astype(np.float32)
        p, net = build_aff_network(a.model_dir, a.fold, a.ckpt, device)
        nets = [net] + [build_aff_network(a.model_dir, a.fold, a.ckpt, d)[1] for d in devices[1:]]
        patch = list(p.configuration_manager.patch_size)
        n_seg = int(p.label_manager.num_segmentation_heads)
        data = preprocess(p, vol)
        print("slab", data.shape, "patch", patch, "n_seg", n_seg, "n_flow", a.n_flow, "gpus", gpu_ids, flush=True)
        logits = sliding_window_logits(net, data, patch, n_seg + a.n_flow, device, devices=devices, nets=nets)
        seg = logits[:n_seg]
        field = logits[n_seg:n_seg + a.n_flow].astype(np.float32)
        field[0] = np.logaddexp(0.0, field[0])            # softplus -> interior distance D >= 0
        if n_seg >= 2:
            e = np.exp(seg - seg.max(0, keepdims=True)); fiber_prob = (e[1] / (e.sum(0) + 1e-9)).astype(np.float32)
        else:
            fiber_prob = (1.0 / (1.0 + np.exp(-seg[0]))).astype(np.float32)
        np.save(f"{out_dir}/fiber.npy", fiber_prob)
        np.save(f"{out_dir}/field_f16.npy", field.astype(np.float16))

    fiber = fiber_prob >= a.fiber_thr
    D = field[0]
    print(f"fiber fg frac {float(fiber.mean()):.4f}; D range {D[fiber].min():.2f}..{D[fiber].max():.2f}", flush=True)
    labels = flow_decode(D, fiber, core_frac=a.core_frac, min_core=a.min_core)
    n_inst = int(len(np.unique(labels[labels > 0])))
    print(f"flow-decode instances: {n_inst}", flush=True)
    fc, carved_binary, carve = carve_from_labels(labels, fiber, fiber_prob)
    np.save(f"{out_dir}/labels.npy", labels.astype(np.int32))
    np.save(f"{out_dir}/fiber_carved.npy", fc)
    np.save(f"{out_dir}/carved_binary.npy", carved_binary.astype(np.uint8))
    print(f"FLOW-CARVE: carved {int(carve.sum())} ({float(carve.mean()):.4f} of vol); "
          f"fg {float(fiber.mean()):.4f} -> {float(carved_binary.mean()):.4f}", flush=True)
    print("EMIT_FLOW_DONE", a.tag)


if __name__ == "__main__":
    main()
