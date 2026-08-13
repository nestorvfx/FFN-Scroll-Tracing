#!/usr/bin/env python3
"""Isolate the sliding-window drift: full-accum-fp32 vs z-crop-fp32 (no half) vs z-crop-fp16 vs brute reference.
Pinpoints whether the drift is the z-crop algorithm or the fp16 transfer. No model/slab needed."""
import numpy as np, torch
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window
from acvl_utils.cropping_and_padding.padding import pad_nd_image

torch.manual_seed(0)
n_out, patch = 6, [32, 32, 32]
net = torch.nn.Conv3d(1, n_out, 3, padding=1)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(dev).eval()
data = np.random.default_rng(0).normal(size=(1, 20, 70, 75)).astype(np.float32)   # z<patch -> padded
pz, py, px = patch


def run(mode):
    d, slicer = pad_nd_image(data.copy(), patch, 'constant', {'constant_values': 0}, True)
    t = torch.from_numpy(d).float()
    Z, Y, X = t.shape[1:]
    zs = slicer[1]; z_lo = zs.start or 0; z_hi = zs.stop if zs.stop is not None else Z
    steps = compute_steps_for_sliding_window((Z, Y, X), patch, 0.5)
    g = compute_gaussian(tuple(patch), sigma_scale=1./8, device=dev, dtype=torch.float32)
    gc = g.cpu()
    if mode == 'full':
        acc = torch.zeros((n_out, Z, Y, X)); wacc = torch.zeros((1, Z, Y, X))
    else:
        acc = torch.zeros((n_out, z_hi - z_lo, Y, X)); wacc = torch.zeros((1, z_hi - z_lo, Y, X))
    with torch.no_grad():
        for sz in steps[0]:
            a, b = max(sz, z_lo), min(sz + pz, z_hi)
            if mode != 'full' and b <= a:
                continue
            for sy in steps[1]:
                for sx in steps[2]:
                    tile = t[:, sz:sz+pz, sy:sy+py, sx:sx+px][None].to(dev)
                    with torch.autocast(dev.type, enabled=(dev.type == 'cuda')):
                        out = net(tile)[0].float()
                    if mode == 'full':
                        acc[:, sz:sz+pz, sy:sy+py, sx:sx+px] += (out * g).cpu()
                        wacc[:, sz:sz+pz, sy:sy+py, sx:sx+px] += gc
                    elif mode == 'crop_fp32':
                        acc[:, a-z_lo:b-z_lo, sy:sy+py, sx:sx+px] += (out[:, a-sz:b-sz] * g[a-sz:b-sz]).cpu()
                        wacc[:, a-z_lo:b-z_lo, sy:sy+py, sx:sx+px] += gc[a-sz:b-sz]
                    else:  # crop_fp16
                        acc[:, a-z_lo:b-z_lo, sy:sy+py, sx:sx+px] += (out[:, a-sz:b-sz] * g[a-sz:b-sz]).half().cpu().float()
                        wacc[:, a-z_lo:b-z_lo, sy:sy+py, sx:sx+px] += gc[a-sz:b-sz]
    acc /= wacc
    if mode == 'full':
        return acc.numpy()[(slice(None),) + tuple(slicer[1:])]
    return acc.numpy()[(slice(None), slice(None)) + tuple(slicer[2:])]


ref = run('full')
for m in ('crop_fp32', 'crop_fp16'):
    r = run(m)
    dl = float(np.abs(r - ref).max())
    dp = float(np.abs(1/(1+np.exp(-r)) - 1/(1+np.exp(-ref))).max())
    print(f"{m:10s} vs full: shape {r.shape}=={ref.shape} max|dlogit|={dl:.2e} max|dprob|={dp:.2e}")
print("logit magnitude: max|ref|=%.2f" % float(np.abs(ref).max()))
