#!/usr/bin/env python3
"""Colored-layer renderer + cube inspector (Tier-4 verification gate AND data explorer).

Loads an instance cube (CT volume + per-sheet integer instance mask) and:
  - prints stats: shape, CT intensity range, #sheets, per-sheet voxel counts, est. sheet thickness;
  - renders a 3-orthogonal-view panel: CT grayscale with each instance ID a DISTINCT color overlay,
    so a human can see how densely/correctly layers sit (and, for synthetic cubes, whether two merged
    sheets stay separable or blob together).

Works on both REAL cubes (volume.nrrd + mask.nrrd) and SYNTHETIC cubes (any vol/lbl arrays).
Usage:  python render_layers.py --cube <dir_with_z_y_x_volume.nrrd> [--out panel.png]
        python render_layers.py --vol a.nrrd --lbl b.nrrd --out panel.png
"""
import os, sys, glob, argparse, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


def load_nrrd(path):
    import nrrd
    data, _ = nrrd.read(path)
    return np.asarray(data)


def find_pair(cube_dir):
    vol = glob.glob(os.path.join(cube_dir, "*_volume.nrrd"))
    msk = glob.glob(os.path.join(cube_dir, "*_mask.nrrd"))
    if not (vol and msk):
        sys.exit(f"no *_volume.nrrd / *_mask.nrrd in {cube_dir}")
    return vol[0], msk[0]


def stats(vol, lbl):
    ids = np.unique(lbl); ids = ids[ids != 0]
    print(f"shape {vol.shape} dtype {vol.dtype} | CT range [{vol.min()},{vol.max()}] mean {vol.mean():.1f}")
    print(f"#sheets (instance ids != 0): {len(ids)} | ids {ids[:12].tolist()}{'...' if len(ids)>12 else ''}")
    fg = (lbl != 0).mean() * 100
    print(f"papyrus voxel fraction: {fg:.1f}%")
    for i in ids[:10]:
        cnt = int((lbl == i).sum())
        print(f"  sheet {int(i):>4}: {cnt:>9} vox ({cnt/lbl.size*100:.2f}%)")
    return ids


def distinct_colors(ids):
    # deterministic distinct colors per instance id (tab20 + hsv fallback), 0 = transparent
    base = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[:, :3]
    out = {}
    for k, i in enumerate(sorted(int(x) for x in ids)):
        out[i] = base[k % 20] if k < 20 else mcolors.hsv_to_rgb([(k * 0.13) % 1.0, 0.7, 0.95])
    return out


def color_overlay(ct_slice, lbl_slice, cmap):
    cn = ct_slice.astype(np.float32)
    lo, hi = np.percentile(cn, 1), np.percentile(cn, 99)
    cn = np.clip((cn - lo) / (hi - lo + 1e-6), 0, 1)
    rgb = np.stack([cn] * 3, -1)
    for i, c in cmap.items():
        m = lbl_slice == i
        rgb[m] = 0.45 * rgb[m] + 0.55 * np.array(c)
    return rgb


def render(vol, lbl, out_png, title=""):
    ids = np.unique(lbl); ids = ids[ids != 0]
    cmap = distinct_colors(ids)
    Z, Y, X = vol.shape; m = [Z // 2, Y // 2, X // 2]
    planes = [(vol[m[0]], lbl[m[0]], "axial z"), (vol[:, m[1]], lbl[:, m[1]], "coronal y"),
              (vol[:, :, m[2]], lbl[:, :, m[2]], "sagittal x")]
    fig, ax = plt.subplots(2, 3, figsize=(16, 11)); fig.suptitle(title, color="w", fontsize=13)
    for j, (c, l, name) in enumerate(planes):
        ax[0, j].imshow(c, cmap="gray"); ax[0, j].set_title(f"{name}  CT", color="w"); ax[0, j].axis("off")
        ax[1, j].imshow(color_overlay(c, l, cmap))
        ax[1, j].set_title(f"{name}  instances ({len(ids)} sheets)", color="w"); ax[1, j].axis("off")
    fig.patch.set_facecolor("#111"); plt.tight_layout()
    plt.savefig(out_png, dpi=85, facecolor="#111"); plt.close()
    print(f"wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube"); ap.add_argument("--vol"); ap.add_argument("--lbl")
    ap.add_argument("--out", default="layers_panel.png")
    a = ap.parse_args()
    if a.cube:
        vp, mp = find_pair(a.cube); vol, lbl = load_nrrd(vp), load_nrrd(mp)
    else:
        vol, lbl = load_nrrd(a.vol), load_nrrd(a.lbl)
    ids = stats(vol, lbl)
    render(vol, lbl, a.out, title=os.path.basename(a.cube or a.vol))


if __name__ == "__main__":
    main()
