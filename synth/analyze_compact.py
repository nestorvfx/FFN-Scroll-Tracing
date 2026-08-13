#!/usr/bin/env python3
"""Extract NUMERIC targets from real compact scroll windows -- these become the composer's unit-test thresholds.

Per window: estimate the sheet-normal direction (structure tensor), rotate so sheets run vertically, then measure
  - seam widths: dark (air) run lengths along the normal, interior only  -> how thick the visible gaps are
  - visible-seam spacing: bright (papyrus) run lengths along the normal  -> how far apart visible seams sit
    (in fused zones this spans MULTIPLE physical wraps -- that is the point: fused contacts are invisible)
  - seam persistence: for thin seams, their extent along the sheet direction before pinching closed
  - fg fraction

Usage: python analyze_compact.py --tif /root/data/10192.tif --centers "3960,5776;3960,1768;3336,1944;6624,4792"
"""
import argparse, json
import numpy as np
import tifffile
from scipy import ndimage as ndi


def dominant_angle(img):
    g = ndi.gaussian_filter(img.astype(np.float32), 2)
    gy, gx = np.gradient(g)
    Jxx = ndi.uniform_filter(gx * gx, 31); Jyy = ndi.uniform_filter(gy * gy, 31)
    Jxy = ndi.uniform_filter(gx * gy, 31)
    ang = 0.5 * np.arctan2(2 * Jxy.mean(), (Jxx.mean() - Jyy.mean()))
    return float(np.degrees(ang))          # gradient direction = sheet NORMAL direction


def runs(mask_1d):
    d = np.diff(np.concatenate([[0], mask_1d.view(np.int8), [0]]))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return e - s


def window_stats(win, force_angle=None):
    """force_angle: for SYNTHETIC volumes the sheet normal is known by construction -- pass it. The structure-tensor
    estimate gets hijacked by rasterization terraces (v4 measured 36-84 deg on stacks whose true normal is 0), and
    then every run-length statistic is taken along the wrong axis."""
    from skimage.filters import threshold_otsu
    ang = dominant_angle(win) if force_angle is None else float(force_angle)
    rot = ndi.rotate(win, 90 - ang, reshape=False, order=1)   # normal -> horizontal (rows cross the sheets)
    core = rot[64:-64, 64:-64]
    thr = float(threshold_otsu(core))
    fg = ndi.gaussian_filter(core, 1.0) >= thr
    seam_w, pap_w = [], []
    for r in range(fg.shape[0]):
        row = fg[r]
        idx = np.flatnonzero(row)
        if idx.size < 10:
            continue
        interior = np.s_[idx[0]:idx[-1] + 1]
        seam_w += runs(~row[interior]).tolist()
        pap_w += runs(row[interior]).tolist()
    # seam persistence: thin dark components' extent along sheet direction (columns here)
    airlab, n = ndi.label(~fg)
    persist = []
    for sl in ndi.find_objects(airlab):
        if sl is None:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if h <= 4 and w >= 8:           # thin, elongated along sheets
            persist.append(w)
        elif w <= 4 and h >= 8:
            persist.append(h)
    def pct(x):
        return [round(float(v), 1) for v in np.percentile(x, [25, 50, 90])] if len(x) else [0, 0, 0]
    return dict(angle=round(ang, 1), fg_frac=round(float(fg.mean()), 3),
                seam_w_p=pct(seam_w), pap_w_p=pct(pap_w), persist_p=pct(persist),
                n_seams=len(seam_w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", required=True)
    ap.add_argument("--centers", required=True, help="'y,x;y,x;...'")
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--out", default="/root/compact_targets.json")
    a = ap.parse_args()
    img = tifffile.imread(a.tif).astype(np.float32)
    lo, hi = np.percentile(img, [1, 99.5])
    img = np.clip((img - lo) / (hi - lo + 1e-6) * 255, 0, 255)
    res = []
    for part in a.centers.split(";"):
        cy, cx = (int(v) for v in part.split(","))
        h = a.size // 2
        res.append(dict(center=[cy, cx], **window_stats(img[cy - h:cy + h, cx - h:cx + h])))
    agg = dict(windows=res)
    for k in ("seam_w_p", "pap_w_p", "persist_p"):
        agg[k + "_median"] = [round(float(np.median([w[k][i] for w in res])), 1) for i in range(3)]
    agg["fg_frac_range"] = [min(w["fg_frac"] for w in res), max(w["fg_frac"] for w in res)]
    json.dump(agg, open(a.out, "w"), indent=1)
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
