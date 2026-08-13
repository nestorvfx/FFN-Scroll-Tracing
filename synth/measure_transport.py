#!/usr/bin/env python3
"""Identity-transport-distance diagnostic (consult section 4): every local model (affinity, seam-class, normal)
can only separate a fused contact if the two wraps become distinguishable SOMEWHERE inside its receptive field.
For each wrap-wrap contact voxel on the truth slab, compute the in-plane geodesic distance (along the contact
interface) to the nearest location where the two wraps are separated by >=1 voxel of true background. The mass
beyond the patch radius (~64 vox) is provably unreachable by any patch-local representation -- if it dominates,
the leverage moves downstream to the tracer (winding-consistency), not to better segmentation.

Usage: python measure_transport.py --slab /root/data/slab
"""
import os, argparse, json
import numpy as np
from scipy import ndimage as ndi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slab", required=True)
    a = ap.parse_args()
    import nrrd
    truth = nrrd.read(os.path.join(a.slab, "truth.nrrd"))[0].astype(np.int32)
    Z = truth.shape[0]
    all_d = []
    for z in range(0, Z, 4):                               # every 4th slice is plenty
        t = truth[z]
        # contact voxels: fg voxel 8-adjacent to a DIFFERENT fg label
        mx = ndi.maximum_filter(t, size=3)
        big = np.where(t > 0, t, np.iinfo(np.int32).max)
        mn = ndi.minimum_filter(big, size=3)
        contact = (t > 0) & (mn < np.iinfo(np.int32).max) & (mx != mn)
        if not contact.any():
            continue
        # separation evidence: fg voxel adjacent to true background (bg within labeled region's bbox --
        # use bg = truth==0 near fg; that includes unlabeled, conservative in BOTH directions, so ALSO compute
        # strict variant: bg surrounded by the same two wraps... keep simple: bg-adjacent contact-free zone)
        bg = t == 0
        near_bg = (t > 0) & ndi.binary_dilation(bg, np.ones((3, 3)))
        sep = near_bg & ~contact                           # separated boundary evidence points
        if not sep.any():
            all_d += [999.0] * int(contact.sum())
            continue
        # geodesic-ish: euclidean distance from each contact voxel to nearest separation-evidence voxel
        # (euclidean lower-bounds geodesic; if even THIS exceeds the patch, geodesic certainly does)
        dist = ndi.distance_transform_edt(~sep)
        all_d += dist[contact].tolist()
    d = np.array(all_d, np.float32)
    out = dict(n=int(d.size),
               p50=float(np.percentile(d, 50)), p75=float(np.percentile(d, 75)),
               p90=float(np.percentile(d, 90)), p95=float(np.percentile(d, 95)),
               frac_beyond_32=float((d > 32).mean()), frac_beyond_64=float((d > 64).mean()),
               frac_beyond_96=float((d > 96).mean()))
    print(json.dumps(out, indent=1))
    json.dump(out, open("/root/transport_diag.json", "w"), indent=1)


if __name__ == "__main__":
    main()
