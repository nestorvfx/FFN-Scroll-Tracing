#!/usr/bin/env python3
"""Measure real 3D wrap anisotropy (Fable-5 audit issue 3): synthetic placement fields undulate isotropically
in-plane, but real wraps run near-straight along the scroll axis and curve in the winding plane, so the network
gets conflicting geometry priors and under-learns axis-wise seam continuation (the context needed to bridge
invisible fused stretches).

Two measurements on the real z=10192 slab (z = scroll axis, y/x = winding plane):
  1. Structure-tensor surface normals -> how fast the normal WANDERS per unit length along z vs in-plane. The
     ratio quantifies undulation anisotropy: >1 means straighter along z (the expected wrap geometry).
  2. Per-axis autocorrelation length of the high-passed intensity (fine-structure coherence length by axis).

Usage: python measure_anisotropy.py --slab /root/data/slab
"""
import os, sys, argparse
import numpy as np
from scipy import ndimage as ndi


def structure_tensor_normals(vol, sig_grad=1.0, sig_tensor=4.0):
    g = [ndi.gaussian_filter(vol, sig_grad, order=[1 if i == ax else 0 for i in range(3)]) for ax in range(3)]
    J = np.empty(vol.shape + (3, 3), np.float32)
    for i in range(3):
        for j in range(i, 3):
            Jij = ndi.gaussian_filter(g[i] * g[j], sig_tensor)
            J[..., i, j] = Jij; J[..., j, i] = Jij
    return J


def normal_wander(vol, mask, sig_grad=1.0, sig_tensor=4.0, step=4, steps=None):
    """Surface normal = eigenvector of the SMALLEST structure-tensor eigenvalue (across-sheet direction).
    Measure the mean angular change of the normal over `step` voxels along each axis, within the foreground.
    Smaller angular change along an axis == the sheet is straighter along that axis."""
    J = structure_tensor_normals(vol, sig_grad, sig_tensor)
    w, v = np.linalg.eigh(J)                                # ascending eigenvalues; v[...,0] = smallest -> normal
    n = v[..., 0].astype(np.float32)                        # (Z,Y,X,3)
    out = {}
    for ax, name in zip(range(3), "zyx"):
        stp = steps[ax] if steps else step                 # per-axis steps: matched NATIVE distance (the ::2
        s1 = [slice(None)] * 3; s2 = [slice(None)] * 3     # in-plane subsampling made 4 array-steps = 8 native
        s1[ax] = slice(0, -stp); s2[ax] = slice(stp, None) # voxels vs 4 along z -- inflating the ratio)
        n1 = n[tuple(s1)]; n2 = n[tuple(s2)]
        m = mask[tuple(s1)] & mask[tuple(s2)]
        dot = np.abs((n1 * n2).sum(-1)).clip(0, 1)          # |cos| -> ignore sign flips of the eigenvector
        ang = np.degrees(np.arccos(dot))[m]
        out[name] = float(np.median(ang)) if ang.size else float("nan")
    return out


def autocorr_len(vol, mask, axis, maxlag=40):
    """1/e autocorrelation length of the sigma-1.2 high-pass along `axis`, averaged over foreground lines."""
    hp = (vol - ndi.gaussian_filter(vol, 1.2)).astype(np.float32)
    hp = np.where(mask, hp, 0.0)
    hp = np.moveaxis(hp, axis, 0)
    N = hp.shape[0]
    v0 = float((hp * hp).mean()) + 1e-6
    ac = [1.0]
    for lag in range(1, maxlag):
        ac.append(float((hp[:N - lag] * hp[lag:]).mean()) / v0)
    ac = np.array(ac)
    below = np.nonzero(ac < 1 / np.e)[0]
    return float(below[0]) if below.size else float(maxlag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slab", required=True)
    a = ap.parse_args()
    import nrrd
    truth = nrrd.read(os.path.join(a.slab, "truth.nrrd"))[0].astype(np.int32)
    vol = nrrd.read(os.path.join(a.slab, "volume.nrrd"))[0].astype(np.float32)
    lo, hi = np.percentile(vol, [1, 99.5]); vol = np.clip((vol - lo) / (hi - lo + 1e-6) * 255, 0, 255)
    mask = truth > 0
    # subsample in-plane for speed (eig on a (Z,Y,X,3,3) tensor is heavy); keep full z
    sl = np.s_[:, ::2, ::2]
    v2, m2 = vol[sl], mask[sl]
    wander = normal_wander(v2, m2, steps=(4, 2, 2))    # z full-res step 4; in-plane ::2 grid step 2 = 4 native
    print(f"normal wander (median deg / 4vox): z={wander['z']:.2f}  y={wander['y']:.2f}  x={wander['x']:.2f}")
    inplane = 0.5 * (wander['y'] + wander['x'])
    print(f"  ANISOTROPY ratio in-plane/along-z = {inplane / (wander['z'] + 1e-6):.2f} "
          f"(>1 => straighter along scroll axis z, as expected)")
    acs = {name: autocorr_len(vol, mask, ax) for ax, name in zip(range(3), "zyx")}
    print(f"autocorr length (1/e, vox): z={acs['z']:.1f}  y={acs['y']:.1f}  x={acs['x']:.1f}")


if __name__ == "__main__":
    main()
