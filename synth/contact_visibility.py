#!/usr/bin/env python3
"""Is each wrap-wrap CONTACT visible SOMEWHERE along its extent, or blind end-to-end?

This is the decisive dataset question behind "is the separation learnable". A human separating fused wraps does
not find a gap at the fused spot -- they FOLLOW the sheet: they see where it is separable before/after the fused
stretch and propagate the boundary through it. A conv net can learn the same propagation, but ONLY if the training
contacts contain a visible ANCHOR to propagate from.

  * If most contacts are PARTIALLY visible (some stretch has an intensity dip, the rest is fused) -> the inference
    is learnable from a 192^3 patch, and any failure is a technique/supervision problem.
  * If most contacts are UNIFORMLY blind end-to-end -> we made the task unlearnable BY CONSTRUCTION in the
    synthesizer, and no loss/architecture change can fix it; the composer must be changed to produce contacts with
    visible anchors (which is also what real compact regions look like).

Per wrap-PAIR (and per connected contact PATCH), we measure the fraction of contact voxels with a visible dip:
    dip = min(core_A, core_B) - interface,  where interface = min(vol[v], vol[v+d]) at the touching pair and
    core_A/core_B are one step further into each wrap (v-d and v+d+d). dip>>0 = a real intensity gap.

Usage:
  python contact_visibility.py --corpus /root/surf/data/synthfuse_corpus --n 8        # SYNTH training data
  python contact_visibility.py --slab /root/data/slab                                  # REAL slab
"""
import os, sys, glob, json, argparse
import numpy as np
from scipy import ndimage as ndi


def contact_stats(vol, inst, dip_thr=8.0, min_contact=200):
    """Returns per-wrap-pair and per-contact-patch visible fractions."""
    Z, Y, X = inst.shape
    pair_tot, pair_vis = {}, {}
    patch_fracs = []
    for d in ([0, 0, 1], [0, 1, 0], [1, 0, 0]):
        dz, dy, dx = d
        s = (slice(0, Z - dz), slice(0, Y - dy), slice(0, X - dx))
        t = (slice(dz, Z), slice(dy, Y), slice(dx, X))
        ia, ib = inst[s], inst[t]
        m = (ia > 0) & (ib > 0) & (ia != ib)               # touching voxels of two DIFFERENT wraps
        if not m.any():
            continue
        va, vb = vol[s], vol[t]
        interface = np.minimum(va, vb)
        # core voxels one step FURTHER into each wrap: core_A = vol[v-d], core_B = vol[v+2d]. All offsets here are
        # unit, so index along the single active axis; out-of-range positions fall back to the interface value so
        # they contribute dip 0 (conservatively "blind") rather than a spurious gap.
        ax = int(np.argmax([dz, dy, dx]))
        N = vol.shape[ax]
        S = va.shape                                        # length N-1 along ax

        def _sl(start, stop):
            q = [slice(None)] * 3
            q[ax] = slice(start, stop)
            return tuple(q)

        core_a = np.full(S, np.inf, np.float32)
        core_a[_sl(1, None)] = vol[_sl(0, N - 2)]           # v-d
        core_b = np.full(S, np.inf, np.float32)
        core_b[_sl(0, S[ax] - 1)] = vol[_sl(2, N)]          # v+2d
        core = np.minimum(core_a, core_b)
        core = np.where(np.isfinite(core), core, interface)
        dip = core - interface
        vis = m & (dip > dip_thr)
        # per wrap-pair totals
        a_ids, b_ids = ia[m], ib[m]
        lo = np.minimum(a_ids, b_ids); hi = np.maximum(a_ids, b_ids)
        keys = lo.astype(np.int64) * 100000 + hi.astype(np.int64)
        vk = vis[m]
        for k, v in zip(keys, vk):
            pair_tot[k] = pair_tot.get(k, 0) + 1
            pair_vis[k] = pair_vis.get(k, 0) + int(v)
        # per connected contact PATCH (a single physical contact region). Vectorized with bincount: the obvious
        # `for pid in range(n): lab == pid` is O(n_patches * n_voxels) and hangs for hours on the real slab, which
        # has ~1e4 contact patches over ~2e8 voxels.
        lab, n = ndi.label(m)
        if n:
            flat = lab[m]                                   # patch id of every contact voxel
            tot = np.bincount(flat, minlength=n + 1)
            visc = np.bincount(flat, weights=vis[m].astype(np.float64), minlength=n + 1)
            keep = tot >= min_contact
            keep[0] = False                                 # label 0 is background
            patch_fracs += (visc[keep] / tot[keep]).tolist()
    pairs = [(pair_vis[k] / pair_tot[k], pair_tot[k]) for k in pair_tot if pair_tot[k] >= min_contact]
    return pairs, patch_fracs


def summarize(tag, pairs, patches):
    if not pairs:
        print(f"{tag}: no contacts"); return None
    fr = np.array([p[0] for p in pairs])
    res = dict(tag=tag, n_pairs=len(fr),
               pairs_fully_blind=round(float((fr < 0.01).mean()), 4),
               pairs_under_5pct=round(float((fr < 0.05).mean()), 4),
               pairs_over_20pct=round(float((fr > 0.20).mean()), 4),
               visible_frac_p10=round(float(np.percentile(fr, 10)), 4),
               visible_frac_p50=round(float(np.percentile(fr, 50)), 4),
               visible_frac_p90=round(float(np.percentile(fr, 90)), 4))
    if patches:
        pf = np.array(patches)
        res.update(n_patches=len(pf),
                   patches_fully_blind=round(float((pf < 0.01).mean()), 4),
                   patch_visible_p50=round(float(np.percentile(pf, 50)), 4))
    print(f"{tag}: pairs={res['n_pairs']}  FULLY-BLIND pairs {res['pairs_fully_blind']:.1%}  "
          f"<5% visible {res['pairs_under_5pct']:.1%}  >20% visible {res['pairs_over_20pct']:.1%}  "
          f"| visible-frac p10/p50/p90 {res['visible_frac_p10']}/{res['visible_frac_p50']}/{res['visible_frac_p90']}")
    if patches:
        print(f"{tag}: contact PATCHES={res['n_patches']}  fully-blind {res['patches_fully_blind']:.1%}  "
              f"median visible frac {res['patch_visible_p50']}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus"); ap.add_argument("--slab")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--dip_thr", type=float, default=8.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = {}
    if a.slab:
        import nrrd
        vol = nrrd.read(os.path.join(a.slab, "volume.nrrd"))[0].astype(np.float32)
        truth = nrrd.read(os.path.join(a.slab, "truth.nrrd"))[0].astype(np.int32)
        p, q = contact_stats(vol, truth, a.dip_thr)
        out["real_slab"] = summarize("REAL slab", p, q)
    if a.corpus:
        import tifffile
        allp, allq = [], []
        for f in sorted(glob.glob(f"{a.corpus}/labelsTr_inst/*.tif"))[:a.n]:
            cid = os.path.basename(f)[:-4]
            img = glob.glob(f"{a.corpus}/imagesTr/{cid}_0000.tif")
            if not img:
                continue
            vol = tifffile.imread(img[0]).astype(np.float32)
            inst = tifffile.imread(f).astype(np.int32)
            p, q = contact_stats(vol, inst, a.dip_thr)
            allp += p; allq += q
        out["synth"] = summarize("SYNTH corpus", allp, allq)
    print("\nINTERPRETATION: if 'FULLY-BLIND pairs' is high, contacts have NO visible anchor to propagate from and "
          "the task is unlearnable AS PRESENTED -> fix the synthesizer. If most contacts are partially visible, "
          "propagation is learnable and the defect is training technique/supervision.")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    print("CONTACT_VISIBILITY_DONE")


if __name__ == "__main__":
    main()
