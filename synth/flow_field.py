#!/usr/bin/env python3
"""Interior distance + eikonal FLOW field: the SOTA representation for separating fused laminar sheets.

WHY (SOTA_PLAN.md, reconciled 2026-07-20): axis-aligned per-offset affinities are degenerate for a spiral (the
sheet normal rotates, so no fixed offset tracks the same-vs-next-wrap decision) and a local per-voxel head cannot
resolve a blind (zero-gap) contact -- a calibrated posterior there converges to the majority prior = a MERGE. A
DENSE FIELD fixes both: it is orientation-free, and predicting a spatially-coherent field forces the network to
integrate its receptive field so a boundary decision made at a visible anchor PROPAGATES along the sheet into a
blind stretch. This is the Omnipose mechanism (Cutler et al., Nat. Methods 2022), which is SOTA precisely for
elongated, densely-packed, TOUCHING objects -- the closest morphological analog to laminated papyrus.

THE TARGET (computed from GT instances; synth has them, real gives them on separated regions -- see mask note):
  D(v)    = interior distance transform = distance from v to the boundary of ITS OWN wrap. At a fused seam the two
            wraps have DIFFERENT ids, so each wrap's own boundary runs along the seam and D dips to ~0 there EVEN
            WITH ZERO INTENSITY GAP. That valley is the separation signal, present by construction.
  flow(v) = unit-normalized gradient of D. The two wraps' flows point in OPPOSITE directions at a seam (each
            "uphill" toward its own core), so the flow diverges exactly at the wall -> a watershed ridge.

THE DECODE (no instances -- uses only predicted D + fiber):
  cores   = fiber & (D >= core_frac * local max)   -> one connected component per wrap (the seam valley splits
            adjacent wraps into separate cores, because D dipped to ~0 between them).
  labels  = marker-controlled watershed on -D within the fiber mask, seeded by the cores. Seeds from distance
            maxima => does NOT over-fragment the way signed-affinity Mutex-Watershed does (that turned every
            uncertain long-range affinity into a mutex edge -> 81,475 instances / 68 wraps).
  carve   = zero fiber voxels 26-adjacent to a different label -> the explicit seam the tracer consumes.

This module is dependency-light (numpy/scipy/skimage) so the TARGET and DECODE logic are unit-tested offline,
before any GPU. The network head that predicts (D, flow) [+ LSD, + embedding] is wired in the trainer separately.

Run:  python flow_field.py --selftest
"""
import sys, argparse
import numpy as np
from scipy import ndimage as ndi


# ------------------------------------------------------------------ TARGET
def flow_target(inst, normalize=True, eps=1e-6):
    """inst (Z,Y,X) int instance labels (0=bg) -> (D, flow).
      D    : (Z,Y,X) float, per-wrap interior distance (0 at each wrap's own boundary, incl. fused seams).
      flow : (3,Z,Y,X) float, unit gradient of D (0 where |grad| ~ 0, e.g. bg and ridges).
    Per-instance EDT so a fused seam (two different ids touching) reads as a boundary for BOTH wraps."""
    inst = np.asarray(inst)
    Z, Y, X = inst.shape
    D = np.zeros((Z, Y, X), np.float32)
    for k in np.unique(inst):
        if k == 0:
            continue
        m = inst == k
        # EDT of the instance mask = distance from each of its voxels to the nearest NON-k voxel (bg OR another
        # wrap). At a fused seam the nearest non-k voxel is the touching other wrap -> distance ~ 0 there.
        D[m] = ndi.distance_transform_edt(m)[m]
    g = np.gradient(D)                                    # [dz, dy, dx]
    flow = np.stack(g, 0).astype(np.float32)
    if normalize:
        n = np.sqrt((flow * flow).sum(0, keepdims=True))
        flow = np.where(n > eps, flow / np.maximum(n, eps), 0.0).astype(np.float32)
    return D, flow


# ------------------------------------------------------------------ DECODE
def _cores(D, fiber, core_frac, min_core):
    """One marker per wrap: threshold the distance so only sheet cores survive, then connected-component them.
    Adjacent wraps have their cores separated by the seam's D-valley, so CC gives distinct markers."""
    thr = core_frac * float(D[fiber].max()) if fiber.any() else 0.0
    core = fiber & (D >= max(thr, 1e-6))
    markers, n = ndi.label(core, structure=np.ones((3, 3, 3), bool))
    if min_core > 1 and n > 0:                            # drop specks so they don't seed spurious instances
        counts = np.bincount(markers.ravel())
        small = np.where(counts < min_core)[0]
        small = small[small > 0]
        if small.size:
            markers[np.isin(markers, small)] = 0
            markers, _ = ndi.label(markers > 0, structure=np.ones((3, 3, 3), bool))
    return markers


def flow_decode(D, fiber, core_frac=0.5, min_core=8):
    """(predicted D, fiber mask) -> instance labels via marker-controlled watershed on -D within fiber.
    Rank/relative (uses D's own max), so it is IMMUNE to a global scale shift of the predicted distance -- the
    exact failure that erased the fiber when an absolute affinity threshold met a globally-collapsed head."""
    from skimage.segmentation import watershed
    fiber = fiber.astype(bool)
    if not fiber.any():
        return np.zeros_like(D, np.int32)
    markers = _cores(D, fiber, core_frac, min_core)
    if markers.max() == 0:                                # no core survived: fall back to fiber CCs (no split)
        lab, _ = ndi.label(fiber, structure=np.ones((3, 3, 3), bool))
        return lab.astype(np.int32)
    labels = watershed(-D, markers, mask=fiber)
    return labels.astype(np.int32)


def flow_target_fast(inst, normalize=True, eps=1e-6):
    """Same (D, flow) target as flow_target but in ONE distance transform instead of one-per-instance -- fast enough
    to recompute per augmented batch during training (per-instance EDT would ~double epoch time).

    Trick: the per-instance interior distance = distance to the nearest voxel that is bg OR a DIFFERENT wrap. Build
    that seed set once (bg + inter-wrap SEAM voxels), then a single EDT gives every fiber voxel its distance to its
    own wrap's boundary -- valley at fused seams included. Seam = a fiber voxel whose 3x3x3 neighborhood holds >=2
    distinct wrap ids (same criterion as carve_labels), so no np.roll wrap-around contamination."""
    inst = np.asarray(inst)
    fg = inst > 0
    D = np.zeros(inst.shape, np.float32)
    flow = np.zeros((3,) + inst.shape, np.float32)
    if not fg.any():
        return D, flow
    BIG = np.iinfo(np.int32).max
    big = np.where(fg, inst, BIG)
    mn = ndi.minimum_filter(big, size=3)
    mx = ndi.maximum_filter(np.where(fg, inst.astype(np.int64), 0), size=3)
    seam = fg & (mn != BIG) & (mn != mx)                 # neighborhood holds >=2 distinct wrap ids
    seed = (~fg) | seam                                  # distance measured FROM bg and from every inter-wrap seam
    D = (ndi.distance_transform_edt(~seed).astype(np.float32)) * fg
    g = np.gradient(D)
    flow = np.stack(g, 0).astype(np.float32)
    if normalize:
        n = np.sqrt((flow * flow).sum(0, keepdims=True))
        flow = np.where(n > eps, flow / np.maximum(n, eps), 0.0).astype(np.float32)
    return D, flow


def thin_region_mask(fiber, single_vox=5.0, tol=0.4):
    """Where can the flow TARGET be trusted on REAL binary labels (no instances)?

    A single separated sheet's interior distance IS computable from the binary mask alone (it is just the sheet's
    own EDT). But a FUSED region reads as one thick slab -> its EDT has ONE ridge down the middle -> it would teach
    a MERGE. So on real cubes we train the field only where the local slab thickness ~ a single sheet, and MASK OUT
    thick (fused) regions. Synth (which has instances) supervises the fused seams; real supplies the anchor/prior on
    the easy regions and closes the 82%-zero-gradient domain gap.

    Returns a bool mask (True = trustworthy/thin). Uses the sphere-fitting (Hildebrand) LOCAL THICKNESS: thickness
    at v = diameter of the largest inscribed sphere that covers v = 2*max{r : some voxel w with EDT(w)>=r has
    |v-w|<=r}. Robust for both thin sheets and thick fused slabs (a nearest-ridge proxy is not). thin iff
    thickness <= single_vox*(1+tol)."""
    fiber = fiber.astype(bool)
    if not fiber.any():
        return np.zeros_like(fiber)
    dt = ndi.distance_transform_edt(fiber)
    th = np.zeros_like(dt)
    radii = np.unique(np.round(dt[dt > 0] * 2) / 2.0)[::-1]   # descending, quantized to 0.5 vox
    for r in radii:
        if r <= 0:
            continue
        reached = ndi.distance_transform_edt(dt < r) <= r    # within radius r of a voxel whose EDT>=r
        upd = fiber & reached & (th == 0)
        th[upd] = 2.0 * r
    return fiber & (th <= single_vox * (1.0 + tol)) & (th > 0)


def carve_from_labels(labels, fiber, fiber_prob=None):
    """Zero fiber voxels 26-adjacent to a DIFFERENT positive instance -> the explicit ~1-2vox seam for the tracer.
    Returns (carved_prob, carved_binary, carve_mask). Mirrors slab_emit_carved.carve_from_labels."""
    big = np.where(labels > 0, labels, np.iinfo(np.int32).max)
    mn = ndi.minimum_filter(big, size=3)
    mx = ndi.maximum_filter(np.where(labels > 0, labels, 0), size=3)
    carve = (labels > 0) & (mn < np.iinfo(np.int32).max) & (mx != mn)
    carved_binary = fiber & ~carve
    if fiber_prob is None:
        fiber_prob = fiber.astype(np.float32)
    return np.where(carved_binary, fiber_prob, 0.0).astype(np.float32), carved_binary, carve


# ------------------------------------------------------------------ SELFTEST
def selftest():
    """Phantom: two 5-vox-thick sheets in ZERO-GAP fused contact + a third across a 4-vox air gap. Intensity shows
    NO seam at the fused contact. The flow TARGET (from GT instances) must encode the seam; the DECODE (from the
    distance field alone, no instances) must (1) separate the two fused sheets, (2) not bridge the air gap, (3)
    place >=80% of the carve within 1 vox of the true seam, (4) not dig into wrap cores, (5) survive a global scale
    shift of the distance field. Same assertion structure as slab_emit_carved.py's selftest, ported to the flow
    representation."""
    Z, Y, X = 24, 40, 48
    gt = np.zeros((Z, Y, X), np.int32)
    gt[:, 5:10, :] = 1                                    # wrap 1  (5 vox thick)
    gt[:, 10:15, :] = 2                                   # wrap 2, ZERO-gap fused contact at the y=10 plane
    gt[:, 19:24, :] = 3                                   # wrap 3 across a 4-vox air gap (y=15..18 is bg)
    fiber = gt > 0

    D, flow = flow_target(gt)
    # (T1) target: distance dips to its floor (~1, a boundary voxel's EDT) at the fused seam and RISES to a ridge
    # inside each sheet. The separation mechanism is the RELATIVE valley (seam << core), not an absolute zero.
    seam_D = D[:, 9:11, :].mean()                         # the two touching boundary planes
    core1_D = D[:, 6:8, :].mean(); core2_D = D[:, 12:14, :].mean()
    assert seam_D <= 1.2, f"target D must dip to its floor at the fused seam, got {seam_D:.2f}"
    assert min(core1_D, core2_D) > 2.0, f"target D must ridge inside each sheet, got {core1_D:.2f}/{core2_D:.2f}"
    assert seam_D < 0.6 * min(core1_D, core2_D), f"seam valley not deep enough vs core ({seam_D:.2f} vs cores)"
    # (T2) flow DIVERGES at the seam: in the sheet half just below the seam (wrap1, y=8) the flow points back down
    # toward wrap1's core (dy<0); just above (wrap2, y=11) it points back up toward wrap2's core (dy>0). Sampled OFF
    # the core ridges (y=7,12), where |grad D|=0 by construction. Opposite signs => a watershed ridge at the seam.
    fy_below = flow[1][:, 8, :].mean()                   # wrap1 half adjacent to the seam
    fy_above = flow[1][:, 11, :].mean()                  # wrap2 half adjacent to the seam
    assert fy_below < 0 < fy_above, f"flow must diverge across the seam, got {fy_below:.2f} / {fy_above:.2f}"
    print(f"target: seam_D {seam_D:.2f} < core_D {core1_D:.2f}/{core2_D:.2f} | flow reverses "
          f"{fy_below:+.2f}->{fy_above:+.2f}")

    # (D1)+(D2) decode from the DISTANCE FIELD ALONE separates the fused pair and not across the gap
    labels = flow_decode(D, fiber, core_frac=0.5)
    dom = {}
    for w in (1, 2, 3):
        vals, cnts = np.unique(labels[(gt == w) & (labels > 0)], return_counts=True)
        assert vals.size, f"wrap {w} got no label"
        dom[w] = int(vals[np.argmax(cnts)]); purity = cnts.max() / cnts.sum()
        assert purity > 0.9, f"wrap {w} fragmented (purity {purity:.2f})"
    assert dom[1] != dom[2], "decode MERGED the two zero-gap fused wraps"
    assert dom[3] not in (dom[1], dom[2]), "decode merged across the air gap"
    n_inst = len(np.unique(labels[labels > 0]))
    print(f"decode: {n_inst} instances; wraps map to distinct labels {dom}")

    # (D3)+(D4) carve localization
    _, carved_binary, carve = carve_from_labels(labels, fiber)
    seam_band = np.zeros_like(fiber); seam_band[:, 8:12, :] = True     # +-2 vox of the true y=10 seam
    at_seam = float((carve & seam_band).sum()) / max(1, int(carve.sum()))
    core = (gt > 0) & ~seam_band
    core_damage = float((carve & core).sum()) / float(core.sum())
    assert at_seam >= 0.8, f"carve not localized at the seam ({at_seam:.2f})"
    assert core_damage < 0.02, f"carve digs into wrap cores ({core_damage:.4f})"
    seam_open = float((carved_binary[:, 10, :] == 0).mean())
    assert seam_open > 0.5, f"seam plane not opened ({seam_open:.2f})"
    print(f"carve: at-seam {at_seam:.2f}, core-damage {core_damage:.4f}, seam-open {seam_open:.2f}")

    # (D5) global scale-shift invariance: multiplying the whole distance field by a constant must not change the
    # decode at all (relative/rank cores). This is the property the absolute affinity carve lacked.
    labels_shift = flow_decode(D * 0.1, fiber, core_frac=0.5)
    assert np.array_equal(labels_shift, labels), "decode must be invariant to a global scale of the distance field"
    print("scale-invariance: decode identical under D*0.1")

    # (D6) robustness to prediction noise: add noise to D, decode must still separate the fused pair
    rng = np.random.default_rng(0)
    Dn = np.clip(D + rng.normal(0, 0.15, D.shape).astype(np.float32), 0, None)
    ln = flow_decode(Dn, fiber, core_frac=0.5)
    d1 = np.bincount(ln[(gt == 1) & (ln > 0)]).argmax()
    d2 = np.bincount(ln[(gt == 2) & (ln > 0)]).argmax()
    assert d1 != d2, "noisy-D decode merged the fused pair"
    print("noise robustness: fused pair still separated under sigma=0.15 D noise")

    # (U8) real-data thin-region mask: a SINGLE separated 5-vox sheet is trustworthy (thin -> trainable on real
    # binary); a FUSED 10-vox double-slab reads as thick -> masked out (its binary EDT would teach one medial
    # surface = a merge). This is what lets the field train on the 1754 real cubes without instance labels.
    rm = np.zeros((16, 40, 40), np.int32)
    rm[:, 6:11, :] = 1                                    # a lone 5-vox sheet (separated) -> THIN
    rm[:, 20:30, :] = 1                                   # a 10-vox fused double-slab      -> THICK
    fb = rm > 0
    thin = thin_region_mask(fb, single_vox=5.0, tol=0.4)
    thin_cov = float(thin[:, 6:11, :].mean())            # lone sheet should be mostly trainable
    thick_cov = float(thin[:, 22:28, :].mean())          # fused slab core should be masked out
    assert thin_cov > 0.8, f"lone thin sheet should be trainable on real, got {thin_cov:.2f}"
    assert thick_cov < 0.2, f"fused slab core must be masked on real, got {thick_cov:.2f}"
    print(f"real-mask: lone sheet trainable {thin_cov:.2f}, fused core masked {1-thick_cov:.2f}")

    # (U1b) the FAST one-EDT target must reproduce the separation behavior of the per-instance target: seam valley
    # below core, and its decode still splits the fused pair. This is the target the trainer actually recomputes.
    Df, flowf = flow_target_fast(gt)
    seam_f = Df[:, 9:11, :].mean(); core_f = min(Df[:, 6:8, :].mean(), Df[:, 12:14, :].mean())
    assert seam_f < 0.6 * core_f, f"fast target seam valley too shallow ({seam_f:.2f} vs core {core_f:.2f})"
    lf = flow_decode(Df, fiber, core_frac=0.5)
    f1 = np.bincount(lf[(gt == 1) & (lf > 0)]).argmax(); f2 = np.bincount(lf[(gt == 2) & (lf > 0)]).argmax()
    f3 = np.bincount(lf[(gt == 3) & (lf > 0)]).argmax()
    assert f1 != f2 and f3 not in (f1, f2), "fast-target decode failed to separate the fused pair / gap"
    print(f"fast target: seam {seam_f:.2f} < core {core_f:.2f}; decode separates fused pair (one EDT)")
    print("SELFTEST_OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        print("nothing to do; run --selftest")
