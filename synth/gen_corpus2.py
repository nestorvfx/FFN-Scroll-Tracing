#!/usr/bin/env python3
"""Sheet-composer corpus generator (v2 -- replaces the gap_collapse-based gen_corpus.py mechanic, now in archive/synth/).

Each sample = compose() (real sheets from unrelated cubes, settled into fused contact; intensity calibrated
against a per-seed REAL compact window of the z=10192 slice) -> regen gates -> optional CT-artifact injection
(rings/streaks; the 5-arm artifact arm beat ctrl) -> optional PASTA amplitude-spectrum jitter (syn-to-real
robustness, Chattopadhyay 2022). Output is Dataset059 drop-in nnU-Net format:
  imagesTr/<id>_0000.tif  (CT uint8 192^3)     labelsTr/<id>.tif  (binary {0,1}, inter-sheet boundaries carved)

Intensity refs all come from the one benchmark slice (scroll 1) -- v1 narrowing, partly offset by per-seed
window diversity + PASTA; revisit if the ep15 screen shows scroll-style overfit.

Usage:
  python gen_corpus2.py --cubes <dir> --tif /root/data/10192.tif --out <dataset_dir> --n 400 [--jobs 6]
  python gen_corpus2.py ... --sample_png /root/corpus_samples.png --n 6   (render sample grid, write nothing)
"""
import os, sys, json, argparse
import numpy as np
import tifffile
import multiprocessing as mp
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sheet_compose import compose
from render_blind import ref_windows
from artifacts import inject_artifacts


def complete_labels(ct, inst):
    """Assign bright instance-less voxels to their nearest instance (<=3 vox): fused-contact skins that lost
    the first-claim race, sheet halos written via the pad path, and near-sheet wisps otherwise train as
    background on papyrus-looking material (measured 11-16% of bright voxels vs 1-2% in real cubes).
    Distant debris stays background, like real gap debris the GP annotations never label."""
    from skimage.filters import threshold_otsu
    x = ct.astype(np.float32)
    bright = x >= threshold_otsu(x)
    dist, idx = ndi.distance_transform_edt(inst == 0, return_indices=True)
    grow = (inst == 0) & bright & (dist <= 3.0)
    out = inst.copy()
    out[grow] = inst[idx[0][grow], idx[1][grow], idx[2][grow]]
    # GROWTH MUST NOT CREATE CONTACT (2026-07-20, contact_visibility root cause). The Otsu bar admits the upper
    # half of the partial-volume ramp around a 1-2 vox air gap, so the two sheets' grown skins met INSIDE the dim
    # valley -- manufacturing "touching" wrap pairs whose interface is dim by construction. Measured consequence:
    # synth contact patches ran ~12-27x more locally visible than real ones (median visible frac 0.038-0.083 vs
    # real 0.003) and NO composer-side blending fix moved it (three rounds: ramp width, smooth order, overwrite
    # gate -- all flat), because these contacts are created at LABELING time, not at compositing time. A model
    # can solve them from the local dip, which is exactly the shortcut real blind contacts do not offer.
    # Fix: revert a GROWN voxel that sits in a multi-id neighborhood ONLY IF it is DIM -- i.e. only where the
    # would-be contact runs through valley material. Growth keeps its purpose everywhere else (skins/halos train
    # as fiber, not background), and at BRIGHT fused interfaces (core-swapped at deposit) growth may still meet
    # across the seam: that adjacency is genuine blind-contact supervision and stripping it also thinned deep
    # labels from p50 4.0 to 2.0 (measured against the production-corpus deep baseline -- in the deep stratum
    # every face is a contact face, so an unconditional revert removed ALL face growth). Threshold: core median
    # minus 10 -- the visibility dip definition is 8, so anything the dip metric could flag as a visible seam is
    # dim here, while core-swapped fused interfaces read at core level and keep their growth.
    fg = out > 0
    big = np.where(fg, out, np.iinfo(np.int32).max)
    mn3 = ndi.minimum_filter(big, size=3)
    mx3 = ndi.maximum_filter(np.where(fg, out, 0), size=3)
    multi = fg & (mn3 != np.iinfo(np.int32).max) & (mx3 != mn3)   # neighborhood holds >=2 distinct ids
    dim = x < (float(np.median(x[inst > 0])) - 10.0)              # valley material (clean ct: pre-artifact/pasta)
    out[multi & grow & dim] = 0
    return out


def carve_labels(inst, mode=None):
    """Binary target with carved boundaries wherever two DIFFERENT sheet ids touch -- the trainer's
    BoundaryWeightedCE recomputes its weight map from this on the fly (same contract as Dataset059).
    mode (env CARVE_MODE): 'symmetric' (default) carves BOTH faces at a contact -> 2-vox gap, guaranteed
    separation but erodes 1 vox off each sheet; 'single' carves only the LOWER-id side -> 1-vox seam,
    each sheet loses ~1 vox instead of 2 (thicker survivors, thinner seam -- verify 0-bridges before use)."""
    mode = mode or os.environ.get("CARVE_MODE", "single")   # single (1-vox seam) ablated strictly better than
    # symmetric: post-carve 3.10 vs 2.00 vox, same survivors, 0 bridges (2026-07-19). Restores fiber for the
    # binary/GASP fiber head without any merge. (Affinity GT is on the uncarved inst channel -> carve-agnostic.)
    fg = inst > 0
    big = np.where(fg, inst, np.iinfo(np.int32).max)
    mn = ndi.minimum_filter(big, size=3)
    mx = ndi.maximum_filter(np.where(fg, inst, 0), size=3)
    boundary = fg & (mn < np.iinfo(np.int32).max) & (mx != mn)
    carve = boundary & (inst < mx) if mode == "single" else boundary
    return (fg & ~carve).astype(np.uint8)


def pasta_jitter(ct, rng, alpha=0.22, p=1.0):
    """PASTA: proportional amplitude-spectrum perturbation -- gain noise grows with frequency, phase kept.
    Applied to the uint8 volume, output re-clipped."""
    x = ct.astype(np.float32)
    F = np.fft.rfftn(x)
    zf = np.fft.fftfreq(x.shape[0])[:, None, None]
    yf = np.fft.fftfreq(x.shape[1])[None, :, None]
    xf = np.fft.rfftfreq(x.shape[2])[None, None, :]
    r = np.sqrt(zf * zf + yf * yf + xf * xf) / 0.5
    nb = 24
    bi = np.minimum((r * nb).astype(int), nb - 1)
    gain_b = 1.0 + alpha * (np.arange(nb) / (nb - 1)) ** p * rng.standard_normal(nb)
    gain_b[0] = 1.0                                        # never touch DC
    F *= np.take(np.clip(gain_b, 0.4, 1.8), bi)
    return np.clip(np.fft.irfftn(F, s=x.shape), 0, 255).astype(np.uint8)


def panel_gates(ct, stratum="std", ref=None, fg_band=None):
    cs = ct[:, ct.shape[1] // 2, :].astype(np.float32)
    lv = float(ndi.laplace(ndi.gaussian_filter(cs, 0.5)).var())
    mu = float(cs.mean())
    from skimage.filters import threshold_otsu
    fg = float((cs >= threshold_otsu(cs)).mean())
    # gate bands measured from THIS composition's own reference window, not hard-coded z=10192 constants
    # (audit 3 item 2: fixed 260/118-150 were single-scanner numbers; a Scroll-4-referenced cube must be
    # judged against Scroll 4's statistics or multi-scroll refs are gated back to Scroll-1 style).
    if ref is not None:
        r = np.asarray(ref, np.float32)
        rl = float(ndi.laplace(ndi.gaussian_filter(r, 0.5)).var())
        rm = float(r.mean())
        lv_cap, mlo, mhi = 1.8 * rl, rm - 16.0, rm + 16.0
    else:
        lv_cap, mlo, mhi = 260.0, 118.0, 150.0
    # per-stratum fg bands: std targets the compact calibration windows; deep targets full fused-mottle
    # (real dense windows reach fg~0.9 with no visible layering).
    # deep fhi capped 0.90->0.82: visual validation found fg>=0.85 tail (37-40 wraps/window) is unphysically
    # crushed -- dark inter-wrap gaps vanish across wide bands, asserting more wraps than there is room for
    # fiber+gap. Cap keeps every asserted wrap separable. Env SC_DEEP_FHI to re-tune.
    _dfhi = float(os.environ.get("SC_DEEP_FHI", "0.82"))
    flo, fhi = (0.70, _dfhi) if stratum == "deep" else (0.58, 0.72)
    if fg_band is not None:
        flo, fhi = fg_band          # per-seed density target (diversity; see gen_one)
    pen = (max(0.0, lv - lv_cap) / lv_cap + max(0.0, mlo - mu) / 20.0 + max(0.0, mu - mhi) / 20.0
           + max(0.0, flo - fg) / 0.12 + max(0.0, fg - fhi) / 0.12)
    return pen, dict(lapvar=round(lv, 1), mean=round(mu, 1), fg=round(fg, 3))


_ART_CACHE = {}


def _art_params():
    """Artifact amplitudes MEASURED from the real reference volume (env SC_ART_REF, default the
    held-out slab), cached per process. Falling back to {} keeps the injector's generic defaults."""
    if "p" not in _ART_CACHE:
        try:
            import nrrd
            vol, _ = nrrd.read(os.environ.get("SC_ART_REF", "/root/data/slab/volume.nrrd"))
            from artifacts import measure_artifact_params
            _ART_CACHE["p"] = measure_artifact_params(vol.astype(np.float32), axis=0)
        except Exception:
            _ART_CACHE["p"] = {}
    return _ART_CACHE["p"]


_REFS = None


def _init_pool(refs):
    # reference windows live in a module global set by the pool initializer: the job tuples used
    # to carry ~9 MB of windows EACH (~3.7 GB pickled through the queue on a 400-cube run)
    global _REFS
    _REFS = refs
    os.environ["OMP_NUM_THREADS"] = "1"       # stop FFT/BLAS oversubscription across workers


def gen_one(job):
    cubes, seed, ref, out, sample = job
    if isinstance(ref, list) and ref and isinstance(ref[0], (int, np.integer)):
        ref = [_REFS[j] for j in ref]
    cid0 = f"synthfuse_{seed:05d}"
    if out is not None and os.path.exists(os.path.join(out, "imagesTr", f"{cid0}_0000.tif")) \
            and os.path.exists(os.path.join(out, "labelsTr", f"{cid0}.tif")):
        return cid0, dict(skipped=True)                     # resume-safe: same code+seed = same cube
    # DEEP-COMPACT stratum, ~20% of the corpus (unbiased audit item 1): near-total coverage, earlier fusion
    # onset, harder squeeze -- whole-context fused mottle, the merger-catastrophic regime the model must see.
    deep = np.random.default_rng(seed * 733 + 5).random() < float(os.environ.get("SC_DEEP_P", "0.20"))
    # deep stratum is ALWAYS winding (audit 3 item 3: at zero-evidence fusion the ordering prior is the
    # only learnable signal -- quasi-independent undulation dilutes it exactly where it matters most)
    # squeeze CALIBRATED to real: source cubes p50=5.9 vox, real compact WOUND p50=4.9 -> cf~0.83. The old
    # deep squeeze (0.40,0.62) thinned deep sheets to 2.0 vox (40% of real), which the fixed 1-vox carve then
    # destroyed (67.7% deep fiber erosion). Papyrus is near-incompressible; compaction is air-gap not material.
    # 0.80,0.95 (was 0.74,0.92): compensates the ~0.5 vox of deep label thickness the clamp-2.8 fused-interface
    # claim losses cost (measured 3.5 vs production 4.0; real wound is 4.9 -- thinning further was not acceptable)
    _dsq = tuple(float(x) for x in os.environ.get("SC_DEEP_SQUEEZE", "0.80,0.95").split(","))
    # deep coverage drives fusion EXTENT: 0.85-0.99 fuses the whole window (cores butt globally -> ~40 wraps,
    # pitch ~5, no gaps -- unphysical: real compact still shows clear gaps, fusion is LOCAL not global). Lowering
    # it makes fusion patchy (zero-evidence contacts where the field is high, real gaps elsewhere) -> realistic
    # ~12-20 wraps that still teach the hard contact. Env SC_DEEP_COV to tune.
    _dcov = tuple(float(x) for x in os.environ.get("SC_DEEP_COV", "0.85,0.99").split(","))
    kw = dict(coverage=_dcov, p_spacer=0.01, fuse_onset=0.32, squeeze=_dsq,
              winding=True) if deep else {}
    # DIVERSITY AS CONTROL, not rejection (advisor 1.0/3.1): the per-seed band and the composer's
    # density draws came from the same seed, so the band could only REJECT compositions -- the airy
    # stratum was empty (render "airy" panel was pixel-identical to the dense one) and every miss
    # cost a full compose. Draw the target FIRST, steer the composer with it, keep the gate as a
    # widened final check.
    band = fgt = None
    if os.environ.get("SC_FG_DIVERSE", "0") == "1" and not deep:
        _lo = float(np.random.default_rng(seed ^ 0x51D).uniform(0.30, 0.62))
        fgt = _lo + 0.07
        band = (_lo - 0.04, _lo + 0.18)
    elif deep:
        # deep stratum steering (measured: unsteered fill_frac draws 0.55-0.97 gave deep fg
        # 0.50-0.55 vs the 0.70 band -- an "airy deep" contradiction and pen ~1.9 thrash).
        # fgt=0.75 pins fill_frac at its cap and p_spacer at its floor; density beyond that is
        # the fusion/coverage machinery's job, so the accept band starts at what the composer
        # can actually reach.
        fgt = 0.75
        band = (0.58, float(os.environ.get("SC_DEEP_FHI", "0.82")))
    s, best = seed, None
    for _ in range(2):
        ct, inst, meta = compose(cubes, s, hist_ref=ref, fg_target=fgt, **kw)
        pen, stats = panel_gates(ct, "deep" if deep else "std", ref=ref, fg_band=band)
        if best is None or pen < best[0]:
            best = (pen, ct, inst, meta, stats, s)
        if pen < 0.6:
            break             # near-band accepts skip the second full compose (profile: a retry
        s += 9973             # doubles cube cost; the gate still keeps best-of on real misses)
    pen, ct, inst, meta, stats, s_used = best
    stats["stratum"] = "deep" if deep else "std"
    inst = complete_labels(ct, inst)                        # against the CLEAN ct, before augmentation
    lab = carve_labels(inst)
    rng = np.random.default_rng(seed * 131 + 17)
    aug = []
    if rng.random() < 0.45:                                # CT artifacts on ~half (5-arm winning arm)
        ct = np.clip(inject_artifacts(ct.astype(np.float32), rng, axis=0,
                                      params=_art_params()), 0, 255).astype(np.uint8)
        aug.append("artifact")
    if rng.random() < 0.5:                                  # PASTA spectral jitter on ~half
        ct = pasta_jitter(ct, rng)
        aug.append("pasta")
    cid = f"synthfuse_{seed:05d}"
    if sample is None:
        tifffile.imwrite(os.path.join(out, "imagesTr", f"{cid}_0000.tif"), ct)
        tifffile.imwrite(os.path.join(out, "labelsTr", f"{cid}.tif"), lab)
        # persist per-sheet instance ids for the LSD/affinity aux head (append_inst_channel.py expects
        # labelsTr_inst/; the audit flagged the v2 corpus was silently opting out). uint16: sheets <= 140.
        tifffile.imwrite(os.path.join(out, "labelsTr_inst", f"{cid}.tif"), inst.astype(np.uint16))
        return cid, dict(seed_used=s_used, pen=round(pen, 4), aug=aug, sheets=meta["sheets"], **stats)
    return cid, (ct, inst, lab, dict(seed_used=s_used, aug=aug, sheets=meta["sheets"], **stats))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubes", required=True); ap.add_argument("--tif", required=True)
    ap.add_argument("--out", default=None); ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--jobs", type=int, default=6); ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--sample_png", default=None)
    a = ap.parse_args()
    # multi-scroll intensity refs (audit issue 3: single-slice anchoring): --tif takes a comma-separated list;
    # per-seed reference windows are drawn from all of them round-robin, so the corpus spans scroll styles.
    refs = []
    for ti, tp in enumerate(a.tif.split(",")):
        img = tifffile.imread(tp.strip()).astype(np.float32)
        lo, hi = np.percentile(img, [1, 99.5])
        img = np.clip((img - lo) / (hi - lo + 1e-6) * 255, 0, 255)
        got = None
        for _sz in (480, 320, 240, 192, 138):     # fall back window size: a sparse/benchmark-slab
            got = ref_windows(img, max(a.n, 8), np.random.default_rng(4242 + 97 * ti), size=_sz)
            if got:                               # reference can fail the 480^2 fg gate while still
                break                             # holding valid smaller compact windows
        if not got:
            raise SystemExit(f"--tif {a.tif}: no reference windows at any size")
        refs.append(got)
    refs = [r for group in zip(*refs) for r in group] if len(refs) > 1 else refs[0]

    if a.sample_png:
        jobs = [(a.cubes, a.seed0 + i,
                 [refs[(i + j) % len(refs)] for j in range(min(len(refs), 10))], None, True)
                for i in range(a.n)]
        with mp.Pool(min(a.jobs, a.n)) as p:
            res = p.map(gen_one, jobs)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(res)
        fig, axes = plt.subplots(3, n, figsize=(3.4 * n, 10.6))
        axes = np.atleast_2d(axes)
        cmap = plt.get_cmap("tab20")
        for c, (cid, (ct, inst, lab, meta)) in enumerate(res):
            cs = ct[:, ct.shape[1] // 2, :]
            ci = inst[:, inst.shape[1] // 2, :]
            cl = lab[:, lab.shape[1] // 2, :]
            axes[0, c].imshow(cs, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            axes[0, c].set_title(f"{cid} {'+'.join(meta['aug']) or 'clean'}\n"
                                 f"fg={meta['fg']} sheets={meta['sheets']}", fontsize=7)
            ov = np.zeros(ci.shape + (4,), np.float32)
            for k, iid in enumerate(sorted(np.unique(ci[ci > 0]))):
                m = ci == iid
                ov[m, :3] = cmap(k % 20)[:3]; ov[m, 3] = 0.85
            axes[1, c].imshow(cs, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            axes[1, c].imshow(ov, interpolation="nearest")
            axes[2, c].imshow(cl, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            for r in range(3):
                axes[r, c].axis("off")
        for r, lab_ in enumerate(["CT (model input)", "sheets colored (separate layers)",
                                  "training label (binary, boundaries carved)"]):
            axes[r, 0].axis("on"); axes[r, 0].set_ylabel(lab_, fontsize=8)
            axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        plt.tight_layout()
        plt.savefig(a.sample_png, dpi=130, bbox_inches="tight")
        print("wrote", a.sample_png)
        return

    os.makedirs(os.path.join(a.out, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "labelsTr"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "labelsTr_inst"), exist_ok=True)
    # STRATIFY the per-seed reference pool across the fg range: a consecutive slice can miss the
    # airy stratum entirely, and one such seed re-triggers the dense-mapping air-tail failure
    # (measured: 9/10 cubes D2 <= 0.05%, one pool-unlucky cube at 5.6%).
    from skimage.filters import threshold_otsu as _ot
    def _wfg(w):
        w32 = np.asarray(w, np.float32)
        try:
            return float((w32 >= _ot(w32)).mean())
        except Exception:
            return 0.5
    order = np.argsort([_wfg(w) for w in refs])
    half = (len(order) + 1) // 2
    inter = []
    for k in range(half):                       # low, high, low, high ... spans the range everywhere
        inter.append(order[k])
        if len(order) - 1 - k > k:
            inter.append(order[len(order) - 1 - k])
    refs = [refs[i] for i in inter]
    pool_k = min(len(refs), 10)
    jobs = [(a.cubes, a.seed0 + i,
             [(i + j) % len(refs) for j in range(pool_k)], a.out, None) for i in range(a.n)]
    metas = {}
    with mp.Pool(a.jobs, initializer=_init_pool, initargs=(refs,)) as p:
        for cid, meta in p.imap_unordered(gen_one, jobs):
            metas[cid] = meta
            if len(metas) % 20 == 0:
                print(f"{len(metas)}/{a.n}", flush=True)
    json.dump(metas, open(os.path.join(a.out, "synthfuse_meta.json"), "w"), indent=1)
    print(f"done: {len(metas)} cubes -> {a.out}")


if __name__ == "__main__":
    main()
