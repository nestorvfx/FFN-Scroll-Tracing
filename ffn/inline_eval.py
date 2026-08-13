"""Inline validation during training: a live NERL + merge-rate curve on UNCONTAMINATED data.

Training loss cannot show whether the net separates wraps (the flow arm had great loss
and ~80% merges), so every cfg.eval_every steps we run a real decode on a small FIXED
panel of held-out cubes. We log BOTH:
  - NERL (Expected Run Length, normalized) -- the PRIMARY selection metric. It has a
    coverage floor: an uncovered skeleton node zeros its run, so a collapsed/empty
    segmentation scores NERL~0 (worst). This is what train.py maximizes for ckpt_best.
  - adjacent-wrap merge rate -- a CONSTRAINT, not the objective. It is coverage-BLIND
    (an empty segmentation scores merge 0.0 = optimum), which is exactly how run-1's
    collapse looked "better". Selection = max NERL among evals with merge <= eps
    (the Januszewski et al. 2018 FFN protocol).
Contamination-free by construction: cubes come only from splits.json["val"] (synth-val +
real-val crops whose whole PARENT cubes are held out); the eval slab is never touched here.

Kept cheap so it doesn't dent training throughput:
  - 3 real + 3 synth val cubes, 128^3 center crops;
  - seeds precomputed ONCE from the GT foreground (fixed spacing/cap) -> evals at
    different steps are directly comparable (no seed-noise jitter);
  - single-direction single-scale flood-fill decode (no consensus). This is HARSHER on
    merges than the consensus used at final eval -- a merge here is a real merge signal;
  - the panel is split across DDP ranks (each rank decodes half), counts all-reduced.

GT adjacency pairs, pair sizes and the blind/gapped classification of each real pair are
precomputed at init; per-eval work is just the decode + a vectorized contingency.
"""
from __future__ import annotations

import dataclasses
import glob
import json
import os
import time

import numpy as np
import torch

from . import inference as I
from . import seeds as S
from .metrics import adjacency_pairs, _pair_is_blind, erl as erl_metric
from .ctstats import air_papyrus_threshold, wrap_period


def measured_lambda(ct) -> float:
    """Validated wrap-period estimator, routed from tracer.rescale.measure_lambda.

    THE BUG THIS FIXES: `_core_validity` scaled its thickness gate by `ctstats.wrap_period`,
    whose autocorrelation locks onto a SUB-HARMONIC (its own docstring records lambda~69 on
    Scroll-3 cores; the validated structure-tensor + run-length estimator measures 10.50, see
    research/09 + research/10 A.1 and tracer/rescale.py's own warning). The 0.6*lambda validity
    gate was therefore ~6.6x too permissive on Scroll-3 and ~1.8x on Scroll-4 -- `core_score`
    and `core_valid_frac`, the only deployment-regime metrics, were mis-scaled rulers.

    Cross-check: when the two estimators disagree by >30% a loud warning is printed (once per
    call site) so silent regressions of either estimator are visible. Returns 0.0 = unknown,
    same contract as before (callers exclude the cube)."""
    import os
    import sys
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if root not in sys.path:
        sys.path.append(root)
    from tracer.rescale import measure_lambda as _ml
    lam = float(_ml(np.asarray(ct)))
    try:
        lam_ac = float(wrap_period(np.asarray(ct)))
        if lam > 0 and lam_ac > 0 and abs(lam_ac - lam) / lam > 0.30:
            print(f"[val] WARNING wrap-period estimators disagree: measure_lambda={lam:.2f} "
                  f"vs ctstats.wrap_period={lam_ac:.2f} (autocorr sub-harmonic lock; using "
                  f"measure_lambda)", flush=True)
    except Exception:
        pass
    return lam


def fair_seeds(coords: np.ndarray, inst: np.ndarray, cap: int, per_inst: int) -> np.ndarray:
    """Seed budget that does not starve thin instances.

    `inference_seeds` returns coords in DESCENDING-EDT order, so a plain `coords[:cap]` keeps the
    globally thickest voxels -- and those cluster on the few fattest wraps. MEASURED at ckpt_405000
    on val cubes (128^3, cap=200):

        cube              nGT   coverage   nPred   NERL
        real ...c0          8      0.607       4   0.4714
        synthfuse_05000    23      0.388       6   0.0538
        synthfuse_05013    28      0.382       6   0.0000

    Six committed objects for 28 ground-truth instances: most instances were never seeded at all,
    and those that were got swallowed into objects spanning several wraps, so every run length went
    to zero. That is what drove `synth_nerl` to ~0 for the ENTIRE run -- a seeding artifact of the
    rank cap, not a representation failure (voxel-level same-vs-other AUC is 0.934 on synth against
    0.819 on real, i.e. the model separates synth BETTER than real).

    Fix: union of the global top-`cap` (so nothing that worked before regresses) with the top
    `per_inst` seeds lying INSIDE each GT instance. The result is re-sorted into descending-EDT
    order, so the decode still grows the most confident objects first and remains comparable across
    steps.
    """
    if len(coords) == 0:
        return coords
    keep = np.zeros(len(coords), bool)
    keep[:cap] = True
    lab = inst[coords[:, 0], coords[:, 1], coords[:, 2]]
    for i in np.unique(lab):
        if i == 0:
            continue
        keep[np.flatnonzero(lab == i)[:per_inst]] = True    # coords are already EDT-descending
    return coords[keep]


class InlineEvaluator:
    def __init__(self, cfg, work_dir: str, device, rank: int = 0, world: int = 1):
        self.cfg = cfg
        self.device = device
        self.rank = rank
        self.world = world
        # eval-time decode config (coarser + capped, agglo stays off)
        self.ecfg = dataclasses.replace(
            cfg, fill_max_steps=cfg.eval_fill_max_steps, agglo_enabled=False)

        splits = json.load(open(os.path.join(work_dir, "splits.json")))
        val = splits["val"]
        real_ids = sorted(i for i in val if i.startswith("real_"))[:cfg.eval_cubes_real]
        synth_all = sorted(i for i in val if not i.startswith("real_"))
        # interleave OLD-corpus and NEW-GEO synth val cubes: plain sorted() puts every old id
        # first (synthfuse_0xxxx < synthfuse_4xxxx), so the panel would never contain the new
        # geometry distribution the resume is being trained on.

        def _sid(c):
            try:
                return int(c.rsplit("_", 1)[-1])
            except ValueError:
                return -1
        new_g = [i for i in synth_all if _sid(i) >= 39000]
        old_g = [i for i in synth_all if _sid(i) < 39000]
        inter = [x for pr in zip(new_g, old_g) for x in pr]
        inter += new_g[len(old_g):] + old_g[len(new_g):]
        synth_ids = (inter or synth_all)[:cfg.eval_cubes_synth]
        self.panel = [(cid, True) for cid in real_ids] + [(cid, False) for cid in synth_ids]

        import tifffile
        c = cfg.eval_crop
        self.cubes = []          # dicts: image(gpu), inst(np), seeds(np), pairs, sizes, blind
        for cid, is_real in self.panel:
            img = tifffile.imread(os.path.join(cfg.corpus_dir, "imagesTr", f"{cid}_0000.tif"))
            inst = tifffile.imread(os.path.join(cfg.corpus_dir, "labelsTr_inst", f"{cid}.tif")).astype(np.int32)
            o = [(s - c) // 2 for s in inst.shape]
            img = img[o[0]:o[0] + c, o[1]:o[1] + c, o[2]:o[2] + c]
            inst = inst[o[0]:o[0] + c, o[1]:o[1] + c, o[2]:o[2] + c]
            if getattr(cfg, "eval_instance_seeds", False):
                # per-instance medial seed field: the union-mask ridge lies ON the contact plane
                # wherever wraps touch, so the historical harness seeded the panel's decodes at
                # exactly the straddle points training excludes (see seeds.instance_inference_seeds)
                coords, _ = S.instance_inference_seeds(inst, min_edt=cfg.inf_seed_min_edt,
                                                       spacing=cfg.eval_seed_spacing)
            else:
                fiber = (inst > 0).astype(np.float32)
                coords, _ = S.inference_seeds(fiber, thr=0.5, min_edt=cfg.inf_seed_min_edt,
                                              spacing=cfg.eval_seed_spacing)
            coords = fair_seeds(coords, inst, cfg.eval_seed_cap,
                                getattr(cfg, "eval_seed_per_inst", 0))
            pairs = sorted(adjacency_pairs(inst, cfg.adjacency_radius))
            sizes = {int(i): int((inst == i).sum()) for i in np.unique(inst) if i != 0}
            blind = set()
            if is_real:
                for (i, j) in pairs:            # classify once; GT+image are fixed
                    if _pair_is_blind(inst, img, i, j, cfg.adjacency_radius):
                        blind.add((i, j))
            image_gpu = torch.from_numpy(
                ((img.astype(np.float32) / 255.0 - 0.5) / 0.5)).to(device)
            self.cubes.append(dict(cid=cid, is_real=is_real, image=image_gpu, inst=inst,
                                   seeds=coords, pairs=pairs, sizes=sizes, blind=blind))
        # ---- unlabelled core cubes: movement telemetry ONLY -------------------------------
        # The labelled panel above is the well-separated regime; measured on run-2, the model traces
        # fine there (frac_1step 0.067) and stalls on compact core material (0.556 here, 0.76-0.85
        # on an uncapped standalone decode of the same cubes). The panel
        # therefore cannot see the failure that matters, so probe the cores directly. They have no
        # instance labels, but frac_1step is a property of the decode trajectory, not of the GT.
        self.cores = []
        for d in getattr(cfg, "eval_core_dirs", []):
            fs = sorted(glob.glob(os.path.join(d, "imagesTr", "*_0000.tif")))
            if not fs:
                continue
            n = cfg.eval_core_cubes
            for f in fs[::max(1, len(fs) // (n + 1))][:n]:
                ct = tifffile.imread(f)
                o = [(s - c) // 2 for s in ct.shape]
                ct = ct[o[0]:o[0] + c, o[1]:o[1] + c, o[2]:o[2] + c]
                fiber = (ct >= air_papyrus_threshold(ct)).astype(np.float32)
                coords, _ = S.inference_seeds(fiber, thr=0.5, min_edt=cfg.inf_seed_min_edt,
                                              spacing=cfg.eval_seed_spacing)
                self.cores.append(dict(
                    cid=os.path.basename(f)[:-9],
                    image=torch.from_numpy(((ct.astype(np.float32) / 255.0 - 0.5) / 0.5)).to(device),
                    seeds=coords[:cfg.eval_core_seed_cap],
                    fibre=int(fiber.sum()),
                    lam=measured_lambda(ct)))      # measured once; the cube's own wrap period
                #      ^ validated estimator (see measured_lambda) -- NOT ctstats.wrap_period,
                #        which mis-scaled this gate 1.8-6.6x on the core cubes

        if rank == 0:
            n_pairs = sum(len(cb["pairs"]) for cb in self.cubes)
            n_blind = sum(len(cb["blind"]) for cb in self.cubes)
            print(f"[val] inline panel: {len(real_ids)} real + {len(synth_ids)} synth "
                  f"cubes, {n_pairs} adjacency pairs ({n_blind} blind), "
                  f"seeds<={cfg.eval_seed_cap}/cube", flush=True)
            if self.cores:
                print(f"[val] core probe (unlabelled, movement only): {len(self.cores)} cubes, "
                      f"seeds<={cfg.eval_core_seed_cap}/cube", flush=True)

    def active_cubes(self, step: int | None = None):
        """The panel to decode at `step`.

        The full synth panel is BUILT at init but only DECODED once `eval_synth_full_step` is
        reached; before that only `eval_cubes_synth_early` synth cubes run. A synth cube costs ~3x a
        real one (23-28 instances per cube against 8-9 -- measured 22.0 s vs 7.4 s single-rank), and
        synth is not what the run is selected on, so the early phase spends the budget where the
        decisions are made and the later phase restores the fuller tripwire once the loss changes
        (w_other, weight_decay) have settled. Building all of them up front keeps the panel FIXED, so
        numbers stay comparable across the switch -- only the subset decoded changes.
        """
        n_syn = self.cfg.eval_cubes_synth
        full_at = getattr(self.cfg, "eval_synth_full_step", 0)
        if full_at and step is not None and step < full_at:
            n_syn = min(n_syn, getattr(self.cfg, "eval_cubes_synth_early", n_syn))
        out, seen = [], 0
        for cb in self.cubes:
            if cb["is_real"]:
                out.append(cb)
            else:
                if seen < n_syn:
                    out.append(cb)
                seen += 1
        return out

    def _core_validity(self, pred, cb):
        """Per-object physical validity on an UNLABELLED core cube.

        Returns (sum of size^2 over VALID objects, n_objects, n_valid, claimed voxels).

        valid <=> p90(2*EDT inside the object) <= core_thick_frac * lambda   [did not fuse across a wrap]
              AND the object is a single connected component               [did not bridge an air gap]

        The two tests are complementary and both are needed: a fused merge is thick but connected, a
        gap-spanning merge is thin but disconnected. Only the largest `eval_core_topk` objects are
        tested -- the score is size^2 weighted, so the tail cannot move it, and this bounds the cost.
        Objects beyond the cap are counted as INVALID (conservative: never credit an untested object).
        """
        from scipy import ndimage as ndi
        lam = cb.get("lam", 0.0)
        ids, cnts = np.unique(pred[pred > 0], return_counts=True)
        claimed = int(cnts.sum())
        if not len(ids):
            return 0.0, 0, 0, 0
        if lam <= 0:
            # No measurable periodicity -> this cube cannot be judged. Return it as EXCLUDED (the
            # caller must not add its fibre^2 to the denominator) rather than scoring every object
            # invalid, which silently dragged the whole panel score to 0.
            return None, len(ids), 0, claimed
        thick_max = self.cfg.core_thick_frac * lam
        order = np.argsort(-cnts)[:self.cfg.eval_core_topk]
        objs = ndi.find_objects(pred)
        vsq, nval = 0.0, 0
        for k in order:
            iid = int(ids[k])
            sl = objs[iid - 1]
            if sl is None:
                continue
            m = pred[sl] == iid
            # 26-connectivity, and "one DOMINANT component" rather than "exactly one". The old rule
            # (exactly one 6-connected component) rejected every object -- thin oblique ribbons are
            # routinely 6-disconnected -- so core_score read 0.0 for ANY model and ranked nothing.
            # Requiring strict connectivity is also stricter than the objective: the pipeline plans
            # to agglomerate fragments downstream. What we actually need to catch is an object whose
            # MASS is split across an air gap, which the 0.9 dominance test does.
            lab_m, nc = ndi.label(m, structure=np.ones((3, 3, 3), bool))
            if nc > 1:
                cc = np.bincount(lab_m.ravel())[1:]
                if cc.max() < 0.9 * cc.sum():
                    continue             # mass bridged an air gap -> two wraps under one id
            edt = ndi.distance_transform_edt(m)
            if float(np.percentile(edt[m], 90)) * 2.0 > thick_max:
                continue                 # spans more than a wrap period -> fused merge
            vsq += float(cnts[k]) ** 2
            nval += 1
        return vsq, len(ids), nval, claimed

    def _merged_pairs(self, cb, pred):
        """Adjacent GT pairs claimed >=merge_theta on BOTH sides by one pred label."""
        theta = self.cfg.merge_theta
        cover = {}                                   # gt id -> {pred labels covering >= theta}
        for gid, sz in cb["sizes"].items():
            pv = pred[cb["inst"] == gid]
            vals, cnts = np.unique(pv[pv > 0], return_counts=True)
            cover[gid] = {int(v) for v, c in zip(vals, cnts) if c / sz >= theta}
        merged = {(i, j) for (i, j) in cb["pairs"] if cover[i] & cover[j]}
        return merged

    @torch.no_grad()
    def run(self, raw_model, amp_dtype, step: int | None = None) -> dict:
        """Decode this rank's share of the panel, all-reduce counts, return
        rates + NERL (real/synth/all). NERL is aggregated EXACTLY across cubes as
        (Σ run-length²)/(Σ perfect²), not a mean of per-cube NERLs."""
        raw_model.eval()

        def predict_fn(inp):
            if self.cfg.channels_last:
                inp = inp.to(memory_format=torch.channels_last_3d)
            with torch.autocast("cuda", dtype=amp_dtype):
                out = raw_model(inp)
            # cfg.move_head makes the model return (logits, move_logits); the decode consumes the
            # tuple itself (inference.py picks the learned gate off it), so pass it straight through.
            if isinstance(out, tuple):
                return out[0].float(), out[1]
            return out.float()

        # ints: [real_merged, real_pairs, blind_merged, blind_pairs, synth_merged, synth_pairs, n_inst]
        counts = torch.zeros(15, dtype=torch.long, device=self.device)
        # floats: [real_runsq, real_perfsq, synth_runsq, synth_perfsq]  (for NERL)
        rl = torch.zeros(7, dtype=torch.float64, device=self.device)
        for k, cb in enumerate(self.active_cubes(step)):
            if k % self.world != self.rank:
                continue
            steps = []
            pred = I.segment_block(predict_fn, cb["image"], cb["seeds"], self.ecfg, K=128,
                                   step_log=steps)
            merged = self._merged_pairs(cb, pred)
            e = erl_metric(cb["inst"], pred)   # NERL run-length sums (volumetric, robust)
            if cb["is_real"]:
                counts[0] += len(merged); counts[1] += len(cb["pairs"])
                counts[2] += len(merged & cb["blind"]); counts[3] += len(cb["blind"])
                rl[0] += e["sum_runsq"]; rl[1] += e["sum_perfsq"]
            else:
                counts[4] += len(merged); counts[5] += len(cb["pairs"])
                rl[2] += e["sum_runsq"]; rl[3] += e["sum_perfsq"]
            counts[6] += len(np.unique(pred[pred > 0]))
            # MECHANISM telemetry: every other metric here is computed on the final label volume, so
            # a model that never moves its FOV scores the same as one that traces. frac_1step sees
            # the difference directly. NOTE: on THIS panel run-2 reads a healthy 0.067 -- the panel
            # is the easy regime. Watch core_frac_1step below for the failure that matters.
            counts[7] += sum(1 for s in steps if s <= 1)
            counts[8] += len(steps)
            rl[4] += float(sum(steps))

        # core probe: same telemetry, on the regime that actually fails
        for k, cb in enumerate(self.cores):
            if k % self.world != self.rank:
                continue
            steps = []
            pred = I.segment_block(predict_fn, cb["image"], cb["seeds"], self.ecfg, K=128,
                                   step_log=steps)
            counts[9] += sum(1 for s in steps if s <= 1)
            counts[10] += len(steps)
            valid_sq, n_lab, n_val, claimed = self._core_validity(pred, cb)
            counts[11] += claimed
            counts[12] += cb["fibre"]
            counts[13] += n_lab
            counts[14] += n_val
            if valid_sq is not None:        # None = wrap period unmeasurable -> exclude the cube
                rl[5] += valid_sq
                rl[6] += float(cb["fibre"]) ** 2
        if self.world > 1:
            import torch.distributed as dist
            dist.all_reduce(counts)
            dist.all_reduce(rl)
        raw_model.train()
        c = counts.tolist()
        rr, rp, sr, sp, tot_steps, core_sq, core_fib_sq = rl.tolist()

        def rate(m, p):
            return round(m / p, 4) if p else None
        def nerl(runsq, perfsq):
            return round(runsq / perfsq, 4) if perfsq > 0 else None
        return dict(
            # NERL (PRIMARY objective; coverage floor -> collapse scores ~0)
            real_nerl=nerl(rr, rp), synth_nerl=nerl(sr, sp), all_nerl=nerl(rr + sr, rp + sp),
            # merge rate (CONSTRAINT only; coverage-blind)
            real_merge=rate(c[0], c[1]), real_pairs=c[1],
            blind_merge=rate(c[2], c[3]), blind_pairs=c[3],
            synth_merge=rate(c[4], c[5]), synth_pairs=c[5],
            all_merge=rate(c[0] + c[4], c[1] + c[5]),
            instances=c[6],
            # Fraction of committed objects that never moved their FOV (1.0 = not tracing at all).
            # Every other metric here is computed on the final label volume, so a model that never
            # moves scores the same as one that traces; this sees the mechanism directly.
            frac_1step=round(c[7] / c[8], 4) if c[8] else None,
            mean_steps=round(tot_steps / c[8], 2) if c[8] else None,
            # THE one to watch: same statistic on unlabelled held-out core cubes. Measured on the
            # run-2 checkpoint: 0.067 on the labelled panel above vs 0.556 here -- only this column
            # sees the collapse. (An uncapped standalone decode of whole core cubes reads higher
            # still, 0.76-0.85; this probe caps seeds at eval_core_seed_cap and they are
            # EDT-descending, so it keeps the strongest seeds and reads low. Track it as a trend,
            # do not compare it to the standalone number.)
            core_frac_1step=round(c[9] / c[10], 4) if c[10] else None,
            # PRIMARY core metric: the NERL form (size^2, so fragmentation self-penalises) counting
            # ONLY physically valid objects (did not cross a wrap period). Label-free, and unlike the
            # confidence/entropy proxies used for unsupervised model selection it cannot be gamed by
            # a confident collapse -- a merged object is excluded no matter how sure the net is.
            core_score=round(core_sq / core_fib_sq, 5) if core_fib_sq > 0 else None,
            core_cov=round(c[11] / c[12], 4) if c[12] else None,
            core_inst=c[13],
            core_valid_frac=round(c[14] / c[13], 3) if c[13] else None)


def log_val(work_dir: str, step: int, stats: dict, took_s: float) -> None:
    rec = dict(step=step, took_s=round(took_s, 1), **stats,
               time_unix=int(time.time()))
    with open(os.path.join(work_dir, "val_progress.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
