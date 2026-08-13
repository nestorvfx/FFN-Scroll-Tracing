# FFN — findings, defects, and the plan

*Audit written 2026-07-26. Measured, corrected, and partially implemented 2026-07-27.*

Companion to `DESIGN.md` (the original design) and `README.md` (the operational guide). **This document
supersedes them wherever they disagree.** It records a full re-read of both FFN implementations, a
literature check, and the code-verified defects — plus the prioritised plan that follows from them.

Two independent analyses (main agent + an Opus subagent) were run against the same question and agreed on
the framing; the defects in §3 were found by the subagent and then verified in the code.

**Read §3.6 first.** The 2026-07-26 audit was written before anything had been executed. Running it changed
the picture: the headline defect is not any single gate or threshold but that **the recurrence is not
running at all** on the material we care about, and the validation panel that training steers on could not
see that. Original text is left standing throughout with `> CORRECTION` / `> MEASURED` blockquotes rather
than rewritten, so the reasoning behind a wrong call stays auditable. Claims are marked **verified in
code**, **measured**, or **inferred** — and where a measurement refuted an inference, both are shown.

**Start here, in order:** §3.6 (what is actually broken) → §2 (geometry, measured) → §4 (the architecture) →
§6 (status: what is live, what is off by default, what is deliberately not built).

---

## 0. The question this answers

> An FFN has a fixed input field of view. Inside one 192³ cube a single physical sheet can be laminated,
> fragmented and scattered into several apparently-separate pieces, and the evidence that they are the same
> sheet lies outside the cube. How is the network supposed to assign these correctly? Do we need a bigger
> input and a retrain? Or is there a pre/post-processing route to run at much larger scale?

**Short answer: the FOV is not the binding constraint, a bigger input is the wrong lever, and the route is
over-segment → agglomerate. The FFN does not need to work out that scattered pieces are one sheet — but
something does, and right now nothing does.**

---

## 1. What an FFN can and cannot do here

### 1.1 The receptive field is a path, not a box (verified in code)

`flood_fill_object` (`ffn/inference.py:39-147`) allocates a POM canvas over the **whole block**, writes each
FOV's logits into it, and re-reads it on the next step. Consecutive FOVs overlap by `33 − 8 = 25` voxels, so
every step conditions on the evidence committed by previous steps. The effective receptive field of a
decision is therefore **the entire traversed path**, not 33³.

This is the mechanism single-pass models lack, and it is real: FFN traces 1.1 mm mean error-free neurite
path with a 33³ FOV, and FAFB-FFN1 segmented 40 teravoxels this way.

### 1.2 What it structurally cannot do

1. **It cannot re-join a disconnected fragment.** The queue only advances to a candidate whose POM already
   exceeds threshold. Where a sheet is genuinely broken the fill terminates, and there is no mechanism to
   hypothesise that a fragment 40 voxels away is the same sheet. The FFN paper needed a *separate* stage for
   this (agglomeration by resegmentation); FAFB needed a further one (gap synthesis + overlap joining).
2. **It is Markovian in the POM field.** The only state carried between steps is a scalar per-voxel
   probability. No identity, no winding index, no "I came from radius r at azimuth θ".
3. **It is greedy and irreversible.** One bad step across a blind contact merges the wrap permanently.
   Oversegmentation-consensus is variance reduction around this, not a fix.
4. **Loop closure is out of scope.** "These two ribbons λ apart are consecutive turns of one spiral" is a
   statement about an integer. No local recurrent grower computes integers.

### 1.3 Bigger FOV is the wrong lever

- It does **not** buy fragment re-association (that needs ~wrap circumference — every intermediate size is
  on the wrong side of a step function) and does **not** buy loop closure.
- Cost is cubic: 33³ → 49³ is **3.3×**, → 65³ is **7.6×**. Combined with the move from 2× RTX 5090 to
  1× RTX 3090 (~3–4× slower), the 300k-step schedule goes from ~2 days to ~3 weeks.
- Sample efficiency gets *worse*: the fill-fraction distribution shifts and the 17-bin balanced sampler —
  already hacked with `min_bin_edge=0.02` to stop near-empty bins teaching predict-low — needs re-tuning.
  Run-1 collapsed on exactly this pathology.

**The one legitimate FOV change** is to make it per-scroll in units of λ (§2), as an ablation, not a rebuild.

**A cheaper way to buy context:** a third input channel carrying a coarse global field (§4, Stage 0). This is
classic auto-context; `ffn_vesuvius` already has the plumbing (`in_channels=3`). +0.03% params, zero
inference cost, and the 33³ FOV then carries information from thousands of voxels away.

### 1.4 The canonical large-volume answer, from the literature

- **google/ffn manual:** inference is **per-subvolume**, "embarrassingly parallel with no dependencies
  between subvolumes"; a global segmentation is then assembled with **union-find**. FFN does *not* stream.
- **Songbird paper (2018):** discard a 32-voxel envelope, extract a 1-voxel plane from the middle of each
  overlap, and merge segments that are **mutually maximally overlapping**. Measured: **−84% mergers,
  +28% splits.** That trade is exactly our economics.
- **Petavoxel human cortex (Shapson-Coe & Januszewski, *Science* 2024):** multi-resolution FFN →
  **"base segments" (explicitly fragments of whole objects)** → agglomeration by FFN resegmentation →
  a separate subcompartment classifier to fix the merges agglomeration introduces.
- FAFB blocked FFN evaluation where predicted neuropil < 12% using a tissue mask. **Our analogue is the
  existing fiber mask — restricting seeds *and moves* to it is free and cuts cost several-fold.**

The field's answer to "object ≫ FOV" is **over-segment then agglomerate**, never a bigger box.

---

## 2. Geometry — MEASURED (this section replaces an earlier, wrong reconciliation)

Measured along the local sheet normal (structure-tensor eigenvector), phantom-validated to <1% on known
lattices, cross-checked against 34,341 ground-truth radial measurements through the slab's 68 traced wraps.

| | slab | real corpus | **synth corpus** | Scroll-3 core | Scroll-4 core |
|---|---|---|---|---|---|
| **λ_nn** nearest neighbouring lamina (vox) | 16.0 | 12.5 | 11.0 | 10.0 | **9.5** |
| CT bright-band FWHM along normal (vox) | 8.6 | 8.0 | **13.6** | 7.1 | 6.6 |
| labelled-mask thickness (vox) | 7.5 | 7.0 | 6.0 | — | — |
| instances >=500 vox per 192^3 | 8 | 9 | **26.5** | ~13-20 | ~13-20 |
| label foreground fraction | 0.32 | 0.31 | **0.58** | — | — |
| **33^3 FOV in units of λ_nn** | 2.1 | 2.6 | 3.0 | 3.3 | **3.5** |
| **delta 8 as a fraction of λ_nn** | 0.50 | 0.64 | 0.73 | 0.80 | **0.84** |

### Three earlier claims REFUTED by measurement

1. **"Labels are medial ribbons ~1/4 of the physical sheet" — WRONG.** Direct volume ratio
   `|labels| / |CT >= air/papyrus threshold|` = **1.07** (real crops) and **1.01** (synth). The labels ARE
   the imaged sheet. The error was reading the Scroll-4 paper's 153.6 um as sheet thickness; it is the
   **wrap period** (= 19.4 vox, which matches the measured Scroll-4 spectral period to ~1%).
2. **"λ ≈ 13-15 on the slab" — WRONG.** Ground truth is **20.0** median (IQR 13.5-28.5); 13-15 was the low
   quartile of a broad distribution.
3. **"33^3 spans only 1.4λ on Scroll 3, breaking the design rationale" — WRONG, and backwards.** That used a
   whole-scroll radial λ=23. Inside the core cubes we actually evaluate on, λ_nn is 10.0 (S3) / 9.5 (S4), so
   the FOV spans **3.3-3.5 λ — MORE neighbouring-lamina context than the slab (2.1λ)**, not less.
   **The FOV is not starved anywhere. Do not enlarge it, and do not rescale.**

### What IS a geometric mismatch: `delta`

delta = 8 against λ_nn = 9.5 on Scroll 4 means one axis step crosses **84%** of the distance to the
neighbouring lamina (vs 50% on the slab), and **53% of Scroll-4 probe locations have a neighbour within
10 voxels** — inside a single step. This argues for **delta 5-6**, not a larger FOV.

### The dominant data defect: the synth corpus

Synth has **CT band FWHM 13.6 against λ_nn 11.0** — the bright bands physically overlap, so there is no
resolvable gap anywhere. Label fg 0.58 vs 0.31-0.32 real; 26.5 sheets/cube vs 8-9. Independently, a mid-slice
Otsu foreground measurement gives synth 0.569 vs real Scroll-1 crops 0.352, Scroll-3 core 0.419, Scroll-4
core **0.226** — **Scroll-4's p90 (0.350) lies below synth's p10 (0.481), i.e. zero overlap.** The corpus
acceptance gate in `gen_corpus2.py` hard-codes `flo,fhi = (0.58,0.72)` std / `(0.70,0.82)` deep, which
excludes 100% of Scroll-4 core and 97% of Scroll-3 core **by construction**. (The mean/lapvar band is
reference-relative and is NOT the excluding mechanism; the foreground band is.)

## 3. Code-verified defects

*Written as five; §3.6 was added on 2026-07-27 and supersedes the others in priority. Blockquoted
corrections inside each subsection record what later measurement confirmed or refuted — the original text
is left standing so the reasoning that produced a wrong call stays visible.*

### 3.1 The movement gate uses a single voxel, not the face plane — top suspect for run-1's collapse

`ffn/inference.py:127-136` and `ffn/movement.py:56-79` both gate a ±8 axis step on
`sigmoid(POM[centre + 8·eᵢ]) ≥ 0.9` — **one voxel**. The paper and `ffn_vesuvius/inference/movement.py:124`
take **`face.max()` over the entire ±delta face plane**.

**Why this is specifically catastrophic for a scroll (geometric argument, not yet measured):** a wrap ribbon
has a local normal **n** that is radial and roughly in-plane (`n_z ≈ 0`). An axis-aligned step of 8 voxels
leaves the ribbon by `8·|n·eᵢ|`, so you stay on a 4-voxel ribbon only if `|n·eᵢ| ≤ 0.25`. At azimuth 45°,
`n ≈ (0, 0.71, 0.71)` — **both** in-plane steps land 5.7 voxels off the ribbon, where POM ≈ 0.05. Only ±z
steps survive, so the fill propagates along z and dies. Over most azimuths of a cylindrical wrap this gate
silently forbids in-plane growth.

The same defect is in **training**: `training_move` gates identically and, under teacher forcing,
additionally requires `inst[seed + cand] == seed_instance`, false for the same reason. When nothing is valid
the FOV **stays put** (`movement.py:76-78`) — so through Phase A and most of Phase B the model is trained as
T repeated forwards at a stationary FOV and never learns to extend a mask. That is the canonical FFN failure
mode, and it compounds with the `logit(0.05)` pad, which biases the net negative at exactly the FOV faces.

**Also:** the current `google/ffn/doc/manual.md` recommends `pad_value: 0.5`, `move_threshold: 0.6`, matched
between train and inference. `ffn_vesuvius` uses `LOGIT_PAD = 0.0` (pad 0.5) and documents why.

> **CORRECTION (2026-07-27).** The sentence above originally read "we use pad `logit(0.05) = −2.94`". That
> was wrong: **there is no `pad_value` in this codebase.** `batched_crop` reads out-of-bounds coordinates by
> `.clamp_`-ing them to the volume, i.e. **edge replication**, so a crop that overhangs the volume repeats the
> boundary rather than padding with a constant. The `0.05` that misled me is `cfg.pom_init`, and it serves
> **three distinct roles** that the single name hides: (i) the initial POM value of every unvisited voxel,
> (ii) therefore the effective value seen at FOV faces that have not been written yet, and (iii) the
> reference the ratchet's "already background" test is written against. Only (i) and (ii) are what the
> paper's `pad_value` is about. Raising it to 0.5 is still right — an unvisited voxel should read *unknown*,
> not *confidently background*, and it is confidently-background precisely at the faces where growth must
> start — but it is a change to `pom_init`, not to a pad, and it touches all three roles at once.

> **CORRECTION (2026-07-27) — the asymmetry is deliberate.** This section reads the paper's centre-gate
> train / face-max infer combination as an inconsistency in *our* code. It is not: the paper trains with a
> centred FOV and moves with face-max at inference on purpose. So "match training to inference" is not by
> itself an argument for changing the gate.

> **~~MEASURED, and it refutes the standalone fix.~~ RETRACTED 2026-07-28.** The 0.630 figure was an
> artifact of the face-SIZE bug documented immediately below (a 33x33 plane reaching +-16 voxels, i.e.
> past the neighbouring lamina). Re-measured on the same run-2 checkpoint with the corrected (2*delta+1)^2
> cuboid face: merge **0.0055 -> 0.120**, while NERL *improves* **0.120 -> 0.179** and stationary objects
> fall to **0.000**. Face-max is better than the centre gate, not catastrophically worse. This number
> governed the tracer-profile design for two runs; see research/02_pipeline_audit.md D3.
>
> The original (now void) claim read: swapping the inference gate to face-max takes merge 0.065 -> 0.630
> and blind-contact merge to 0.689. The geometric argument above
> is sound, but the threshold (0.9) and the position distribution the network was trained under are both
> calibrated for the centre-voxel rule, so the gate cannot be changed alone. It is now part of
> `FFNConfig.as_tracer()`, which changes gate + threshold + `pom_init` + `delta` together and requires a
> **from-scratch** retrain.

> **WHAT DID LAND (2026-07-27): the face *size* was wrong, independently of the gate rule.** `face_offsets`
> built its face with `h = fov // 2`, i.e. a **33×33 plane** at `fov=33`, reaching ±16 voxels. On the
> Scroll-4 core that is ≈1.7 × λ_nn, so the max would routinely be read **on the neighbouring lamina** — a
> face-max over that plane is not "the frontier of this sheet", it is "is there any papyrus nearby", which is
> true everywhere in a packed scroll. Corrected to `h = int(delta)`, the (2δ+1)² cuboid face the paper
> actually specifies. This is a bug fix, not a tuning choice, and it is live by default.

### 3.2 The unit tests structurally cannot see it

`tests/phantoms.py:9-30` builds **axis-aligned planar slabs** — the one geometry where centre-voxel gating is
equivalent to face-max — with **per-sheet distinct intensities**, so the oracle separates them trivially. The
suite cannot exhibit either this bug or the blind-contact problem.

### 3.3 "Agglomeration is unsafe at blind contacts" rests on a straw-man implementation

`config.py:136-141` disables agglomeration because the regrow-at-interface test false-merges blind contacts.
But `candidate_pairs` (`inference.py:313-333`) picks a decision point **on the A/B seam** and runs **one**
fill from it. A seed on a blind interface is *guaranteed* to grow into both wraps.

The paper's criterion is materially stronger: seeds at the **EDT maxima inside each fragment**, both segments
**removed** from the subvolume, **two independent regrowths** that must agree, and a four-part acceptance test
— `iou > 0.8` **and** both consistencies `> 0.6` **and** `deleted_voxels/num_voxels < 0.02` (a direct measure
of model confusion) — with retry-under-exclusion up to 8×.

**Conclusion: the finding that FFN agglomeration is unsafe here is not supported by the implementation that
produced it.** Reimplement the real criterion before accepting that verdict.

### 3.4 Block-wise inference is dead code

`block_grid` / `stitch_blocks` (`inference.py:377-416`) are **called by nothing** — `evaluate.py` and
`inline_eval.py` both decode whole volumes. `test_block_tiling_separates_same_as_whole`
(`tests/test_inference.py:76-86`) does **not** tile. DESIGN checklist item 19 is unimplemented.

Additionally `stitch_blocks` uses a **one-sided greedy** rule (≥50% overlap → take majority global label)
rather than the paper's **mutual-maximal-overlap on mid-halo planes**, which is what bought −84% mergers.

### 3.5 The 2× consensus term upsamples with `np.repeat`

`evaluate.py:89-91,146`. The paper explicitly avoids naive upsampling: it filters low-res objects to
≥100,000 voxels, then **seeded-watersheds the high-res EDT** using the upsampled labels as seeds, "so
voxel-level differences between the two segmentations [don't] generate new segments". On 4-voxel ribbons,
`np.repeat` of a 2× segmentation is likely a large **spurious split** generator.

> **CONFIRMED AND FIXED (2026-07-27).** It was. Dropping the 2× term takes coverage **0.446 → 0.664** and
> NERL **0.040 → 0.104**; the term's apparent merge benefit was **shattering, not separation**, visible in
> renders as axis-aligned blocks cutting across inter-lamina gaps. There is also no mechanism for it to help
> here: multi-scale consensus works in the paper because 2× changes the FOV *in physical units relative to
> the object*, and our isotropic-rescale sweep found coverage and merge **strictly monotone across five
> scale factors with no optimum** — there is no second scale at which this material behaves differently.
> `consensus_scales` now defaults to `[1]`. Forward+reverse consensus at 1× is retained.

---

## 3.6 The finding that supersedes the rest: **the model is not flood-filling**

Measured on held-out Scroll-3/4 core cubes with the run-2 checkpoint (`steps_probe.json`):

| | frac_1step | median steps | median span |
|---|---|---|---|
| held-out core cubes | **0.76 – 0.85** | **1** | **0** |

Three quarters to five sixths of all committed objects are **a single forward pass at a stationary FOV**.
The median object never moves at all. Renders agree: ribbons are chopped into one-window chips. Whatever
else is true, the recurrence — the entire premise of an FFN — is not running. Every defect above is
downstream of this one.

**The labelled val panel cannot see it.** Same checkpoint, same decode, `frac_1step` on the panel:
**0.067**, with `mean_steps` 37.9. The panel is built from corpus cubes, which are the well-separated
regime the model already handles; the failure lives on compact, disturbed core material that the panel does
not contain. A ~12× gap between the two, and **the panel is the number training was steering on** — run-2
logged a healthy merge rate the whole way down.

*Landed:* `InlineEvaluator` now also probes **unlabelled held-out core cubes** and reports
`core_frac_1step` (merge rate needs GT adjacency, but movement does not — it is a property of the decode
trajectory). Enable with `train.py --core-dirs <dir>[,<dir>]`. Foreground for seeding comes from a
2-Gaussian air/papyrus EM fit (`ffn/ctstats.py`), never a percentile — percentile thresholds are
composition-dependent, which is what made Scroll-4 look "too compressed" (2 sheets → 49 after a physical
threshold).

### The POM ratchet — landed, on by default

The paper's split-biasing update: once the FFN has written a **background** verdict for a voxel, a later
pass may not raise it. Measured on the labelled panel: merge **0.071 → 0.028**, blind-contact merge
**0.044 → 0.000**.

Scope matters and the two readings are not equivalent here. `pom_ratchet_scope="step"` freezes against the
**live** canvas, including between refinement iterations at one FOV position; `"position"` freezes only
against the canvas as it stood on entry to the position. `"position"` measured **no effect at all**
(0.071 → 0.071), because `inf_fov_iters` re-forwards each position and that inner loop is exactly what the
strict rule constrains. The paper's own wording — inconsistent predictions *"between iterations"* — is the
`"step"` reading, so that is the default. Trade-off, stated plainly: `"step"` blocks a predictor that needs
repeated forwards at a stationary FOV to build confidence, which is what `inf_fov_iters` exists for, so
`test_iterative_fov_grows_what_single_pass_cannot` now pins `pom_ratchet=False`.

---

## 4. The recommended architecture

Small-FOV network, weights largely unchanged. Stages 0, 4 and 5 are new; 1–3 exist in some form.

**Stage 0 — global frame + coarse context (new, cheap).** Umbilicus curve, then the existing polar-block wrap
segmentation at 4× per z-slab (memory: `unsup-wrap-segmentation-sota`, BCubed F₀.₃ = 0.314, 5.3× baseline).
Emit `radial_phase` (fractional turns from the umbilicus) and `coarse_id` at full resolution. These are
**priors**, wrong at exactly the blind contacts — used as gates and agglomeration priors, never hard
constraints. Optionally feed as the **third input channel** (§1.3).

**Stage 1 — mask + seeds.** As now, but also **block FFN moves** outside the fiber mask (FAFB's trick).

**Stage 2 — block-wise FFN inference.** Blocks ~384×384×64, halo ≈ `fov/2 + max_move + margin ≈ 40`, npz per
block (interruptible, parallel). Discard a 32-voxel envelope; reconcile by **mutual-maximal-overlap on
mid-halo planes + global union-find**. Not a model decision — pure bookkeeping.

**Stage 3 — oversegmentation-consensus.** Keep forward + reverse at 1× (where the merger reduction comes
from). Either fix the 2× term (§3.5) or drop it until fixed. Can be computed per-block before global assembly.

**Stage 4 — split recovery, three tiers in order:**
- **4a. Geometric union-find (safe, deterministic, no model calls).** Fragment RAG over pairs within ~2λ.
  Merge only when normals align < 20°, **along-normal offset < 0.3λ**, the connecting segment crosses no
  third fragment, and `|Δradial_phase| < 0.25` turns. **This cannot merge adjacent wraps by construction —
  they are exactly λ apart along the normal.** It works precisely where intensity evidence is zero.
  *Caveat:* folds, where two plies are anti-parallel with small offset, need an explicit carve-out.
- **4b. FFN resegmentation, implemented correctly** (§3.3), on the residual pairs, gated through 4a first so
  a blind contact never reaches the model.
- **4c. (v2) Learned pair classifier** on surviving hard pairs — trainable from the sheet banks by cutting
  known sheets (see §5).

**Stage 5 — global winding consistency (the only thing that can fix a fully-blind contact).** Run winding
assignment on the **instance graph**, not on voxels: fragments are nodes, agglomeration edges are
constraints, the umbilicus gives an angular coordinate. A merged pair shows up as a **phase discontinuity of
one full turn** — an integer, checkable, non-local. Where violated, cut.

**Cost reality.** The FFN paper measured every voxel processed **59×** per run, ×2.38 for consensus, ×2.16 for
agglomeration ≈ 74× a plain CNN pass. With a fiber mask at 20–30% occupancy this is realistic for a slab or a
core region; a full 9778×3550×3400 scroll (118 Gvox) on one 3090 is **not** — plan masked, core-region,
block-wise runs.

---

## 5. What our datasets now enable that DESIGN.md did not assume

DESIGN.md was written when the synth corpus derived from two reference regions and real instance data was
believed unavailable. Since then:

| asset | what it is | why it matters now |
|---|---|---|
| **Scroll4Sheets** (344) + **Scroll3Sheets** (360) | 704 real single sheets, jump-free, per-sheet masks + standalone crops | Composer input with **two new scrolls' geometry**, including Scroll 3's λ=23 (tighter than anything in the current corpus) |
| **Scroll3/4 EvalCubes** (100 each) | connected core cubes, no labels, held out | A **second and third** real hard-regime gate; currently the only real gate is the z=10192 slab |
| Scroll-1 GT (40 crops) | true per-wrap instance labels | The only source of *real* merge/split ground truth; the zero-jump validator |
| 160 real harmonized crops (HF `05_ffn`) | real CT + harmonized instances | Appearance grounding, already wired into preprocess |
| synthfuse corpus (400 cubes) | per-sheet instances, dense fused | Current training source; **regenerable and now improvable** |
| composer (`synth/sheet_compose.py`) | validated fusion synthesis | **Unlimited** labelled blind contacts — and unlimited labelled **split/no-split pairs** for 4a/4c |

The composer's ability to manufacture *labelled fragment pairs by cutting known sheets* is the piece that
makes Stage 4a testable and Stage 4c trainable. That supervision did not exist when DESIGN.md was written.

---

## 5.5 Research dossier (2026-07-27/28)

Four independent investigations plus the measurements that verified or refuted them are preserved in
**[research/](research/)** — start with [research/README.md](research/README.md), which carries the
executive summary and the ranked action list. Headline results:

- **NERL = coverage^2 x fragmentation.** Coverage 0.75-0.83, fragmentation factor **0.18**. Perfect
  fragment reassembly is worth **~5x with zero model change** — larger than any data or training lever.
- **The commit threshold is hard-coded at 0.5 and 0.5 is wrong by construction**: class-balanced BCE
  with soft targets makes `sigmoid(POM) >= 0.5` commit voxels of true posterior **q ~ 0.25**. The gate
  demands q ~ 0.85 to *move* but q ~ 0.25 to *commit*.
- **Checkpoint selection has been inert**: `ckpt_best.pt` / `ckpt_core_best.pt` were never written,
  because `select_merge_eps = 0.05` is unreachable at that threshold.
- **The top proposal is a global winding-phase field** `u(x)` whose level sets *are* the sheets, making
  adjacent-wrap merges unrepresentable and fragment reassembly transitive and free. Model-free,
  validatable in days against official data. See [research/06](research/06_inference_architecture.md).
- **Pairwise geometric agglomeration is refuted** — anti-correlated with truth in a lamination
  (precision 0.40). See [research/07](research/07_verifications.md).
- **The whole-scroll design is [research/09](research/09_whole_scroll_pipeline.md)**, which supersedes
  06. It keeps the winding objective but replaces the continuous Poisson solve with an **integer
  synchronisation on a fragment graph** under the umbilicus monodromy constraint `∮∇u·dl = 1`.
  Measured for that report: the autocorrelation wrap-period estimator (`ctstats.wrap_period`) is
  **3.5× wrong** on radial profiles (77 vs GT 21), an oracle-λ demodulated phase still drifts
  **−0.81 turns over 24**, and **0 of 43 adjacent-wrap pairs on the slab are 100 % blind** (median
  90 %) — the evidence to separate every pair exists somewhere along it. Section 4's Stage 4a
  (geometric union-find) and Stage 5 (winding on the instance graph) are replaced by that single
  solve.

## 6. Status, stated plainly

- **Run-1 collapsed** (5 instances from 130). The config carries its post-mortem (prior-bias reset, NERL
  replacing coverage-blind merge-rate).
- ~~**There are no run-2 artifacts in the repo**~~ — **wrong**, corrected 2026-07-27. They are on HF at
  `05_ffn/run2/`. Everything in §3.6 and the blockquoted corrections above is measured on the run-2
  checkpoint, not inferred from run-1.
- **The model does not trace** (§3.6) — `frac_1step` 0.76–0.85 on held-out cores, median 1 step, median span
  0. This is the finding that matters; the rest are downstream.
- **The labelled val panel was steering training on the wrong regime** (§3.6) — 0.067 vs 0.556 on the same
  checkpoint. Now instrumented with `core_frac_1step`.
- **NERL's disconnected-GT worry is refuted.** I flagged that NERL might be structurally unreachable because
  GT instances can be disconnected. Measured oracle NERL on the panel: **0.989**. The metric is fine; the
  gap to 0.104 is the model.
- **Landed and on by default:** POM ratchet (`"step"` scope), the `face_offsets` size fix, `consensus_scales
  = [1]`, and the core movement probe.
- **Landed and off by default**, because each needs a from-scratch run to validate and the base config is
  deliberately kept reproducing run-2 so a new run is attributable: `train_offcentre`, `min_fov_steps`,
  `w_other`/`w_blind`, and the whole `FFNConfig.as_tracer()` profile (`train.py --tracer`).
- **Not implemented, deliberately:** the auxiliary geometry head, the error-detection network, and the
  winding-graph cut (§4 stages 4c/5).
- **The split-recovery hole is real and self-contradictory in the docs.** README says "splits are cheap
  (agglomeration recovers them)"; config says "agglomeration is OFF, the splits it would recover are
  downstream-free". Both cannot hold. README_STATE's "holes < 40–60 px interpolated ≈ free" refers to small
  gaps in a *ridge*, **not** to consensus shattering a wrap into many instances — and NERL (run-size²) goes
  to 0 under fragmentation. **The disabled agglomeration is paid directly in the selected objective.**
- Nothing currently recovers: consensus over-splits, fills terminated at a discontinuity, fragments below the
  500-voxel commit floor (dropped to background), or splits across block boundaries.
- ~~the box is **1× RTX 3090 (24 GB)**~~ — stale. The current box is **2× RTX 5090 (32 GB each)**, 80 cores,
  386 GB RAM, matching DESIGN's assumption. No rescaling needed.

---

## 7. References beyond DESIGN.md §11

- Shapson-Coe, Januszewski et al. *A petavoxel fragment of human cerebral cortex reconstructed at nanoscale
  resolution.* Science 384 (2024) — multi-resolution FFN → base segments → agglomeration → merge-error
  classifier. https://pmc.ncbi.nlm.nih.gov/articles/PMC11718559
- `google/ffn/doc/manual.md` — per-subvolume inference, union-find reconciliation, current recommended
  `pad_value 0.5 / move_threshold 0.6` matched train↔infer, resegmentation acceptance thresholds.
- Li, Lindsey, Januszewski et al. *Automated reconstruction of a serial-section EM Drosophila brain with
  flood-filling networks and local realignment* (FAFB-FFN1) — multi-scale pipeline, tissue-mask gating,
  400×400×100 segmentation subvolumes / 60×60×30 agglomeration.
- Tu & Bai. *Auto-context and its application to high-level vision tasks.* IEEE PAMI 32 (2010) — the
  coarse-context-as-input-channel argument.
- u-Segment3D (Nature Methods 2025) — consensus 3D instance assembly from overlapping subvolume tiles.
