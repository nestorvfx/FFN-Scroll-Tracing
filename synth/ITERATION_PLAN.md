# Synthetic-corpus iteration plan — until it is 100% spot-on

Binding goal (user-confirmed): **(1) Look** — indistinguishable from real compact scroll CT, up to
Scroll-3-core density, with clean labels; **(2) Content** — delamination as real *branching* (split/
run/rejoin under one label) interleaved with true contacts, in every combination with compact packs;
**(3) Diversity** — wide controlled ranges (density, curvature, gaps, delam richness).

## Protocol — every iteration, no exceptions

1. **All axes per iteration.** Each cycle ships changes against look AND content AND diversity —
   never one metric in isolation. (Paid lesson: census improved while realism regressed.)
2. **Same-seed A/B.** Regenerate the identical seed set; the only delta is the change under test.
3. **Render gate first.** Filled semi-transparent labels, slices centred on split/branch regions,
   side-by-side: NEW | OLD synth | REAL S1 | REAL S3/S4 core. A change that worsens the render is
   rejected regardless of metrics.
4. **Metric gate second.** Census vs real deciles (split-col median, inst>10% split, gap-width
   histogram vs DONOR distribution, fg, λ, seam-dip) + w1/w2 ratio + aperture autocorrelation
   (lens vs slit). Leakage probe (matched-class discriminator vs real-vs-real floor) before any
   corpus regeneration decision.
5. **Full finishing pipeline always** — through `gen_corpus2` (artifacts, PASTA, strata, gates,
   harmonization). Raw `compose()` outputs are never judged.
6. **Read full files before editing; research web (no community sources) when a mechanism is
   unclear; think through the physics before the knob.**

## Base configuration (the trajectory correction)

- **Donors: Scroll-1 windows** (the 160 real corpus crops shimmed to donor dirs) — clean harmonized
  masks, real delamination (47.7% of instances). Scroll-3 ribbons are a FUTURE additional stratum,
  admitted only after passing the same render gate (needs de-striping + sub-voxel path proven).
- **Keep** (already validated): ply-bundling ≤12 + instance-purity gate; skin-only fusion core-swap;
  in-plane-only morphology (no depth closing — it welds lens tapers shut); sub-voxel deposit
  (J1: linear content + signed-distance occupancy; the integer snap was verified as the comb's
  cause); smoothed displacement injection (J2); NaN guards; per-cube spacer/coverage draws;
  crease/kink fields; per-sheet conformity draws; delam-richness donor weighting (SC_DELAM_K).
- **Per-scroll squeeze policy**: S1 donors keep the calibrated squeeze (they are unwound-puffy);
  already-compact donors (S3 ribbons, later) ride near 1.0.

## Open work queue (attack together, in iteration-sized bundles)

A. **S1-donor shim + old-pipeline run** with the delam-preserving composer; first full render gate.
B. **Advisor P1 set**: per-column monotone depth warp in the squeeze (D1 — preserves blister lenses);
   curvature-driven ply opening (D3 — aperture peaks at bend apexes, ply asymmetry, physically);
   pap_acc label-claim fix (pad air must never block a material claim — Generator A);
   winner-take-all labels from fractional occupancy once J1 is settled.
C. **Look finishing**: low-frequency intensity field + cupping; Fisher-separability penalty in the
   accept gate (stop selecting for bimodality); Otsu guard for unimodal (core-like) references;
   re-raise papyrus interior detail (fpap) once geometry is sub-voxel; asymmetric fleck damping fix
   (dark lacunae currently suppressed 3.5×).
D. **Compact-core stratum**: reference-window mixture spanning S1 / S4-core / S3-core; per-cube λ
   target with a λ gate (inst-per-band 1.0–1.6); partial-span admit rate up for the deep stratum.
E. **Verification battery** before any regeneration call: 30-cube census, compact-only sub-census
   (delam INSIDE fg>0.5 cubes), leakage probe, and the visual panel — then the decision is the
   user's.

## References

- Advisor report (renders judged, code-verified mechanisms): task output 2026-07-30 — J1/J2/J3
  (implemented), J4/J5, D1–D3, C1–C6, width-1 generators A/B, suggested order.
- `research/10_scale_and_lamination.md` — measured lamination/scale facts.
- Memory: `synth-dataset-goal` — the binding goal statement.

## Gate-16 state (2026-07-30, advisor round-2 landed)
All numeric gates green on news6 (seeds 7400, box): D1 0.46% / D2 0.001% / D3 1.95% /
COVER unlabelled-fg 2.9% (old 0.7, real GT 24 partial) / STRAIGHT ~22 = real range (stacking-axis metric).
Wall 1m45 per 10 cubes. Render: FFN/gate16.png (airy 0.31 / mid 0.42 / dense 0.47 / old / real).
Landed: winner-take-all labels; pap_acc phase field; stationary PSD air background; Voronoi fill removed;
fleck point process + solved detail amps + asymmetric clip; fg_target steering (accept loop 3->2);
FFT streaks; 1/8 own field; hoisted supmin.
OWED before bulk regen (user decides bulk): re-point leakage_probe at compose(); delam census on final
config; single-vs-fused-lamina discriminator; 30-cube visual sample; deposit dense-rewrite, --jobs 72,
ref-index pickling (advisor 3.3/3.6).

## Pre-bulk battery (2026-07-30, all run on news7/news8, box)
- delam_census.py: SYNTH split-med 0.101 (donor IQR 0.080-0.242), gap-mode 3 vs donor 4, med-thick 7 vs 10
  (= calibrated squeeze). PASS.
- leakage_probe2.py (composer-specific; old leakage_probe.py tested retired v1 mechanics): air ΔAUC +0.014
  PASS; contact ΔAUC +0.079 OK (residual core-swap texture tell — watch).
- fused-lamina discriminator (in delam_census.py): synth 6.84% vs donors 3.89%; excess attributed to
  squeeze-closed same-sheet ply seams (intended supervision). Donor mask leakage is the real 3.89% channel.
- 30-cube visual: FFN/sheet30.png. Residuals: deep-stratum zigzag kink echo (2/30), left-border bright
  slivers (~5/30), deep fg 0.50-0.55 vs 0.70 band (fg steering not wired for deep).
- Speed: ref-index pickling + pool initializer (OMP_NUM_THREADS=1) bit-identical; 30 cubes / 2m17 at
  --jobs 72; projected 400-cube bulk 12-15 min.
Bulk regeneration decision: USER.

## Geo-diversity round (2026-07-30, advisor round-3: scroll-geometry critique)
Critique verdict: single-valued height-field deposit => self-contact (the ONLY contact=CONTINUE case),
hairpins/M-folds/eddies unrepresentable; stack normal pinned to one axis; no collinear tear hard-negative.
LANDED (sheet_compose.py):
- fold_warp_crop(): divergence-free vortex/saddle fold warp (analytic RK backward map, ONE resample per
  volume), applied post-composite PRE-harmonize (no strain-grain tell); small-angle SO(3) (<=17 deg);
  center-crop from SC_CANVAS=232 margin (kills first-wall corrugation + fill_frac edge); voids fill from
  the stationary air field. Knobs: SC_GEODIV (0 = axis-pure for gates), SC_WARP_P=0.55, SC_ROT_P=0.8,
  SC_CANVAS, SC_OUT.
- tear junctions (SC_TEAR_P=0.10): two donors collinear across a ragged cut at the SAME settled height —
  the missing false-merge hard negative.
- two-octave load_field (contacts open/close along one pair); burn-in roughness on first obstacle;
- crease band now per-sheet-gain (0.35-1.15): apexes aligned, amplitude varies — killed the zebra chevron.
Renders: gate17.png (untamed), gate18.png (tamed, same seeds). Eddy+hairpin+self-contact visible (09202),
tears (09203/04), tilted stacks. Axis-pure gates UNCHANGED: D1 0.41 D2 0.001 D3 2.14 COVER 3.1 (news11).
NOTE: fastgates ax=2 metrics are only valid on SC_GEODIV=0 batches (folds rotate the local stacking axis).
Wall: 12 warped cubes (232-canvas) / 2m18 at jobs 24.

## Gap-closing round (2026-07-30 evening) — BULK BLOCKED on one open gap
CLOSED: deep-stratum fg steering (fgt=0.75, band 0.58-0.82); donor fused-lamina screen
(donor_screen.py -> /root/s1bank_excl.json, 44/1106 sheets excluded, SC_EXCL wired into compose);
bank rebuilt with smooth-field core-fill jitter (sigma-8 offsets).
OPEN — contact-channel leakage probe: pre-geo-round measured DAUC +0.079 (OK, news7 code state);
post-geo-round measures +0.14..0.15 (TELL) on news13/news14. NOT tears (SC_TEAR_P=0 unchanged 0.883).
Jitter made it worse per-voxel (0.881) and smooth barely better (0.872) vs exact-nearest 0.808 —
consider REVERT of core-fill jitter. Remaining suspects to bisect (add env knobs): two-octave
load_field (rim-dominated positives — smaller patches = more transition rim in the positive class),
burn-in first-obstacle roughness, donor screen. Also improve the probe itself: restrict positives
to FUSED patches (high local papv adjacency, no air between) instead of all inter-instance
boundaries — tear/open-gap faces contaminate the positive class with genuine edges.
Verified this round: D1 0.48 D2 0.001 D3 2.11 COVER 3.1 (news13, axis-pure) — all green.

## Contact-tell closure campaign (2026-07-30 night) — root cause FOUND, partially closed
Measurement fixed first: 0.808-vs-0.88 was DOUBLY confounded (different seeds + multiple code deltas).
Same-seed A/B: code delta only +0.039. Probe v2.1 (both classes through-material, air-free): REAL fused
contacts read as interior (0.652) — synth 0.77 (old) / 0.86 (new). Knob bisect (oct2/burn-in/screen/tears):
ALL flat -> not the movers. Jitter: BOTH variants worse, REVERTED to exact-nearest (bank rebuilt).
Compression-fusion v1 (SC_FUSE_MODE=squeeze) WORSE (0.897, D3 5.6%) — resampling low-pass + density band;
default stays coreswap. Per-cube attribution: AUC 0.82-0.94 in EVERY std cube -> pervasive.
ROOT CAUSE: inter-donor statistical seam — donors from different source cubes abut at fused contacts with
different first/second-order stats; a real contact is one continuous medium. FIX 1 LANDED: per-donor
core-level harmonization (60% pull toward composition median, mean-only, occ-gated) -> 0.862->0.835,
gates green (D1 0.51 D2 0.000 D3 2.02 COVER 3.1).
NEXT (to close remaining +0.18): (1) second-moment match — per-donor core STD pull (multiplicative,
~0.5 strength, same occ gate); (2) det-band/spectrum match per donor at harvest (rank-map each donor's
core hf marginal to bank median); (3) re-baseline gate: real 0.652 is ONE annotation set — add
GP-carve-aware real sampling; (4) fold-warp self-contacts are seam-free contact-continue supervision
(same donor) — verify probe on SC_GEODIV=1 batches after (1)-(2).
