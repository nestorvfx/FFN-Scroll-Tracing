# synth/ — data generation, metrics, eval (grouped by pipeline stage)

See the top-level `README.md` for the pipeline map and `SOTA_PLAN.md` for the current plan.

## DATA / PREPROCESSING
- `synth_merge.py` — core composition library (gap_collapse mechanics, shared geometry helpers). Imported by
  nearly everything below; not run directly.
- `sheet_compose.py` — the composer: settles REAL sheets (from unrelated instance cubes) into blind fused
  contact, intensity-calibrated per-seed against a real compact window. Gate: `test_compose.py` (8/8).
- `preharvest.py` — SC_BANK sheet cache (6.6–50× compose speedup, bit-identical output).
- `gen_corpus2.py` — corpus generator v3 (400 cubes, per-ref gates, 2-scroll refs, label completion with the
  dim-valley clamp, instance channel persisted to `labelsTr_inst/`).
- Augmentation mechanics: `artifacts.py` (CT rings/streaks + PASTA spectrum jitter), `crumple.py`
  (developable-cone folds; uses `_dcone_profile.npz`), `weld.py`, `svpv.py` (sub-voxel partial-volume seams),
  `render_blind.py` (real-window intensity refs), `render_layers.py` (invoked by synth_merge for renders).
- `mk_synthdataset.py` — Dataset230 assembly (059 + corpus, leak-safe split). Then `../prep_arm.sh`, then
  `append_inst_channel.py` (2-ch seg [binary, inst], parallel, atomic).
- QA/calibration: `analyze_compact.py` (real-window stats), `contact_visibility.py` (blind-contact fraction
  vs real — corpus gate), `audit_carve.py` (carve erosion / bridges), `leakage_probe.py`, `cc_validate.py`,
  `carve_ablation.py` (pending carve-width study, README_STATE §6.1c).

## TRAINING SUPPORT
- `flow_field.py` — flow arm target + watershed decode + real-data thin-region mask (`--selftest` = U1/U4/U8).
- `malis_blast.py` — radial-blast-weighted maximin reweight (U3; proves size-product weighting is backwards).
- `overfit_flow.py` / `overfit_test.py` — overfit-one-batch learnability gates (flow / affinity heads, U9).
- `offset_sweep.py` — affinity offset selection sweep.
- `bench_ddp.py` — DDP vs single-GPU throughput measurement (numbers quoted in ../train.sh).

## EVAL / METRICS
- `wrap_erl2.py` — **CTF, the single decision gate** (radial blast-radius cost, bootstrap CIs, `--selftest`,
  truth / truth_carved calibrations).
- `wrap_erl.py` — v1 wrap-ERL (kept: scored alongside for continuity by queue2/eval scripts).
- `slab_emit_flow.py` / `slab_emit_carved.py` (decode-v2 + affcarve, `--selftest`) / `gasp_decode.py` /
  `slab_predict.py` — slab prediction + decode emitters (flow / affinity-MWS / GASP / plain fiber).
- `diag_affinity.py`, `diag_affinity_indomain.py`, `diag_sw.py` — pre-decode diagnostics (per-channel AUC,
  visible-vs-blind strata; NEVER gate on pooled AUC — untrained prior-bias head scores 0.837 pooled).
- `tstr_eval.py`, `tstr_eval_aff.py` — held-out cube merger/coverage guardrails.
- `paired_boot.py` — paired bootstrap CI over shared held-out cubes (the ranking discipline).
- `measure_fusion.py`, `measure_blind.py`, `measure_thickness.py`, `measure_anisotropy.py`,
  `measure_real_geometry.py`, `measure_transport.py` — real-data measurements that calibrate the corpus.

Superseded one-off diagnostics/renders live in `../archive/synth/`.
