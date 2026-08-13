"""Central configuration for the FFN system.

Every hyperparameter that the design (DESIGN.md) fixes or exposes as a knob lives
here as a single dataclass so train/eval/preprocess share one source of truth and
a run is fully described by one serialized config.

Values are the design defaults (DESIGN.md 4.2, 5.3, 6). Measured corrections
from the on-box audit (2026-07) are noted inline.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class FFNConfig:
    # ---- geometry (DESIGN.md 5.3, 2.6) ----
    fov: int = 33                     # isotropic field of view (~2.1 lamination periods)
    delta: int = 8                    # FOV movement step (half a period)
    move_threshold: float = 0.9       # POM prob at a face-center to allow a move
    # MOVEMENT GATE (FFN candidate 3). "center" = the historical single-voxel read at
    # centre+delta*e_i; "facemax" = Januszewski et al. / google-ffn max over the whole
    # fov x fov face plane. MEASURED model-free at real training coordinates (move_reach.py,
    # delta=8, teacher-forced): "center" leaves 19.5% of seeds unable to take ANY first step,
    # reaches 6.5% of the walk lattice and covers 55.9% of the in-ball instance; "facemax"
    # gives 0.0% / 69.9% / 99.5%. MUST match between train and inference -- swapping only at
    # inference with center-trained weights ALTERS the merge/NERL trade (NOTE: the old headline
    # "0.065 -> 0.630" is RETRACTED -- it was an artifact of the since-fixed 33x33 face-size bug;
    # the re-measured numbers are merge 0.0055 -> 0.120 with NERL improving 0.120 -> 0.179, see
    # FINDINGS 3.8 / research/07). The shipped run-5b/run-6 checkpoints trained under the TRACER
    # profile (facemax, delta 6, walk_radius 12, train_offcentre) -- verified by dumping
    # ckpt_712500['cfg'] on 2026-07-31.
    move_gate: str = "center"          # "center" | "facemax"
    # POM canvas value for NOT-YET-VISITED voxels. This is the paper's pad_value, and it is
    # NOT the same quantity as tgt_lo (the BCE target for background) even though the code has
    # always used one number for both. google/ffn's current manual recommends pad_value 0.5
    # ("unknown"); logit(0.05) tells the net every unvisited voxel is confidently background,
    # which is exactly the wrong prior at the FOV faces where growth must happen.
    pom_init: float = 0.05            # 0.05 = historical; 0.5 = paper
    # PAPER POM RULE (Januszewski 2018 Methods): "all POM voxels within the FoV were updated,
    # except those that were previously updated by the FFN, had a prior value < 0.5, and a new
    # value larger than the prior value (this biased the network towards splits in areas where
    # predictions were not consistent between iterations)". Absent from this codebase until now;
    # both write paths overwrote unconditionally, so a voxel the FFN had already rejected as
    # background could be re-claimed by a later FOV -- exactly how a fill leaks across a blind
    # contact it had already refused. MEASURED (6 labelled real val cubes, ckpt_merge_best,
    # single decode): merge 0.071 -> 0.028, blind-contact merge 0.044 -> 0.000, cov 0.737 -> 0.703.
    pom_ratchet: bool = True
    # "step"     = freeze against the live canvas (also between refinement iterations at one FOV).
    #             STRICTER than the paper; this is the variant with the measured benefit above.
    # "position" = freeze only against the canvas on entry to the FOV position (literal paper
    #             reading). Measured to have NO effect here, because inf_fov_iters re-forwards
    #             each position and that inner loop is what the strict rule constrains.
    pom_ratchet_scope: str = "step"
    # RATCHET FREEZE THRESHOLD, in probability. The paper's rule freezes a previously-written
    # voxel whose prior is < 0.5 against being raised -- but 0.5 is a coin flip, and the model's
    # measured p on the neighbouring lamina is ~0.48 (research/10 A.2), so the rule commits
    # hardest exactly where the model is least informative. Sweep {0.5,0.3,0.2,0.1} on the panel
    # before trusting any value; 0.5 reproduces the paper/historical behaviour bit-for-bit.
    pom_ratchet_freeze_p: float = 0.5
    # RATCHET PARITY IN TRAINING. train_step historically NEVER passed `written` to apply_update,
    # so the network has never trained under the freeze rule it is deployed under (measured
    # inference-only effect at ckpt_merge_best: merge 0.071->0.028, blind 0.044->0.000, coverage
    # 0.737->0.703). True = allocate a per-example written mask and apply the rule in the unroll.
    # Default False: enabling it changes training dynamics and belongs to its own run.
    pom_ratchet_train: bool = False
    # POM / RECURRENT-STATE CORRUPTION (research/11 #1). Training has never produced a canvas holding
    # a FALSE claim, so the net is never asked to retract one; a merge is precisely a leak the
    # recurrence amplifies (POM says fg -> net re-confirms -> movement gate opens). With prob
    # `pom_corrupt_p` a trajectory gets a defect injected at a random step: a leak blob claiming a
    # patch of an ADJACENT instance, a hole erased from the true mask, or logit softening toward
    # pom_init. The target is untouched (always the true mask), so "this POM voxel is wrong, predict
    # low" becomes directly supervised. Canvas-level only -> cannot collapse a gap or drift a label.
    # Risk is OVER-injection teaching the net to distrust its POM channel -> under-extension; keep
    # p <= 0.3 and watch core_frac_1step. 0.0 = off and bit-compatible.
    pom_corrupt_p: float = 0.0
    pom_corrupt_kinds: str = "leak,hole,soften"   # which defect kinds are sampled (uniformly)
    pom_corrupt_leak_frac: float = 0.25   # leak ball volume <= this x seed volume in the patch
    pom_corrupt_hole_frac: float = 0.25   # hole ball volume <= this x seed volume in the patch
    pom_corrupt_soften: float = 0.5       # fraction of the way from the logit toward pom_init
    # --- default-off; each needs a training run to validate (see FINDINGS.md) ---
    train_offcentre: bool = False     # start the training FOV off-centre, not dead-centre
    min_fov_steps: int = 1            # refuse to commit an object whose fill never moved
    # Loss weight on voxels of a DIFFERENT instance. Now meaningful: `balanced_bce` normalises each
    # half by its SUM OF WEIGHTS, so this redistributes emphasis between air and neighbouring-lamina
    # voxels while pos:neg stays 50:50. Before that fix it silently rescaled the whole negative half
    # (w=4 -> negative term x2.6), which is why every previous experiment with it was uninterpretable.
    # 2.5 targets the measured deficit: AUC same-vs-air 0.988 (solved) vs same-vs-other-instance
    # 0.819, with mean predicted probability 0.48 on the neighbouring lamina -- i.e. the model is
    # undecided exactly where merges happen, and the contact band is 3.1% of voxels but only 6% of
    # the loss.
    w_other: float = 2.5
    w_blind: float = 1.0              # loss weight on different-instance voxels at a blind contact
    # POM soft targets: same instance vs everything else (DESIGN.md 5.3)
    tgt_hi: float = 0.95
    tgt_lo: float = 0.05
    # logit clamp for the recurrent POM accumulation (~logit(0.999))
    logit_clamp: float = 6.9

    # ---- COMMIT CALIBRATION (see `commit_threshold`) ----
    # How many times worse a MERGE is than a SPLIT. A merge between adjacent wraps shifts the
    # winding number of every wrap outside it and corrupts the global unwrap; a split is
    # recoverable by agglomeration. Elkan (2001): the cost-optimal posterior threshold is
    # R/(R+1). R = 1 is the pure calibration fix with no asymmetry expressed.
    #
    # MEASURED, full val panel, ckpt_135000, commit_seed_cc=True throughout (real merge is
    # quantised at 1/18 = 0.056 and blind at 1/10, so read the NERL column as the reliable one):
    #
    #   R        tau     real NERL   real merge   BLIND merge   instances
    #   legacy   0.500     0.278       0.278          0.5           30
    #   1        0.724     0.252       0.278          0.5           31
    #   3        0.860     0.225       0.278          0.5           36
    #   5        0.893     0.203       0.278          0.5           41
    #   10       0.921     0.155       0.111          0.2           47   <- the knee
    #   20       0.935     0.108       0.111          0.2           60
    #
    # R=10 is chosen deliberately and it is a POLICY choice, not an optimum: it costs 41% of NERL
    # to halve the BLIND merge rate. The justification is the error asymmetry -- a blind-contact
    # merge is the one failure nothing downstream can repair (geometric rules cannot see a fused
    # contact, and FFN resegmentation provably ACCEPTS it: both regrowths agree, so iou and both
    # consistencies pass with confidence), whereas splits have a measured ~5x recovery path via
    # agglomeration. We are paying in the recoverable currency to buy down the unrecoverable one.
    #
    # REVISIT when agglomeration lands (splits become genuinely cheap -> push R higher) or when a
    # larger evaluation corpus exists (18 real / 10 blind pairs cannot resolve these differences;
    # the two evals either side of this measurement swung real merge 0.056 -> 0.333).
    # Set commit_cost_ratio=1.0 for the pure calibration fix, or override for raw coverage.
    commit_cost_ratio: float = 10.0
    #
    # ...BUT THE DEFAULT BELOW OVERRIDES THIS, and the reason is the decisive measurement:
    # raising tau does NOT merely fragment objects, it permanently DISCARDS voxels. Sweeping the
    # seed budget on 3 labelled cubes (ckpt_140000, commit_seed_cc=True):
    #
    #   seed cap    coverage @0.5   coverage @0.921   gap
    #      200          0.566           0.274        51.6%
    #      400          0.670           0.344        48.7%
    #
    # The gap does not close as seeding doubles, and mechanically it cannot: a voxel the model
    # scores at p=0.75 is never committed at tau=0.921 by ANY seed, so it is lost rather than
    # merely unvisited. Fragmentation is recoverable (instances 20 -> 48+); lost voxels are not.
    #
    # That is decisive given where this pipeline is going. In the target architecture (global
    # winding-phase field, research/06) merges are UNREPRESENTABLE by construction and fragments
    # reassemble transitively -- both free. Nothing refunds a voxel that was never committed. So a
    # high threshold pays in the only currency the architecture cannot repay, to buy down the one
    # error it already eliminates. Half the papyrus discarded also fails the stated objective
    # ("trace fully, in voxels") directly.
    #
    # RAISE the threshold (drop this override) if you need a merge-safe operating point BEFORE the
    # assembly stage exists: tau=0.921 measured all_merge 0.202 -> 0.049 on 183 pairs and blind
    # 0.5 -> 0.2, which is real and well-resolved -- it simply costs coverage we cannot get back.
    # Positive fraction of a training FOV, i.e. mean(target > 0.5). Training measures this every
    # step (`frac_active`) and records an EMA in the checkpoint, so the commit threshold
    # self-calibrates to the corpus instead of being guessed. The default is the value measured
    # on the run-5 corpus; a checkpoint's own figure overrides it.
    train_pos_frac: float = 0.251
    # Explicit override in output units. None = derive from the two knobs above (preferred).
    # Default 0.5 = the MEASURED optimum, not a legacy leftover. The Elkan derivation above says
    # 0.724 (symmetric) or 0.921 (R=10); empirically both cost coverage for no gain the architecture
    # needs, i.e. the model is NOT calibrated the way the derivation assumes -- which is exactly the
    # caveat the calibration literature attaches to it. Set to None to use the derived value.
    commit_threshold_override: Optional[float] = 0.5
    # Keep only the connected component containing the seed. Strict Pareto win when measured.
    commit_seed_cc: bool = True
    # What to do when the SEED VOXEL ITSELF is below the commit threshold (p~0.48 straddle case):
    #   "keep"    = historical: return the unrestricted mask -- which silently DISABLES the
    #               seed-CC merge guard exactly where merges are made (a mask with several
    #               disconnected components, possibly on different wraps, commits under one id);
    #   "nearest" = snap to the nearest above-threshold voxel within seed_pad+2 and keep ITS
    #               component (preserves the guard and the object);
    #   "reject"  = commit nothing (maximally merge-safe, costs coverage).
    commit_seed_miss: str = "keep"

    # ---- network (DESIGN.md 4.2) ----
    fmaps: int = 32                   # feature maps in the conv stack
    depth: int = 8                    # number of residual modules (6-12)
    # Residual skip homotopy (ffn.model.ResModule). 0.0 = the historical `relu(x) + F(x)` this file
    # actually implemented (an in-place ReLU aliased the skip); 1.0 = `x + F(x)`, the reference
    # google/ffn convstack_3d block. DEFAULT 0.0 so run-5 checkpoints keep their exact function --
    # hard-switching a trained model flips 55% of voxel decisions (loss 0.352 -> 6.493). Ramp it to
    # 1.0 over a few 10k steps to adopt the reference architecture continuously, which is also the
    # precondition for function-preserving depth growth.
    skip_alpha: float = 0.0
    in_channels: int = 2              # image + POM (+ negative evidence when neg_channel)
    # NEGATIVE-EVIDENCE CHANNEL (research/13 Lever #1). pom_init(0.05) == tgt_lo(0.05) and
    # apply_update overwrites raw logits, so "visited and confidently rejected" and "never visited"
    # are THE SAME NUMBER to the network: the recurrent state has full dynamic range for positive
    # evidence and zero for negative. True -> carry a second canvas r = max over visits of (1-p),
    # fed as a third input channel. +864 params (in_conv1 2->3), zero-initialised so the function is
    # BIT-IDENTICAL at step 0 and ckpt_712500 fine-tunes directly. Also what makes pom_ratchet_train
    # meaningful: with r the freeze is visible, so safety need not be bought as global timidity.
    neg_channel: bool = False
    # SUPERVISED MOVEMENT HEAD (research/13 Lever #2). The deployed gate is max over the
    # (2*delta+1)^2 = 169-voxel face >= 0.9 -- an extreme-value test on a quantity trained under a
    # MEAN criterion. For that gate to be even 50/50 safe at a blind contact the per-voxel P(p>=0.9)
    # on the neighbouring lamina must be < 1-0.5^(1/169) = 0.41%; measured mean p there is 0.45. No
    # improvement in a mean fixes a tail that strict. True -> a ~460-param head pools the face slab
    # of the final feature map into 6 gate logits, supervised by a target movement.training_move
    # ALREADY computes and discards. Gives a second, calibratable knob so merge control stops being
    # the same scalar as growth control.
    move_head: bool = False
    move_head_lambda: float = 0.2     # weight of the aux gate BCE (ramped from 0)
    move_head_warm: int = 10000       # steps driving movement by facemax before switching to learned
    move_head_pos_weight: float = 1.0  # >1 penalises FALSE-OPEN (merge-ward) more than false-closed
    move_threshold_learned: float = 0.5  # theta_move on the learned gate logit (calibratable)

    # ---- partitions / coordinates (DESIGN.md 5.4) ----
    # lom_radius = fov//2 + delta ; 17 fill-fraction bins per the paper
    partition_bins: List[float] = field(default_factory=lambda: [
        0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.075, 0.1,
        0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # Collapse fix (run-1 post-mortem): equal-per-bin sampling gave the near-empty
    # lowest-fill bins (~0.01% of candidates) a large fixed batch share whose target is
    # ~95% "background", teaching the lazy predict-low solution. Exclude bins whose lower
    # edge is below this from Phase B/C sampling (Phase A already uses >=0.05).
    min_bin_edge: float = 0.02
    min_instance_size: int = 2000     # drop tiny partial edge sheets from training
    coord_margin: Optional[int] = None  # raises the derived margin; never lowers it (see `margin`)
    # Refuse to train on an index whose candidates would produce EDGE-REPLICATED patches. Only set
    # False for toy phantoms in tests, where the volume is deliberately smaller than a real patch.
    # On real data this guard is what stands between you and a silently wasted run -- see
    # volumes._check_index_margin and FINDINGS.md 3.7.
    strict_margin: bool = True

    # ---- training (DESIGN.md 6) ----
    batch_per_gpu: int = 32
    lr: float = 1e-3
    # AdamW DECOUPLED decay: the per-step pull is lr*wd. At lr=1e-4 and wd=1e-4 that is 1e-8, and the
    # measured ratio of effective-step to decay-pull is 1e3-1e5 per tensor -- regularisation was
    # functionally OFF for the whole run. Meanwhile the real arm is measurably overfitting (train
    # balanced-BCE 0.295 vs val 0.351 on real; synth train/val are identical at 0.366/0.365, so it is
    # specifically the 130 real crops being memorised). 1e-2 gives ~10% shrinkage over 100k steps.
    # NOTE this is also why raising real_frac would be the WRONG response to the synth question: more
    # real per batch worsens the arm that is already overfitting.
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    max_steps: int = 300_000
    warmup_steps: int = 20_000        # Phase A (T=1)
    rampB_steps: int = 80_000         # Phase B end (T ramp 1->max_unroll)
    max_unroll: int = 8               # T at full curriculum
    walk_radius: int = 16             # max FOV displacement from seed in a training example (= 2*delta)
    seed_pad: int = 1                 # radius of the high-confidence seed disc (vox)
    additive_pom: bool = False        # False=absolute-logit overwrite (paper); True=additive-in-logit (ablation)
    teacher_force_steps: int = 60_000 # steps of teacher-forced (in-instance) movement

    # ---- WSD extension of a COMPLETED run (see scripts/train.py::lr_at) ----
    # 0 = plain cosine. >0 = the step at which a Warmup-Stable-Decay schedule takes over,
    # normally the step of the checkpoint being extended. Cosine is horizon-DEPENDENT, so a
    # finished run cannot be lengthened by raising max_steps -- that recomputes the same cosine
    # and jumps the LR back up ~100x into a converged model, which is the documented cause of
    # instability when re-warming from the minimum (Ibrahim et al. 2024).
    extend_base_step: int = 0
    # Stable-phase LR. 1e-4 = 0.1x the original peak; continued-pretraining recipes use
    # 0.05-0.1x for the same reason -- enough to make progress, not enough to destroy the
    # converged solution. Raise only with evidence.
    extend_peak_lr: float = 1e-4
    extend_warm_steps: int = 5_000     # linear re-warm out of the annealed state
    extend_decay_frac: float = 0.25    # final fraction of the extension spent annealing
    # ---- augmentation (DESIGN.md 5.7; policy from the 2026-07 stress-test) ----
    # Runtime GPU augmentation, one transform per patch held across the unroll.
    # Every magnitude below is the measured SAFE operating range for thin (5-6 vox)
    # sheets: it does not collapse an inter-wrap gap, thin a sheet below ~3 vox, or
    # drift the image<->label boundary. Break points found by the sweep are noted.
    # All parameters are continuous-random; each op is probability-gated per example.
    augment: bool = True
    # -- geometry (spatial_augment): octahedral(exact) x continuous rot/scale/shear,
    #    fused into ONE affine grid_sample (img bilinear, label nearest, reflect pad).
    aug_geom: bool = True
    aug_rot_deg: float = 16.0      # max |rotation| about a random 3D axis (break ~35deg: fragmentation; 16 keeps the fused-affine label integrity SAFE)
    aug_scale: float = 0.12        # max per-axis anisotropic content-scale dev (break ~0.35: thinning; pure downscale is the known corruptor)
    aug_shear: float = 0.06        # max shear (break ~0.5: fragmentation)
    aug_p_geom: float = 0.9        # prob of the continuous part (octahedral orientation always applied)
    # -- appearance (intensity_augment): all label-invariant, each prob-gated
    aug_gain: float = 0.25         # multiplicative contrast +/- (very safe; no break <=0.8)
    aug_bias: float = 0.10         # additive brightness +/- (break ~0.4)
    aug_gamma: float = 0.5         # gamma exponent in [1/(1+g),1+g] (break exp 2.0 -> gap-cue loss)
    aug_contrast: float = 0.30     # contrast about mean +/- (very safe)
    aug_noise: float = 0.05        # additive gaussian sigma (break ~0.18: boundary drift)
    aug_blur_sigma: float = 1.20   # max gaussian blur sigma (break ~2.5: inter-wrap band erased)
    aug_biasfield: float = 0.30    # smooth low-freq multiplicative inhomogeneity (break ~0.8)
    aug_p_gain: float = 0.8
    aug_p_bias: float = 0.5
    aug_p_gamma: float = 0.5
    aug_p_contrast: float = 0.4
    aug_p_noise: float = 0.7
    aug_p_blur: float = 0.3
    aug_p_biasfield: float = 0.4
    # optimisation / precision
    amp_dtype: str = "bfloat16"       # Blackwell strong bf16
    # CPU-resident cube stacks for corpora larger than GPU memory (3560 cubes = ~75 GB vs 32 GB
    # card). Only the cropped training patches move to the GPU per step: ~13 MB/optimizer-step at
    # B=32/patch=65, negligible against ~2 opt-steps/s. True = historical all-on-GPU behaviour.
    cubes_on_gpu: bool = True
    # AsyncPatchLoader look-ahead when stacks are CPU-resident: how many future steps' batches are
    # gathered/staged in parallel worker threads. 2 covers steady-state overlap; raise only if
    # step-time jitter shows the loop waiting on get().
    prefetch_depth: int = 2
    channels_last: bool = True
    compile: bool = True
    # NOTE: plain "max-autotune" captures CUDA-graphs, which are fragile with the recurrent
    # multi-forward-per-step DDP+grad-accum loop once on-GPU augmentation perturbs the graph
    # memory pool (RuntimeError: output of CUDAGraphs overwritten). Keep the autotuned
    # triton kernels but DROP cudagraph capture -- negligible speed cost on this tiny net.
    compile_mode: str = "max-autotune-no-cudagraphs"

    # hard-negative mining (Phase C): oversample centers near inter-sheet contacts
    hardneg_frac: float = 0.5         # fraction of Phase-C batch drawn from contact voxels
    hardneg_radius: int = 6           # vox from an inter-sheet contact
    contact_radius: int = 2           # a contact = >=2 instances within this radius (5^3 nbhd)
    # AUDIT FIX 3: make "hard" mean BLIND (no CT-intensity gap at the interface) rather
    # than any contact. Only blind contacts are the metric-relevant merge cases; gapped
    # seams are easy. A contact voxel is blind if the local-min intensity over the contact
    # window stays above gap_intensity_frac * mean(fg) (no dark gap voxel nearby).
    blind_hardneg: bool = True
    gap_intensity_frac: float = 0.5   # blind if local-min intensity >= this * mean(fg)

    # ---- data / splits (DESIGN.md 5.6) ----
    synth_val_n: int = 60             # held-out synth cubes (stratified 12 deep / 48 std)
    seed: int = 1234
    # AUDIT FIX 1: real harmonized instance cubes (ash2txt volumetric-instance-labels,
    # intensity-harmonized to the slab). Mixed into Phase C at real_frac of the batch; a
    # real_val_n holdout gives checkpoint selection a REAL-appearance gate (not synth-only).
    real_frac: float = 0.20           # fraction of Phase-C batch drawn from real cubes
    real_val_n: int = 15              # held-out real cubes for the real val gate

    # ---- inference / agglomeration (DESIGN.md 6.7) ----
    inf_move_threshold: float = 0.9
    inf_move_gate: str = "center"      # keep EQUAL to move_gate; see the note above
    # Train/infer parity fix (run-1 post-mortem): in training the FOV can stay put and
    # re-forward with its own POM (confidence builds over steps); single-pass inference
    # broke that contract and starved movement. Re-forward each visited FOV until the
    # POM converges (max prob change < tol) or iters is hit.
    inf_fov_iters: int = 6
    inf_fov_tol: float = 0.05
    inf_seed_min_edt: float = 1.0     # min EDT for an inference seed
    seed_spacing: int = 6             # grid-thin inference seeds to <=1 per spacing^3 cell
    seed_reject_radius: int = 3       # discard seeds within N vox of a committed segment
    fill_max_steps: int = 20_000      # safety cap on a single flood-fill
    # DROPPED the 2x term. Multi-scale consensus works in the paper because 2x changes the FOV in
    # PHYSICAL units; our isotropic-rescale sweep found coverage/merge strictly monotone with no
    # optimum across 5 factors, so there is no second scale that behaves differently here. What the
    # term actually contributed was np.repeat quantisation -- verified visually as axis-aligned
    # blocks cutting across inter-lamina gaps. Measured: cov 0.664 -> 0.446 and NERL 0.104 -> 0.040
    # for a merge reduction that is shattering, not separation. Forward+reverse at 1x is retained.
    consensus_scales: List[int] = field(default_factory=lambda: [1])

    # --- unlabelled core probe (movement telemetry on the TARGET regime) ---
    # Directories of held-out core cubes (imagesTr/*_0000.tif). No labels needed: the probe
    # reports frac_1step only. Empty list disables the probe.
    eval_core_dirs: List[str] = field(default_factory=list)
    eval_core_cubes: int = 2          # cubes sampled per directory
    eval_core_seed_cap: int = 60      # seeds per core cube (keeps the probe ~1 min)
    # An object is PHYSICALLY VALID if it did not cross into the neighbouring wrap. Scaled by the
    # cube's OWN measured wrap period lambda, never an absolute voxel count -- lambda is ~69 vox on
    # Scroll-3 cores and ~19 on Scroll-4, so a constant would be right for one scroll at most.
    # 0.6 is derived: a single sheet occupies duty*lambda with duty measured 0.34-0.50 on real
    # cores, a merged pair occupies >= 1.0*lambda; 0.6 sits in the gap.
    core_thick_frac: float = 0.6
    eval_core_topk: int = 200         # objects thickness-tested per cube (largest first)
    consensus_directions: int = 2     # forward + reverse seed order
    agglo_radius: int = 5             # candidate-pair search radius (vox)
    agglo_consistency: float = 0.6    # mutual reclaim fraction to merge two segments
    # AUDIT FIX 2: the regrow-at-interface agglomeration merge test is UNSAFE at blind
    # contacts (a fill reseeded on a blind interface grows into both wraps -> false merge,
    # the exact catastrophe). Splits it would recover are downstream-free. So default it OFF:
    # ship the oversegmentation-consensus output as canonical. Set True only with a learned
    # (not regrow) merge classifier.
    agglo_enabled: bool = False
    # --- RESEGMENTATION agglomeration (ffn/agglomerate.py), the 2018 Online Methods criterion ---
    # The verdict that gated agglo_enabled=False was measured against an interface-seeded
    # straw man (FINDINGS 3.3). The real test seeds at the EDT maximum INSIDE each fragment,
    # removes both segments, runs TWO independent regrowths, and accepts only if they agree.
    agglo_iou: float = 0.8            # IoU between the two regrowths (they must describe one object)
    # max fraction of A u B left unclaimed by BOTH regrowths. The paper uses 0.02; MEASURED here a
    # flood fill leaves 14.4% of a COMPLETELY UNAMBIGUOUS single sheet unclaimed (thickness-
    # independent -- it is the FOV-reach/`inb` constraint, not partial volume), so 0.02 is
    # unreachable in this pipeline by construction. 0.25 sits above that structural floor while
    # still catching genuine model confusion. Documented departure; the discriminating power is in
    # `agglo_iou` (measured 0.95 same-sheet vs 0.00 different-wrap) and mutual consistency.
    agglo_deleted_frac: float = 0.25
    agglo_retries: int = 8            # retry-under-exclusion when a regrowth is degenerate

    # ---- inline validation during training (live merge-rate curve) ----
    # Every eval_every steps, decode a FIXED panel of held-out val cubes (real + synth,
    # never trained on; slab untouched) and log adjacent-wrap merge rate to
    # val_progress.jsonl. Coarse single-run decode, panel split across DDP ranks:
    # ~30-60 s per eval -> a few % overhead at 10k cadence.
    # PANEL SIZE IS THE BINDING CONSTRAINT ON EVERY DECISION IN THIS FILE, and it was set far too
    # small. At 3 real cubes the panel yields 18 adjacency pairs and 10 blind pairs, so real_merge
    # quantises at 1/18 = 5.6% and blind at 10%: 0/18 vs 3/18 is Fisher p = 0.23, i.e. NOT
    # significant. Measured eval-to-eval sigma on real_nerl is 0.029-0.057, while the expected gain
    # from 100k further steps is ~0.02 -- the instrument could not resolve the thing it existed to
    # measure, and the 0.4461 "peak" at 240k is +3.7 sigma above the run mean (a maximum-of-N
    # artifact, P~0.40 over 24 draws).
    #
    # splits.json holds 30 real + 60 synth val cubes; we were using 3 + 3. Raising to 20 real gives
    # ~120 real pairs (quantum 0.8%) and ~60 blind. Cost is bounded by raising eval_every in step:
    # measured 44 s for 6 cubes across 2 ranks, so ~3 min for 26 cubes -> ~5% overhead at a 12.5k
    # cadence, against ~2% before. That is the correct trade: an eval that cannot resolve its own
    # metric is not cheap, it is worthless.
    # WHERE THE BUDGET GOES. Measured on the 26-cube panel: 279 s single-rank (~140 s across the 2
    # DDP ranks, which do split the panel). A synth cube costs ~3x a real one -- 23-28 instances per
    # cube against 8-9 -- so 6 synth cubes were ~47% of the eval bill while contributing nothing the
    # run is selected on (synth_nerl reads ~0.01 whatever we do; see below). Spending the budget on
    # real cubes instead keeps the decision-relevant statistics at full strength: 96 real adjacency
    # pairs and 86 blind pairs, against 18/10 before. ~95 s across 2 ranks = ~2.9% overhead at this
    # cadence. Synth is kept only as a tripwire (a sudden change means the corpus or loader moved).
    eval_every: int = 12_500
    eval_cubes_real: int = 20
    eval_cubes_synth: int = 6         # FULL synth panel (built at init, decoded from the step below)
    eval_cubes_synth_early: int = 2   # ...only this many until `eval_synth_full_step`
    # Absolute step at which the full synth panel switches on. 0 = always full. Set to
    # resume_step + 50_000 so the cheap panel covers the settling period right after a loss/config
    # change, then the fuller synth tripwire returns. Panel membership is fixed at init either way,
    # so metrics stay comparable across the switch.
    eval_synth_full_step: int = 0
    eval_crop: int = 128              # center-crop val cubes to this (cheaper decode)
    eval_seed_cap: int = 200          # max GLOBAL (EDT-descending) seeds per cube
    # ...plus this many seeds INSIDE each GT instance. An EDT-ranked global cap is not a uniform
    # subsample: it starves thin instances, which is why synth (23-28 instances/cube) decoded to 5-6
    # objects with coverage 0.38 and NERL ~0 for the whole run. See inline_eval.fair_seeds.
    eval_seed_per_inst: int = 8
    eval_seed_spacing: int = 8        # coarse fixed seed grid (comparable across steps)
    # Eval-harness seed field. False = historical: inference_seeds on the UNION mask
    # (fiber=(inst>0)), whose Blum ridge lies ON the contact plane wherever two wraps touch --
    # the panel then measures a pathology its own harness manufactures. True = per-instance
    # medial seeds (seeds.instance_inference_seeds). Changes panel numbers; flip only at a
    # re-baseline and say so in RUN_STATE.
    eval_instance_seeds: bool = True
    eval_fill_max_steps: int = 4000   # per-object fill cap during inline eval

    # ---- eval (DESIGN.md 7) ----
    adjacency_radius: int = 3         # GT wraps adjacent if within this many vox
    merge_theta: float = 0.10         # a pred label merges a pair if it claims >=theta of both
    gateA_merge_rate: float = 0.05
    gateA_voi_merge: float = 0.5
    gateA_bridge_ok_frac: float = 0.90
    eval_seed_sets: int = 5
    # ---- SOTA METRIC FIX (run-1 collapse post-mortem) ----
    # The primary objective is Expected Run Length (NERL, metrics.erl), selected under a
    # merge-rate CONSTRAINT -- the Januszewski et al. 2018 FFN protocol: "the checkpoint
    # with the highest expected run length among the set with the least mergers". merge-rate
    # is coverage-BLIND (an empty/collapsed segmentation scores the optimum 0.0, which is
    # exactly how run-1's collapse looked "better"), so it is only ever a CONSTRAINT. NERL
    # has a coverage floor (an uncovered skeleton node zeros the run), so it cannot be gamed
    # by collapse and is the metric to maximize.
    # NERL-best: only ckpts with all-panel merge <= this are eligible. Raised 0.05 -> 0.25 because
    # 0.05 was UNREACHABLE at the shipped operating point (all-panel merge 0.202), so ckpt_best.pt
    # and ckpt_core_best.pt were NEVER WRITTEN for the whole of run-5 -- the primary selection rule
    # was silently inert for 140k steps. A gate nothing can satisfy is not a safety margin.
    # This is a collapse guard, not the objective: NERL is already coverage-floored and
    # merge-catastrophic (a merge zeroes the runs of BOTH wraps), so it cannot be gamed by a model
    # that merges everything. The real fix is a larger evaluation corpus -- 183 pairs quantises this
    # at 0.005 and the 18-pair real subset at 0.056, which is why two consecutive evals of the same
    # model swung real merge 0.056 -> 0.333.
    select_merge_eps: float = 0.25
    gateA_nerl: float = 0.25          # Gate-A COVERAGE FLOOR: NERL must exceed this (empty seg -> NERL 0 -> FAIL)
    # SECOND best-checkpoint criterion (faithful to the FFN protocol: "fewest mergers first,
    # ERL to break ties" -- Januszewski 2018). NERL alone over-penalizes over-splitting, which
    # is downstream-FREE in our economics, so it can mis-rank a low-merge checkpoint below a
    # higher-NERL one. So we ALSO track ckpt_merge_best = the lowest REAL merge rate among
    # checkpoints that clear a NERL coverage floor (the floor keeps collapse -- NERL~0 -- from
    # ever winning it). The held-out slab arbitrates NERL-best vs merge-best at the end.
    select_nerl_floor: float = 0.015  # merge-best eligibility: all-panel NERL must be >= this (rejects collapse)

    # ---- paths (parameterized; set by scripts from env/CLI) ----
    corpus_dir: str = "/root/surf/data/synthfuse_corpus"
    slab_dir: str = "/root/data/slab"
    work_dir: str = "/root/surf/ffn_work"     # preprocessed indices, checkpoints, splits

    def as_tracer(self) -> "FFNConfig":
        """Retrain profile: unstick the movement gate, change nothing else that removes a brake.

        REVISED after run-3 (2026-07-27) diverged. The first version changed seven coupled things at
        once, every one of which made growth EASIER, and it collapsed to a single object per cube at
        1.6x the true fibre volume with merge 1.0. Post-mortem in FINDINGS.md 3.7. What survives:

          move_gate/inf_move_gate  face-max over the (2*delta+1)^2 cuboid face (the paper's rule).
                                   This is the ONE change that has to happen: the centre-voxel gate
                                   samples a single voxel at centre +- delta, which on an oblique
                                   ribbon is off-sheet at most azimuths. Measured on Scroll-4 core,
                                   fraction of seeds that cannot take a first step: centre@0.9 0.74,
                                   face-max@0.9 0.012.
          delta                    8 -> 6; delta 8 is 0.84*lambda_nn on the Scroll-4 core, so one step
                                   crosses most of the way to the neighbouring lamina.
          walk_radius              16 -> 12 (see the dataclasses.replace below; the prose "24" was
                                   stale) so the model
                                   sees a POM shaped like a trace. NOT larger: `margin` derives from
                                   train_patch, and on 192^3 cubes a bigger patch shrinks the legal
                                   candidate region toward the cube centre. walk_radius 48 gave
                                   patch 129 and only ~4% of the volume was legal.
          train_offcentre          start the FOV off-centre: P(object at FOV edge) is ~0 in training
                                   and ~1 at inference.

        DELIBERATELY REVERTED -- each removed a brake, none was required by the gate change:

          move_threshold 0.6   ->  0.9.  Taken from the paper's manual, not from our data. Our own
                                  probe says face-max@0.9 already drops the Scroll-4 stall fraction
                                  to 0.012 -- identical to 0.6 -- while 0.6 left 5.55 of 6 directions
                                  open. It bought nothing and removed the gate.
          pom_init 0.5         ->  0.05. The strongest growth-permissive lever: it makes every
                                  unvisited voxel read "unknown" instead of "background", so nothing
                                  outside the fill resists being claimed. Defensible in the paper's
                                  sparse-neurite setting; in a dense laminar volume the hard problem
                                  is knowing where the object ENDS.
          w_other 4.0          ->  1.0. Weights different-INSTANCE voxels 4x while air stays at 1x,
                                  so it relatively de-emphasises the air term -- in a corpus that is
                                  already 73% papyrus against 34% on real core. Direction defensible,
                                  magnitude unvalidated, and it pushed the wrong way here.
          min_fov_steps 2      ->  1. Drops objects whose fill did not move. In an over-growing model
                                  that keeps only the sprawling ones: run-3 committed ONE object at
                                  50k. A diagnostic, not a selection rule.

        Train from scratch, NOT from a run-2 checkpoint -- the gate change alters the movement
        distribution the optimizer state was built around.

        RE-PREPROCESS after changing walk_radius: `margin` is derived from `train_patch`, so a stale
        index will be rejected at startup (see volumes._check_index_margin).
        """
        import dataclasses
        return dataclasses.replace(
            self, move_gate="facemax", inf_move_gate="facemax",
            delta=6, walk_radius=12, train_offcentre=True)

    @property
    def lom_radius(self) -> int:
        return self.fov // 2 + self.delta

    @property
    def commit_threshold(self) -> float:
        """Threshold on the network's OUTPUT for committing a voxel to the object.

        Derived, not tuned. `balanced_bce` weights the positive and negative halves of each FOV
        equally while the object occupies only a fraction f of it, so the output is a prior-shifted,
        soft-target-squashed image of the true posterior q:

            p*(q) = tgt_lo + (tgt_hi - tgt_lo) * r*q / (r*q + 1 - q),    r = (1 - f) / f

        Committing at the cost-optimal posterior q* = R/(R+1) therefore means thresholding the output at

            tau = tgt_lo + (tgt_hi - tgt_lo) * (r*R) / (r*R + 1)

        The old hard-coded 0.5 corresponds to q = 0.25 at the measured f = 0.251 -- i.e. it committed
        every voxel a quarter-likely to be ours, which is the dominant source of adjacent-wrap merges.
        """
        if self.commit_threshold_override is not None:
            return float(self.commit_threshold_override)
        f = min(max(self.train_pos_frac, 1e-3), 1 - 1e-3)
        r = (1.0 - f) / f
        rr = r * max(self.commit_cost_ratio, 1e-6)
        return float(self.tgt_lo + (self.tgt_hi - self.tgt_lo) * rr / (rr + 1.0))

    @property
    def commit_posterior(self) -> float:
        """The true posterior the commit threshold corresponds to (for logging/auditing)."""
        R_ = max(self.commit_cost_ratio, 1e-6)
        return float(R_ / (R_ + 1.0))

    @property
    def train_patch(self) -> int:
        """Side of the training patch cropped per example (the FOV plus its whole walk range)."""
        return self.fov + 2 * self.walk_radius

    @property
    def margin(self) -> int:
        """Edge margin candidate seeds must respect.

        MUST cover HALF THE TRAINING PATCH, not just the inference FOV+delta. `batched_crop`
        resolves out-of-bounds coordinates with `.clamp_`, i.e. EDGE REPLICATION -- an overhanging
        patch gets the boundary slice extruded, together with its instance label, into a prism
        running to the patch edge. That fabricates objects which extend in a straight line forever,
        which is precisely what an FFN must not learn.

        This bit us hard (2026-07-27). `margin` used to return `lom_radius = fov//2 + delta`, sized
        for the inference walk only. Raising walk_radius 16 -> 48 grew the patch 65 -> 129 while the
        margin stayed at 22, so **91.2% of training patches were edge-replicated** with bands up to
        40 voxels deep (run-2: 30%, <=8 voxels). The model learned to grow without bound -- by 50k
        steps it emitted ONE object per cube at 1.6x the true fibre volume, merge rate 1.0.
        Deriving the margin from `train_patch` makes the failure unreachable by construction.
        """
        need = self.train_patch // 2
        return max(need, self.lom_radius) if self.coord_margin is None \
            else max(self.coord_margin, need)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FFNConfig":
        with open(path) as f:
            d = json.load(f)
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})
