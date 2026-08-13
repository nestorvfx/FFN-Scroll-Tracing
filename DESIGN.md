# 3D Flood-Filling Networks for Papyrus-Sheet Instance Separation — DESIGN & PLAN

**Status:** Design complete, ready to implement. No open TODOs.
**Author:** Analysis/Design agent (Fable 5).
**Scope:** This document is the single source an implementing engineer follows. It specifies architecture, data preparation, training, inference/agglomeration, evaluation, an ordered build checklist with unit tests, risks/mitigations, and a full citation list. It does **not** contain implementation code beyond illustrative pseudo-config; the second agent writes the code.

> **Reading guide.** Sections 1–3 are the "why" and the verified data facts. Sections 4–8 are the "what to build." Section 9 is the ordered checklist. Sections 10–11 are risks and references. If you only read one thing before coding, read §4 (architecture), §5 (data → FFN format), §7 (evaluation), and §9 (checklist).

---

## 0. Executive summary — the decisions

| # | Question | Decision |
|---|----------|----------|
| 1 | Which FFN variant | **A from-scratch PyTorch reimplementation of the original Januszewski et al. FFN core** (recurrent single-object flood-fill with a compact 3-D pre-activation ResNet), GPU-native, AMP + `torch.compile`, DDP across both 5090s. We do **not** fork `google/ffn` (TF1, unmaintained, no modern GPU features) and we do **not** switch to a single-pass successor (LSD/affinities/Omnipose) — we already measured single-pass methods merge ~80% of blind contacts; FFN's recurrence is the whole point. |
| 2 | Network | 2-input-channel (image, POM) 3-D CNN: input module (conv–ReLU–conv) + **8 full-pre-activation residual modules** (ReLU–conv–ReLU–conv) + 1×1×1 logit head; 3×3×3 kernels, 32 feature maps, SAME padding, ~0.5 M weights. |
| 3 | Field of view | **Isotropic 33×33×33** (≈ 2 lamination periods) so the network always sees the target sheet **and** both neighbours and can learn to *stop at the gap*. Deltas **8,8,8**; movement threshold **0.9**; POM soft targets **0.95 / 0.05**. |
| 4 | Training data | **Synthetic corpus (400 cubes, per-sheet instance labels)** is the primary asset — it manufactures unlimited correctly-labelled blind-contact geometry. Augment with **real harmonized instance cubes** (must be fetched from HF; see §3.4) for appearance grounding. Dataset059 (real binary) is **not** used for the recurrent instance loss (no instances) but is available for optional encoder pre-training. |
| 5 | Seeds | **Medial-axis (3-D skeleton) seeding per instance**, not blob-EDT peaks — our objects are thin sheets, so seeds must be guaranteed interior to a single lamina. |
| 6 | Held-out eval | The **z=10192 slab (64×2042×1667, 68 wraps)** is strictly held out. Primary north-star metric: **adjacent-wrap pairwise merge rate** (maps 1:1 to the winding-number catastrophe). Secondary: **ERL, VOI(split/merge), Adapted-Rand**, computed with `funlib.evaluate` semantics. |
| 7 | Synth gate | Before ever touching the slab, gate on a held-out **synth-val split** (stratified std/deep) using instance merge-rate + VOI + a carved-skeleton bridge count. |

---

## 1. Problem & why FFN (tied to our pipeline)

### 1.1 The pipeline and the failure mode
Surface detection predicts a papyrus-fiber mask from scroll CT. A downstream tracer (`villa` / volume-cartographer, `spiral2.cpp`) **Guo–Hall thins** that mask into a skeleton graph and fits a winding spiral. Two adjacent wraps stay distinct **only** where the mask has a genuine zero-gap between them. Wherever touching sheets are connected by even a thin bridge of foreground, the thinning bridges them, the skeleton merges, and the **winding number of every outer wrap shifts** — a catastrophic *merge* error that corrupts the entire spiral outward of the bridge.

In compact / condensed scroll regions, **~73% of adjacent-wrap contacts have no local intensity gap** ("blind" contacts): the CT simply shows continuous bright material where two sheets press together. Every **single-forward-pass** model we built — binary+carve, affinities, an Omnipose-style distance/flow field — **failed** on real blind contacts: ~80% of adjacent wrap-pairs still merge regardless of the decode. The reason is structural: a single forward pass has no mechanism to *propagate a boundary decision* from a visible anchor (where a gap is locally resolvable) into an adjacent blind fused stretch. The decision at each voxel is made independently, from local evidence that, by definition, is absent in blind contacts.

### 1.2 Why FFN is the right class of model
Flood-Filling Networks (Januszewski et al., arXiv:1611.00421, 2016; *Nature Methods* 15:605, 2018) are the connectomics state of the art for exactly this regime. An FFN is a **recurrent 3-D CNN that grows one object at a time from a seed**, maintaining a *predicted object map* (POM) that accumulates the network's evidence about which voxels belong to the current object. Because the POM is fed back as a second input channel, the network at each step **conditions on its own prior high-confidence decisions**: easy/visible regions of a sheet inform the hard/blind stretches. This is precisely the boundary-propagation mechanism our single-pass models lack.

FFN is **merge-averse by construction**: it commits to *one* object and asks "does this voxel continue the current object?", rather than partitioning everything at once. In the standard neuron-segmentation benchmark (Sheridan/Funke et al., *Nat. Methods* 20:295, 2023, Local Shape Descriptors) FFN achieves the **lowest merge error of any method measured** (VOI-merge 1.188 vs 2.741 baseline affinities), at the cost of somewhat higher split error and much higher compute. For our problem, **a split is a nuisance (recoverable by agglomeration) but a merge is catastrophic** (irrecoverable winding shift). This asymmetry is exactly FFN's bias. The original FFN also holds the field's error-free-path-length record (mean error-free neurite path ~1.1 mm; ~4 mergers per 97 mm — an order of magnitude better than affinity+watershed).

We want **3-D** FFN because our data is inherently volumetric (192³ synth cubes; a 64-thick real slab); the depth dimension carries real sheet continuity that a 2-D per-slice tracer throws away.

### 1.3 What "success" means here
A separated instance segmentation of the held-out slab in which **adjacent wraps are almost never assigned the same label** (target: adjacent-wrap pairwise merge rate < 5%, from the current ~80%), even across blind fused contacts, while tolerating a moderate rate of *splits* (a wrap broken into 2–3 pieces) that downstream agglomeration or proofreading can stitch.

---

## 2. Data we have — verified on the box (2× RTX 5090, 32 GB each, 64 cores, 251 GB RAM)

All facts below were **measured directly** on `root@90.84.236.62` (read-only), not assumed.

### 2.1 Synthetic corpus — the crown jewel  `/root/surf/data/synthfuse_corpus/`
- **400 cubes**, each **192×192×192**.
- `imagesTr/<id>_0000.tif` — CT, **uint8, range 0–255** (full 256 levels used).
- `labelsTr/<id>.tif` — **binary** fiber mask (uint8 {0,1}), inter-sheet gaps carved.
- `labelsTr_inst/<id>.tif` — **per-sheet instance labels, uint16**, `0 = background`, then contiguous sheet IDs `1..N`. Measured example: 28 sheets in one cube; **median 27 sheets/cube**.
- `synthfuse_meta.json` — per-cube metadata: `sheets`, `fg` (foreground fraction), `stratum` (`std`/`deep`), `aug` (`pasta` spectral aug present on a subset), `lapvar`, `pen`, `mean`.
- **Stratum split: 320 `std` + 80 `deep`** (20% deep = whole-context fused mottle, hardest merger regime, always-winding).
- **Foreground fraction: median 0.61** (deep cubes up to ~0.74) — i.e. these are *dense/fused* scenes, exactly the hard regime we want to train on.
- **This is the only asset with exact per-voxel wrap identity in dense fused scenes.** It is the primary FFN training source.

### 2.2 Real harmonized instance cubes — **NOT PRESENT on the box** (prerequisite)
The path named in the brief (`/root/data/cubes/instance-labels-harmonized/`) **does not exist**; the only directory under `/root/data` is `slab/`. These are the composer's *source* sheets (real appearance + real instance structure) and are valuable for domain grounding. **They must be fetched from Hugging Face** `nestorvfx/ScrollData` (read token in the brief) — look under `01_base_dataset` and `04_resources`. **Action for the implementer:** locate, download, count, and inspect them *before* finalizing the train split (§5.6). If they cannot be recovered, training proceeds on synth-only + a domain-adaptation plan (§10, Risk R4); this is acceptable but weaker.

### 2.3 Real binary cubes (Dataset059)  `/root/surf/data/nnUNet_raw/Dataset059_s1_s4_s5_patches_frangiedt/`
- **1754** real annotated scroll patches, **binary fiber only** (`labels: {background:0, fiber:1}`), Tiff3D.
- **No instances** → cannot supply the FFN recurrent instance loss. Used *only* (optionally) for encoder pre-training / domain appearance (§6.6).

### 2.4 Preprocessed Dataset230  `/root/surf/data/nnUNet_raw|preprocessed|results/Dataset230_059plus_synthfuse/`
- **2153** training items = 1754 (059) + ~400 synth. nnU-Net format, binary label semantics in `dataset.json`. The synth entries here are the 2-channel `[binary, inst]` variant. **We do not train FFN from Dataset230**; we read instances straight from the raw synth corpus (§2.1) to avoid nnU-Net's resampling/normalization. Listed for completeness only.

### 2.5 Eval slab — held out  `/root/data/slab/`
- `volume.nrrd` — **64×2042×1667, uint8 (0–248 observed)**.
- `truth.nrrd` — **64×2042×1667, uint16**, **68 traced wraps** (69 unique values incl. background; IDs non-contiguous, e.g. 1,2,25,26,…,440).
- `bbox.json` — `z0=10192, y0=2802, x0=2471`; umbilicus at **local (y,x)=(1018,821)**; `label_fraction=0.2295` (of the labeled sub-region), overall **foreground fraction 0.156**.
- Per-wrap voxel counts: **min 83 (partial edge wraps), median 223 k, max 2.94 M**.
- **Note the geometry:** the slab is only **64 voxels thick in z**. Wraps appear as long thin arcs in the (y,x) plane, ~4 vox thick in the radial direction, extending along the winding direction and through the 64-vox z. Any FOV depth ≤ 64 fits.

### 2.6 Geometry measurements that drive FFN hyperparameters (measured, not assumed)
- **Sheet thickness ≈ 4 voxels** in BOTH synth and slab (median EDT half-thickness = 2.0 → thickness 4.0). Synth per-sheet *max* thickness ≈ 13 vox (at folds/stacks). Slab p90 thickness ≈ 7.5 vox.
- **Lamination period ≈ 15–16 voxels** (sheet ~4 + gap ~11–12), per project memory and consistent with the measured thickness.
- **Domain gap to note:** synth cubes are *denser* (fg median 0.61) than the real slab (fg 0.156). Synth deliberately over-represents the fused regime (good for the hard cases) but the encoder must also generalize to sparser real regions — mitigations in §5.5/§10.

**Implication for FOV:** an isotropic **33³** FOV spans ≈ 2.1 lamination periods, so it always contains the target lamina plus at least one neighbour on each side — the network can *see* the gap it must respect. This is the single most important adaptation from neuron-FFN (blobby, tens of voxels thick) to sheet-FFN (thin, 4-vox laminae). See §5.3.

---

## 3. Hardware & software baseline
- **2× RTX 5090 (32 GB each)**, 64 CPU cores, 251 GB RAM.
- Python `/root/surf/venv/bin/python`, **torch 2.8.0+cu128**, scipy, skimage, tifffile, pynrrd, numpy. `source /root/surf/env.sh` for paths.
- 5090 = Blackwell (sm_120); torch 2.8+cu128 supports it. **Use bf16 AMP** (Blackwell has strong bf16), `torch.compile` (inductor), channels-last-3d where it helps.
- The original `google/ffn` config trained on a **12 GB P100**; our 32 GB cards give ample headroom for larger batches / deeper nets / `torch.compile` graphs.

---

## 4. Architecture — the chosen model and why it is the efficient GPU-native option

### 4.1 What exists (surveyed), and why we pick a clean PyTorch reimplementation

| Option | What it is | Verdict for us |
|--------|-----------|----------------|
| `google/ffn` (TF1) | Original reference implementation; `compute_partitions.py`, `build_coordinates.py`, `train.py`, `run_inference.py`; configs `depth=12, fov_size=[33,33,33], deltas=[8,8,8]`. Trained/tested on a single P100; "not configured for multi-GPU." | **Reference, not runtime.** TF1 is EOL; no bf16/compile/DDP ergonomics on Blackwell. We mine it for the *algorithm* (partitions, seed policy, POM update, movement, agglomeration) and reimplement in PyTorch. |
| `diluvian` (Keras/TF) | Community FFN re-impl; defaults to a U-Net body; weights loss by fill fraction rather than resampling. | Reference for design choices (U-Net body is an option; fill-fraction weighting is a valid alternative to 17-class resampling). Not maintained; TF/Keras. |
| **LSD / MTLSD / AcRLsd** (Sheridan & Funke, *Nat. Methods* 2023) | Single-pass affinities + 10-D local-shape-descriptor auxiliary; **~2 orders of magnitude faster** than FFN, competitive VOI. | **Rejected as the primary method:** it is *single-pass* — the exact class we already measured to merge ~80% of blind contacts. LSDs improve boundary detection but still decide each voxel from local evidence. We keep it in mind only as an *auxiliary head* (§6.5) and as an *agglomeration/oversegmentation* front-end. |
| Mutex-watershed / PatchPerPix / MALIS | Single-pass affinity decoders / structured losses. | Same single-pass limitation. Not primary. |
| Parallel/cross-classification FFN (Meirovitch et al. 2016/2019) | Flood-fills many objects at once / classifies cross-sections. | Interesting for inference throughput but adds complexity and is not needed at our data scale; noted as a future speed option, not v1. |

**Decision:** implement a **faithful PyTorch FFN core** (the recurrent single-object grower), because (a) the recurrence is the mechanism that solves our problem and no single-pass successor replaces it; (b) a clean PyTorch build gives us bf16, `torch.compile`, DDP, and batched multi-seed inference for free on the 5090s, which is what "efficient GPU-native in 2026" means in practice; (c) the network is tiny (~0.5 M weights), so training cost is dominated by data movement and the recurrent unroll, both of which PyTorch handles well. This is *more* efficient than resurrecting TF1 and *more* correct than any single-pass model for blind contacts.

### 4.2 The network (exact spec)
Faithful to Januszewski 2018, Fig. 5 ("full pre-activation residual modules", He et al. 2016):
- **Inputs:** 2 channels, shape `[B, 2, D, H, W]` with `D=H=W=33`.
  - Channel 0: CT image crop, normalized (§5.2).
  - Channel 1: current **POM** (logit-space seed field; see §6.2), initialized to the seed prior and updated recurrently.
- **Body:** all convolutions 3×3×3, **SAME** padding (zero-pad to preserve size), **32 feature maps**, ReLU.
  - **Input module:** `Conv3d(2→32) → ReLU → Conv3d(32→32)`.
  - **8 residual modules**, each full-pre-activation: `y = x + Conv3d(ReLU(Conv3d(ReLU(x))))` (32→32→32).
  - **Head:** `Conv3d(32→1, kernel 1×1×1)` → single-channel **logit** map, SAME size 33³.
- **Parameter count** ≈ 0.47 M (matches the paper's 472,353). No batch-norm (FFN uses none; keeps the recurrent statistics stable). 
- **Runnable config anchor** (from `google/ffn`): `{"depth": 12, "fov_size": [33,33,33], "deltas": [8,8,8]}`. Here `depth` in the TF config counts conv layers in the module stack; our "8 residual modules + input module + head" = 19 conv layers total, matching the paper. Implement **8 residual modules** and expose `depth` as a config knob (allow 6–12 for ablation).
- **Optional body swap (ablation flag, not default):** a small 3-D U-Net body (diluvian-style) can replace the ResNet stack behind the same 2-in/1-out interface. Keep the ResNet as default per the paper's convergence finding.

### 4.3 Efficiency choices (2026, Blackwell)
- **bf16 autocast** for the forward/backward; keep POM accumulation and loss in fp32.
- **`torch.compile(model, mode="max-autotune")`** — the model is static-shape (33³), ideal for inductor.
- **DDP over 2 GPUs** for training (§6.4): each rank draws independent seed batches; grads all-reduced. ~2× throughput.
- **Batched recurrence:** unroll a *fixed* number of FFN steps per training example (§6.1) with the whole batch on-GPU; no host round-trips inside the unroll.
- **channels_last_3d** memory format for the conv stack.
- Inference: **batched multi-seed** flood-fill (process K seeds' FOVs as a batch), plus block-wise tiling for the 2042×1667 slab (§6.7).

---

## 5. Data plan — turning our assets into FFN-ready training data

### 5.1 Which assets, for what
- **Train (recurrent instance loss):** synth corpus instance labels (§2.1) + real harmonized instance cubes (§2.2, once fetched).
- **Synth-val (in-domain gate):** a held-out, stratified subset of the synth corpus (§5.6).
- **Held-out real test:** the slab (§2.5) — never seen in training or gating.
- **Optional encoder pretrain:** Dataset059 binary (§2.3) — self-supervised or binary-fiber warm-up of the input module only (§6.6).

### 5.2 Image normalization
CT is uint8 0–255 in both synth and slab. Normalize **identically** at train and inference:
`x = (img.astype(float32) / 255.0 - 0.5) / 0.5` → range ≈ [−1, 1]. (Do **not** use nnU-Net's per-image z-score; FFN wants a fixed global mapping so the POM channel and image channel stay comparably scaled.) Record the exact transform in a `norm.json` shipped with the checkpoint.

### 5.3 FOV, deltas, POM — adapted to thin sheets
- **FOV = 33×33×33** (isotropic). Rationale in §2.6: spans ≈2 lamination periods → target lamina + neighbours in view → the network can learn the *stop-at-gap* boundary. Smaller FOVs (e.g. 25³, ~1.5 periods) risk seeing only the target and one neighbour and are offered as an ablation, not default. Larger (49³) wastes compute and dilutes the thin-structure signal.
- **Deltas = 8,8,8.** A face-step of 8 vox moves the FOV center by half a period. **Crucial thin-sheet behaviour:** a step in the *radial* (through-thickness) direction lands on a gap or a neighbour lamina, where the POM is low, so the movement gate (POM ≥ 0.9 at the candidate center) **naturally suppresses cross-lamina jumps**. Steps *along* the sheet (in-plane winding direction, and through z) keep the center inside the same lamina and are the productive moves. This is why deltas 8 work despite 4-vox thickness — we rely on the gate, not on the step being smaller than the thickness. (Ablation: deltas 6 for finer control if edge cases appear.)
- **POM soft targets:** voxels of the **same instance as the FOV-center's instance = 0.95**, all others (background *and* other sheets) **= 0.05**. This is the standard FFN target and is exactly what teaches "other sheet = not me," i.e. the separation objective.
- **Movement threshold = 0.9** on POM probability at candidate face-centers (matches the paper). 
- **Fill-fraction partitioning:** compute per-example active fraction `f_a` and bin into the paper's **17 classes** (thresholds `t = [0,0.01,0.02,0.03,0.04,0.05,0.06,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]`); sample each class **equally** during training (via `build_coordinates` balancing, §5.4). Our thin sheets cluster at *low* fill fraction (a 4-vox lamina in a 33³ FOV fills ≈ 4/33 ≈ 12% per crossing), so class balancing is what prevents the net from collapsing to "predict thin." Keep all 17 classes; expect most mass in classes ≤ 0.2 — that is fine and intended.

### 5.4 Coordinate/partition generation (our analogue of `compute_partitions` + `build_coordinates`)
Reimplement in PyTorch/NumPy, operating **per instance** on the `labelsTr_inst` volumes:
1. **`lom_radius`** (local-object-mask radius) `= fov//2 + delta = 16 + 8 = 24 → [24,24,24]`. For each voxel `A`, compute the **quantized fraction of voxels within radius `lom_radius` sharing A's instance label** — this is the "partition" value. Skip instances smaller than `min_size` (set `min_size = 2000` voxels to drop tiny partial edge sheets; the slab has wraps down to 83 vox but those are eval-only).
2. **Quantize** the fraction into the 17 partition bins.
3. **`build_coordinates`:** emit a list of `(volume_id, x, y, z, instance_id, partition_bin)` training locations such that **each partition bin is represented ≈ equally**. Respect a **margin = lom_radius (24)** from volume edges so a full FOV+delta always fits. Persist as a compact `.npy`/`.parquet` index (not TFRecord).
4. **Seed policy for training centers = medial-axis restricted (§5.5).** The center voxel of each training example must be strictly interior to a single lamina; use the skeleton mask to filter candidate centers.

### 5.5 Seeding — the key thin-sheet adaptation (train and inference)
Neuron-FFN seeds by EDT local peaks of a Sobel-thresholded image (blob interiors). Our sheets are thin, so an EDT peak is only ~2 vox from a boundary and a mis-seed easily straddles two laminae. Instead:
- **Training seeds:** per instance, compute the **3-D medial axis** (`skimage.morphology.skeletonize_3d` on the instance mask, or EDT ridge). Sample FOV centers **on the skeleton** (guaranteed one-lamina-deep, maximally interior along thickness). This yields seeds that are unambiguously inside one sheet and spread along its extent.
- **Inference seeds (slab / unlabeled):** we do not have instances at inference. Seed from the **predicted fiber probability** (or the existing binary surface model's mask): (a) threshold to a fiber mask, (b) EDT, (c) **skeletonize**, (d) take skeleton voxels as candidate seeds in descending EDT order. Discard any seed within **3 voxels of an already-committed segment** (paper rule) to avoid re-seeding the same sheet. This gives dense along-sheet seeding while respecting thickness.
- **Seed prior in the POM:** initialize channel-1 POM with a small high-confidence disc at the seed (logit ≈ +logit(0.95) inside a 1–2 vox ball, logit(0.05) elsewhere), matching the paper's initialization.

### 5.6 Train / synth-val / held-out split
- **Held-out real test:** the slab. Untouched until §7.3.
- **Synth-val:** hold out **60 of the 400 synth cubes**, **stratified** to preserve the 80/320 deep/std ratio → **12 deep + 48 std**. Selection by hashing the cube id (deterministic, documented). The remaining **340** cubes train.
- **Real harmonized cubes (once fetched, count = R):** if `R ≥ 30`, hold out ~15% as a **real-instance mini-val** (a second, tougher gate closer to slab appearance); use the rest for training. If `R < 30`, put all into training and rely on synth-val + slab.
- **No cube appears in two splits.** Log the split manifest to `splits.json`.

### 5.7 Augmentation
The corpus already bakes CT-artifact + PASTA spectral aug. Add cheap, geometry-preserving on-the-fly aug (isotropic data, so full octahedral symmetry is legal):
- Random **90° rotations** and **flips** over all 3 axes (24-element rotation group). 
- Small **intensity jitter** (±10% gain, ±0.05 bias) and mild **Gaussian noise** (σ≈0.02) to close the synth→real appearance gap.
- **Random affine ±10°** small-angle rotations + ±5% scale (trilinear for image, nearest for labels), *sparingly* — thin sheets are sensitive to interpolation; verify labels stay 4-vox thick after warp (unit test §9).
- **Do not** add elastic warps that thin sheets below ~2 vox (would create false gaps). 

---

## 6. Training procedure

### 6.1 The recurrent training step (one example)
Per training location (from §5.4):
1. Crop the CT FOV (33³) and build the POM channel initialized at the seed (§5.5). Set the **target** = same-instance soft mask (0.95 in-instance, 0.05 else), 33³.
2. **Unroll the FFN for `T` steps** (start `T=1` and *schedule up*; see §6.3). At each step:
   - Forward the 2-channel input → logit map `Δ`.
   - **Update POM** in logit space: `POM_logit ← clip(POM_logit + Δ, [-C, +C])` over the current FOV (C≈logit(0.999)≈6.9). (This additive-in-logit update is the FFN recurrence; the paper stores POM as probabilities and rewrites the FOV — additive-logit is the numerically stable, standard PyTorch-FFN equivalent.)
   - Choose the next FOV center by the **movement policy**: among the 6 face-center candidates at ±delta, move to those whose POM probability ≥ 0.9 (queue them); for the *training* unroll use the paper's simplification of following the max-scoring in-bounds move (teacher-guided by the label to stay inside the instance for early curriculum, then policy-driven).
3. **Loss:** per-voxel **sigmoid (binary) cross-entropy** between the POM logits and the soft target, summed/averaged over the FOV, accumulated across the `T` unrolled steps. This is the original FFN loss (voxelwise logistic loss on POM). Weight by **class-balanced sampling** (§5.3), not by a loss reweight (keep loss clean); optionally add diluvian-style fill-fraction weighting as an ablation.
4. **Optional auxiliary losses (flags, default off for v1, see §6.5):** soft-clDice on the binarized POM vs instance skeleton (topology), and/or an LSD auxiliary head.
5. Backprop through the whole unroll (truncated BPTT of length `T`). Optimizer **AdamW**, lr `1e-3` cosine-decayed, weight decay `1e-4`, grad clip `1.0`, bf16 autocast.

### 6.2 POM representation
- Stored per FOV as **fp32 logits**; the image channel is normalized fp32→bf16 at autocast. The POM input channel to the net is `sigmoid(POM_logit)` mapped to [−1,1] (same scale as image) OR the raw logit clipped — pick one and unit-test both channels are O(1) magnitude (§9). Default: feed `2*sigmoid(POM_logit)-1`.

### 6.3 Curriculum (critical for FFN convergence)
FFN notoriously fails to train if you start with long unrolls on hard scenes. Schedule:
1. **Phase A (warm-up), ~20 k steps:** `T=1` (single step, i.e. plain seeded segmentation), seeds on **std** cubes only, only "easy" partition bins allowed (fill ≥ 0.05). Goal: learn the basic "same-sheet vs not" mapping.
2. **Phase B, ~80 k steps:** ramp `T` 1→**8**, add all partition bins, add **deep** cubes at their natural 20% rate. Enable full movement policy.
3. **Phase C, ~200 k+ steps:** full `T` (8–16), full corpus incl. real harmonized cubes, full augmentation, hard-negative mining (oversample locations whose center is within 6 vox of an inter-sheet contact — these are the blind-contact decision points; identify them from the instance labels as voxels with ≥2 distinct instances in a 5³ neighbourhood).
Log the movement-restricted vs free-policy loss separately.

### 6.4 Dual-GPU strategy
- **Training:** `torchrun --nproc_per_node=2`, DDP. Each rank has its own data loader over the shared coordinate index (disjoint shards). Effective batch = per-GPU batch × 2. Per-GPU batch target **64–128** examples of 33³×2ch (fits easily in 32 GB at bf16; the net is tiny — batch is bounded by the unroll activation memory, so tune with `T`). Gradient all-reduce each step. Save checkpoints from rank 0 only.
- **Inference:** the slab is large (2042×1667×64). Split into overlapping blocks (e.g. 384² × 64, halo = FOV+max-move = 24) and **assign blocks round-robin to the 2 GPUs**; each GPU runs independent flood-fills; stitch via the agglomeration pass (§6.7). Alternatively one GPU trains while the other runs eval during development.

### 6.5 Optional auxiliary heads (ablation, default OFF for v1)
- **soft-clDice** (Shit et al., CVPR 2021) on the binarized POM vs the current instance's skeleton — encourages topological continuity of the grown sheet (relevant because our objects are thin manifolds where connectivity is everything). Cheap-ish but adds a differentiable soft-skeleton (min/max-pool) cost; use **Skeleton Recall Loss** (Kirchhoff et al., ECCV 2024) as the cheaper drop-in if soft-clDice is too slow in 3-D.
- **LSD auxiliary** (10-D local shape descriptor regression) as a secondary head on the encoder — shown to improve boundary sharpness. Only add if v1 shows boundary bleed.

### 6.6 Optional encoder warm-up on Dataset059
Pre-train the **input module + first few residual blocks** as a binary-fiber predictor on the 1754 real binary patches (059), then load those weights before FFN training. This grounds the encoder in *real* CT appearance (059 is real; synth is composited) and can close part of the synth→real gap. Keep it a **flag**; measure with/without on synth-val and the real mini-val.

### 6.7 Inference & agglomeration (the decision-point pass)
Follow the FFN inference + agglomeration recipe, adapted:
1. **Seed** (§5.5, inference variant) → serial flood-fill each seed to a fixed segment; discard seeds within 3 vox of committed segments. Each committed segment gets a unique label.
2. **Oversegmentation-consensus** (the paper's 82×-fewer-mergers trick): run the flood-fill **forward and reverse** and at **2 scales** (full-res and 2× in-plane downsample). Take the **consensus**: two voxels share a label only if they co-segment in *all* runs. This deliberately **over-splits** (splits ×2) but **crushes mergers** (the paper: mergers ÷82). For us, over-splitting is cheap and merging is fatal — this trade is exactly right. **This is a required pass, not optional.**
3. **FFN agglomeration** to recover the splits we *want* to recover (adjacent pieces of the *same* wrap, never across a gap): for each pair of segments with voxels within a **5×5×5** radius, compute a **decision point** (midpoint of the shortest line joining them) and **reseed a flood-fill there**; merge the pair only if the flood-fill grown from the decision point covers both with mutual consistency (each segment's regrowth substantially reclaims the other). Because the gap between two *different* wraps has low fiber probability, a decision point in a true gap fails the consistency test → the wraps stay separate. Tune the consistency threshold on synth-val (§7).
4. Output: `uint16` instance volume aligned to the input, plus a per-voxel max-POM confidence map.

---

## 7. Evaluation protocol & metrics

Two gates: **(A) in-domain synth-val gate** (fast, run every checkpoint) and **(B) held-out slab test** (run rarely, only after A passes). All metrics computed with **`funlib.evaluate` semantics** (RAND, VOI, ERL, NVI/NID) plus our bespoke merge-rate.

### 7.1 Primary north-star: adjacent-wrap pairwise merge rate (slab & synth-val)
This maps **1:1** to the winding-number catastrophe and is the metric the whole effort exists to move.
- **Adjacency graph of GT wraps:** two GT wraps `i,j` are *adjacent* if any voxel of `i` lies within radius `r=3` of any voxel of `j` (they touch/nearly-touch — the contacts that can merge). Enumerate all adjacent pairs `P_adj`.
- **Predicted-merge test for a pair:** assign each GT voxel the predicted label; a pair `(i,j)` is **merged** if a single predicted label `L` covers ≥ `θ` fraction of *both* wrap `i` and wrap `j` (use `θ = 0.10`, i.e. one predicted segment claims ≥10% of each — robust to a few stray voxels). 
- **Adjacent-wrap merge rate** `= |merged pairs| / |P_adj|`. **Lower is better.** Current single-pass baseline ≈ **0.80**. **Target < 0.05** (SOTA), **stretch < 0.02**.
- Report the **blind-contact subset** separately: restrict `P_adj` to pairs whose contact region has **no local intensity gap** (compute: along the contact interface, fraction of interface voxels where CT stays above a gap threshold). This is the honest hard-case number.

### 7.2 Connectomics metrics (secondary, standard, comparable to literature)
Computed on the **skeletonized** wraps (each wrap → a medial curve, `skeletonize_3d`; edge lengths in voxels):
- **ERL (Expected Run Length)** — the FFN paper's own metric. **Definition (implement exactly):**
  1. For each GT wrap `k`, build its skeleton; total GT path length `L = Σ_k len(s_k)`.
  2. Map each skeleton node to its predicted segment ID.
  3. A predicted segment is **"merged"** if it contains skeleton nodes from **≥2 different GT wraps**.
  4. For each wrap, split its skeleton into **maximal contiguous runs** where consecutive nodes share one predicted segment ID **and that segment is not merged** (a merged segment contributes runs of length **0**). Let the run lengths be `{l_j}`.
  5. **`ERL = (Σ_j l_j²) / L`** — the path-length-weighted expected error-free run length. **Higher is better.** (Interpretation: pick a random point on a wrap weighted by length; ERL is the expected length of the error-free arc containing it. A single merge zeros out an entire long run — this is why ERL "disproportionately punishes merges," Sheridan/Funke 2023 — exactly our priority.)
  6. **NERL (normalized)** `= ERL / ERL_perfect`, where `ERL_perfect = (Σ_k len(s_k)²)/L` (the score of a flawless segmentation on this skeleton set). Report both ERL (voxels) and NERL (0–1).
- **VOI split & merge** (Meilă 2007; CREMI convention): `VOI = VOI_split + VOI_merge`, computed voxel-wise between predicted and GT instance volumes (background excluded). Report the two components separately — **`VOI_merge` is our secondary north-star** (FFN's known strength; target `VOI_merge < 0.5`). Lower is better.
- **Adapted-Rand error** (SNEMI3D convention) — a single-number over/under-segmentation summary; report Adapted-Rand-F, split-precision and merge-recall. Lower error is better.
- Use `funlib.evaluate` (RAND, VOI, ERL, NVI, NID; requires `graph_tool`) as the reference implementation; wrap it, and **unit-test our ERL against a hand-built toy** (§9) so we're not blindly trusting the dependency.

### 7.3 Protocol & "what good looks like"
- **Gate A (synth-val, every eval):** run full inference+agglomeration on the 60 held-out synth cubes. **Pass if:** adjacent-wrap merge rate < **0.05** *and* VOI_merge < **0.5** *and* the **carved-skeleton bridge count** (number of thinning bridges between distinct predicted instances after Guo–Hall thinning — the pipeline's actual failure signal) is **0** on ≥ 90% of cubes. Only checkpoints passing Gate A proceed.
- **Gate B (held-out slab, rare — final/near-final only):** full slab inference+agglomeration, compute §7.1 + §7.2 on the 68 wraps. **SOTA target:** adjacent-wrap merge rate **< 0.05** (from ~0.80), NERL **> 0.6**, VOI_merge **< 0.5**. Report the blind-contact merge subset prominently.
- **Seed-robustness:** because single-seed FFN ranking is noisy (project memory: seed-σ can dominate), report metrics as **mean ± σ over ≥5 seed sets** and rely on the **oversegmentation-consensus** output (§6.7) as the canonical result, not a single flood-fill.
- **Never** tune hyperparameters on the slab; all tuning on synth-val / real mini-val.

---

## 8. What could go wrong on thin laminar sheets (vs blobby neurons) — and mitigations

| Risk | Why it's specific to us | Mitigation |
|------|------------------------|-----------|
| **R1 — FOV too small to see the gap** | Neuron FFN never needs to represent a 1-voxel gap between two parallel walls; our whole task is that gap. If FOV shows only the target lamina, the net can't learn "stop." | FOV 33³ ≈ 2 periods guarantees neighbour laminae in view (§5.3). Hard-negative mining at contact voxels (§6.3 Phase C). Ablate 25³/33³/49³ on synth-val. |
| **R2 — Movement jumps across a blind contact into the neighbour** | With 4-vox thickness and 8-vox steps, a radial move could land in a neighbour lamina; at a *blind* contact the POM there may be high (no image gap). | Movement gate at 0.9 + oversegmentation-consensus (forward/reverse/multiscale) which *splits* rather than merges when runs disagree (§6.7). Curriculum teaches the net that "other sheet = 0.05 target." Optionally shrink deltas to 6 (ablation). |
| **R3 — Seeds straddle two laminae** | EDT peaks sit ~2 vox from a boundary in a 4-vox sheet → ambiguous. | Medial-axis (skeleton) seeding (§5.5) — one-lamina-deep by construction. |
| **R4 — Synth→real domain gap** | Synth is denser (fg 0.61) and composited; slab is sparser (0.156) real CT. Net may overfit synth texture. | Real harmonized cubes in training (§2.2/§5.6); 059 encoder warm-up (§6.6); intensity/noise aug (§5.7); real mini-val gate; PASTA aug already present. |
| **R5 — Thin-structure interpolation destroys labels** | Affine/elastic aug can thin a 4-vox sheet to 2 vox or open false gaps. | Nearest-neighbour label warp + post-warp thickness assertion (§9 test); no aggressive elastic warps (§5.7). |
| **R6 — Split explosion** | Oversegmentation-consensus intentionally over-splits; a wrap could shatter into many pieces, hurting ERL. | FFN agglomeration pass (§6.7) recovers same-wrap splits via decision-point reseeding; splits are recoverable (unlike merges). Tune consistency threshold on synth-val to balance. |
| **R7 — Recurrent training divergence** | FFN is finicky; long unrolls on hard scenes early = collapse (we've seen affinity collapse before). | Curriculum §6.3 (T:1→8, easy→hard), grad clip, logit clamp on POM, monitor movement-restricted loss separately, prior-bias init on the head. |
| **R8 — ERL non-monotonic on small volumes** | ERL under-weights terminals and fragments when skeletons exceed the volume (Sheridan/Funke 2023). Slab is only 64 thick. | Use **adjacent-wrap merge rate as the true north-star** (§7.1); treat ERL/NERL as corroborating. Skeletonize within-slab and normalize (NERL). |
| **R9 — Inference cost on the full slab** | FFN inference is serial/expensive; slab is 2042×1667×64. | Block-wise dual-GPU (§6.4), batched multi-seed, bf16+compile. The slab is small vs connectomics volumes; a few GPU-hours is acceptable. |
| **R10 — Harmonized cubes unrecoverable** | They're not on the box. | Synth-only training is a valid fallback (synth is the crown jewel); prioritize R4 mitigations harder (059 warm-up, stronger aug). Document the degraded expectation. |

---

## 9. Implementation checklist (ordered; each item names its unit test)

**Milestone 0 — Data acquisition & audit**
1. Fetch **real harmonized instance cubes** from HF `nestorvfx/ScrollData` (`01_base_dataset`, `04_resources`); count `R`, inspect shapes/dtypes/label semantics. *Test:* every cube loads, `0=bg`, instance IDs contiguous per cube, thickness ≈4 vox (EDT check).
2. Build a `manifest.json` of all training volumes (synth 400 + real R) with path, shape, n_instances, fg, stratum. *Test:* counts match §2 (400 synth; 320 std/80 deep).

**Milestone 1 — Preprocessing to FFN format**
3. Implement image normalization (§5.2) + `norm.json`. *Test:* round-trip range ≈[−1,1]; identical transform reload.
4. Implement **medial-axis seed extraction** per instance (§5.5). *Test:* on a synth cube, every seed voxel has exactly one instance label and EDT ≥ (thickness/2 − 1).
5. Implement **`compute_partitions`** (per-instance local same-label fraction, `lom_radius=[24,24,24]`, 17 bins, `min_size=2000`). *Test:* a synthetic 2-sheet toy gives fraction=1 deep inside a sheet, ~0.5 at a contact, 0 in background.
6. Implement **`build_coordinates`** (balanced across 17 bins, margin 24). *Test:* histogram of emitted bins is ≈uniform; no coordinate within 24 of an edge.
7. Implement the **train/synth-val/real-mini-val split** (§5.6) → `splits.json`. *Test:* disjoint sets; deep/std ratio preserved in synth-val (12/48).

**Milestone 2 — Model & recurrence**
8. Implement the **2-in/1-out 3-D pre-activation ResNet** (§4.2), config-driven depth. *Test:* forward on 33³ returns 33³ logits; param count ≈0.47 M at depth-8; SAME padding preserves shape.
9. Implement **POM logit update + FOV crop/paste** utilities (§6.1–6.2). *Test:* additive-logit update is idempotent when Δ=0; clamp bounds hold; channel magnitudes O(1).
10. Implement the **movement policy** (6 face candidates at ±delta, gate 0.9, visited-set at reduced resolution). *Test:* on a straight synthetic tube, the FOV walks along it and refuses a 0.05-POM lateral step.

**Milestone 3 — Training loop**
11. Implement the **recurrent training step** with truncated BPTT length `T`, sigmoid-CE POM loss, AdamW, bf16, grad-clip (§6.1). *Test:* single-example overfit — the net drives POM→target on one FOV within 200 steps (loss→~0). *(This is the go/no-go sanity gate; if a single FOV won't overfit, stop and debug.)*
12. Implement the **curriculum scheduler** (§6.3, T:1→8, easy→hard, deep-rate ramp, hard-negative mining at contact voxels). *Test:* scheduler emits correct T and bin-mask per step; contact-voxel miner finds voxels with ≥2 instances in a 5³ nbhd.
13. Wire **DDP (`torchrun --nproc_per_node=2`)**, disjoint shards, rank-0 checkpointing, `torch.compile`, channels_last_3d (§6.4). *Test:* 2-GPU run reproduces 1-GPU loss curve within noise for the first 500 steps at matched effective batch; throughput ≈2×.
14. (Optional flag) **059 encoder warm-up** (§6.6). *Test:* warm-started net matches cold net's forward interface; loads without shape errors.

**Milestone 4 — Inference & agglomeration**
15. Implement **inference seeding** from predicted fiber prob → EDT → skeleton, descending order, 3-vox discard rule (§5.5). *Test:* on a synth cube (using its binary label as the "fiber prob"), seeds land one-per-lamina and cover all sheets.
16. Implement **single-seed flood-fill to fixed segment** (queue, movement, termination). *Test:* on a synth cube, flood-filling from a seed recovers that seed's instance with IoU > 0.8 (before agglomeration).
17. Implement **oversegmentation-consensus** (forward+reverse × 2 scales, intersection labelling) (§6.7). *Test:* consensus never merges two GT instances that any single run kept apart (merge count monotonically non-increasing vs single run).
18. Implement **FFN agglomeration** (5³ candidate pairs, decision-point reseed, mutual-consistency merge) (§6.7). *Test:* on a synth cube with an artificially split single sheet, agglomeration re-merges it; on two adjacent different sheets with a carved gap, it does **not** merge.
19. Implement **block-wise dual-GPU slab inference** with halos + cross-block stitch (§6.4). *Test:* tiling a synth cube into overlapping blocks reproduces the whole-cube result within ε.

**Milestone 5 — Evaluation**
20. Implement **adjacent-wrap merge rate** (§7.1) incl. the blind-contact subset. *Test:* a perfect segmentation → 0.0; a segmentation that fuses two known-adjacent wraps → 1/|P_adj| increment; toy hand-check.
21. Implement/verify **ERL & NERL** (§7.2) — wrap `funlib.evaluate` and **cross-check against a hand-built toy skeleton** (two wraps, one merge → ERL drops to the known closed-form value). *Test:* toy matches analytic ERL.
22. Implement **VOI(split/merge) + Adapted-Rand** (via `funlib.evaluate`/skimage). *Test:* identical segmentations → 0; a single merge raises VOI_merge, a single split raises VOI_split.
23. Implement the **carved-skeleton bridge count** (Guo–Hall thin the predicted per-instance masks, count inter-instance skeleton bridges) — the pipeline's real failure signal for Gate A. *Test:* two cleanly separated sheets → 0 bridges; one fused pair → ≥1 bridge.
24. Wire **Gate A (synth-val)** and **Gate B (slab)** runners with mean±σ over ≥5 seed sets (§7.3). *Test:* Gate A runs end-to-end on 3 cubes and emits the metric JSON.

**Milestone 6 — Campaign**
25. Warm-up (Phase A) → confirm single-FOV overfit and basic seeded fill on synth-val.
26. Phases B→C full training; checkpoint every N; run **Gate A** each checkpoint; keep the Pareto-best on (merge-rate, VOI_merge, bridge-count).
27. Ablations on synth-val: FOV {25,33,49}; deltas {6,8}; ResNet vs U-Net body; ±059 warm-up; ±soft-clDice aux; ±real cubes.
28. **Once and only once** the best checkpoint clears Gate A convincingly, run **Gate B on the slab**; report §7.1+§7.2 with the blind-contact subset called out.

---

## 10. Open items resolved (no deferred unknowns)
- **Harmonized cubes location** — resolved: not on box; fetch from HF (§2.2, Milestone 0). Fallback path defined (R10).
- **FOV/deltas for thin sheets** — resolved by measurement (§2.6) + gate reliance (§5.3): 33³ / 8.
- **Seeding for thin sheets** — resolved: medial-axis (§5.5).
- **Loss & recurrence** — resolved: voxelwise sigmoid-CE on POM, truncated BPTT, curriculum (§6).
- **Merge vs split priority** — resolved: oversegmentation-consensus (split-favoring) + agglomeration recovery (§6.7); metrics weight merges (§7).
- **Metric definitions** — resolved to formula level (§7.2) with reference impl + unit tests (§9).
- **Dual-GPU** — resolved: DDP train, block-wise inference (§6.4).

---

## 11. References (URLs)

**Flood-Filling Networks (core method)**
- Januszewski, Maitin-Shepard, Li, Kornfeld, Denk, Jain. *Flood-Filling Networks.* arXiv:1611.00421 (2016). https://arxiv.org/abs/1611.00421
- Januszewski et al. *High-Precision Automated Reconstruction of Neurons with Flood-Filling Networks.* bioRxiv 200675 (2017) — full Methods (seed policy, FoV movement, POM soft labels 0.95/0.05, 17 partition classes, architecture Fig. 5, agglomeration, oversegmentation-consensus). https://www.biorxiv.org/content/10.1101/200675.full
- Januszewski et al. *Nature Methods* 15:605–610 (2018). https://www.nature.com/articles/s41592-018-0049-4 · PubMed https://pubmed.ncbi.nlm.nih.gov/30013046
- Google Research blog, *Improving Connectomics by an Order of Magnitude* (2018). https://research.google/blog/improving-connectomics-by-an-order-of-magnitude
- Reference implementation `google/ffn` (TF1; `compute_partitions.py`, `build_coordinates.py`, config `depth=12, fov_size=[33,33,33], deltas=[8,8,8]`). https://github.com/google/ffn
- `diluvian` — community Keras/TF FFN re-implementation (U-Net body option; fill-fraction loss weighting). https://github.com/aschampion/diluvian
- Ging et al. *Scaling Distributed Training of Flood-Filling Networks…* arXiv:1905.06236 (2019) — voxelwise cross-entropy on POM, distributed training. https://arxiv.org/pdf/1905.06236
- Kuhn/Reims et al. *Exploring Flood Filling Networks for Instance Segmentation of XXL-Volumetric and Bulk Material CT Data.* J. Nondestructive Eval. (2020) — FFN applied to **CT** (not EM). https://link.springer.com/article/10.1007/s10921-020-00734-w

**Efficient single-pass successors / alternatives (surveyed, not chosen as primary)**
- Sheridan, Nguyen, Deb, Lee, Saalfeld, Turaga, Manor, Funke. *Local Shape Descriptors for Neuron Segmentation.* Nature Methods 20:295–303 (2023) — LSD/MTLSD/AcRLsd, ~100× faster than FFN, benchmark table (FFN VOI-merge 1.188). https://localshapedescriptors.github.io · bioRxiv https://www.biorxiv.org/content/10.1101/2021.01.18.427039v2.full-text · PMC https://pmc.ncbi.nlm.nih.gov/articles/PMC9911350
- Funke et al. *Large Scale Image Segmentation with Structured Loss Based Deep Learning (MALIS).* IEEE TPAMI 41:1669 (2019).
- `PyTorch Connectomics` (affinity/U-Net toolbox, reference for 3-D conn-seg engineering). https://github.com/zudi-lin/pytorch_connectomics

**Thin-structure / topology-preserving losses (for auxiliary heads & robustness)**
- Shit et al. *clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation.* CVPR 2021 (soft-skeleton via min/max-pool; Algorithms 1–2). https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.pdf
- Kirchhoff et al. *Skeleton Recall Loss for Connectivity-Conserving and Resource-Efficient Segmentation.* ECCV 2024 (cheaper differentiable-skeleton alternative to soft-clDice). https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09904.pdf

**Evaluation metrics**
- `funkelab/funlib.evaluate` — reference RAND, VOI, ERL, NVI, NID (requires graph_tool). https://github.com/funkelab/funlib.evaluate
- Meilă. *Comparing clusterings — Variation of Information* (2007) — VOI split/merge.
- Arganda-Carreras et al. *SNEMI3D / Adapted-Rand error* — instance seg metric.
- Plaza & Funke. *Analyzing Image Segmentation for Connectomics.* Front. Neural Circuits 12:102 (2018) — ERL vs VOI properties, why ERL punishes merges. https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2018.00102/full
- He, Zhang, Ren, Sun. *Identity Mappings in Deep Residual Networks* (full pre-activation residual modules — the FFN body). ECCV 2016. arXiv:1603.05027.

**Architecture / efficiency**
- Köpüklü et al. *Resource Efficient 3D CNNs.* ICCV-W 2019 (depthwise/inverted-residual 3-D blocks — reference for the optional efficient body). https://openaccess.thecvf.com/content_ICCVW_2019/papers/NeurArch/Kopuklu_Resource_Efficient_3D_Convolutional_Neural_Networks_ICCVW_2019_paper.pdf

---

*End of DESIGN.md — no deferred TODOs. Implementer starts at Milestone 0.*
