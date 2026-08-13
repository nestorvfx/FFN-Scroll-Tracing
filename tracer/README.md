# tracer/ — production whole-volume FFN decode

The production inference stack for tracing sheets through arbitrary volumes with the trained FFN.
Deliberately a **separate module** from `FFN/ffn/inference.py`: that path serves training-time
evaluation and must stay byte-stable while runs are live; this path is where production decode
evolves. Uses **one checkpoint** (the latest) by design — no ensembling in v1.

## Design, and why each piece is the way it is

```
volume (any scroll, any size, uint8/uint16)
  │  normalize.py    INTENSITY harmonization to the training distribution (bg-median / fg-p95
  │                  anchors — the exact map ingest_real.py applied to every real training crop;
  │                  fg defined mask-free via the EM air/papyrus threshold)
  │  blocks.py       tile into core+halo canvases (halo ≥ fov/2 + walk margin)
  │  rescale.py      GEOMETRIC harmonization, per block: measure the wrap period λ from the image
  │                  (structure-tensor normal + run-length, model-free, ~2 s), upsample by
  │                  s = λ*/λ (clamp [1, 2]) so the material appears at the training period,
  │                  decode there, map labels back. Fill budget ×s², commit floor ×s³ travel
  │                  with it. Fallback when the resample is skipped: λ-RELATIVE decode constants
  │                  (delta ≤ λ/2, spacing/floor/budget ratios) — geometry-only, weaker.
  │  decode.py       per-block: EM fibre mask → medial-ridge seeds (NO cap; grid-thinned at
  │                  constant PHYSICAL density) → batched viability screen → serial flood fills,
  │                  thickest-first, with the COMMITTED-NEIGHBOUR BARRIER (below)
  │  stitch.py       across blocks: mid-plane mutual-maximal-overlap matching only
  │                  (label voting is banned — measured −42% NERL), global union-find,
  │                  each block claims exactly its core
  └─ global int32 label volume (+ per-fill provenance: λ, s, mode, truncations)
```

### Why geometric normalization is in the production path

The network is a single-scale filter bank (18 stacked 3³ convs, ERF r50=7/r90=13 vox) with a
measured tuning curve peaking at the training wrap period λ*≈15.5. The Scroll-3/4 cores sit at
λ≈10.3 — **zero overlap with the training distribution** — and there the model assigns the
*neighbouring wrap* p̂=0.73 (the merge mechanism, measured). Controlled experiment on labelled
Scroll-1: compacting to core period raised merge 0.073→0.317; upsampling the same image back
dropped it to 0.024. On 2 Scroll-4 core cubes against the released winding meshes: recall
19.6→34.9% / 17.1→23.1%, zero merges introduced. λ is measured from the image alone, so this is
self-calibrating — no scroll metadata required.

### The committed-neighbour barrier (`fill.py`)

Carry-forward of already-traced objects, in **inhibitory** form. Before a fill starts, every voxel
committed to *another* object is written into the POM canvas at `logit(tgt_lo)` ("definitely not
me") and pinned there after every network write. Three effects, all one-directional:

1. the classifier **sees** prior commitments in its POM input channel (today it is blind to them —
   `claimed` only vetoes the movement queue);
2. the fill cannot recruit claimed voxels, so a leak across a blind contact into a traced wrap
   is stopped at the interface instead of being clipped at commit;
3. it is merge-asymmetric **by construction**: a barrier can only suppress growth, never cause it.

Cost: decode order matters more (first-come-first-served already made it matter at commit time;
the barrier moves the same information earlier, where the network can react to it). Order stays
thickest-first (descending EDT), the best available confidence proxy.

### Stitching rule (`stitch.py`)

On the mid-plane of each block-pair overlap, fragment `a` (block A) and `b` (block B) are united
iff each is the other's **maximal** overlap and the overlap clears a floor. Mutual-max is a
matching, not a vote: a merge cannot be created by accumulation of small overlaps, and a fragment
that disagrees with everything simply stays split (recoverable). Every block claims only its core,
so no voxel is ever written twice.

### Production caps

`fill_max_steps = 20000` (the training-eval path's 4000 is an inline-eval economy that TRUNCATES
sheets at 320³+ — measured: the "105 fragments / 14 wraps" collapse), seeds uncapped
(grid-thinned by spacing, never truncated by rank — an EDT-ordered cap deletes whole regions).

## Files

| file | contents |
|---|---|
| `config.py` | `TracerConfig` — every knob, with provenance comments |
| `rescale.py` | geometric normalization: per-block λ, resample to λ*, scaled scalars, λ-relative fallback |
| `normalize.py` | anchors from training slab; mask-free harmonization of raw volumes |
| `fill.py` | production flood fill: barrier + hoisted buffers + truncation stats |
| `decode.py` | per-block decode: seeds → screen → fills → labels + stats |
| `blocks.py` | core+halo tiling |
| `stitch.py` | mid-plane mutual-max matching, union-find, core assembly |
| `run.py` | whole-volume CLI (zarr/npy out, per-block progress, GPU selection) |
| `test_gt.py` | arms test on labelled cubes: eval-path baseline vs production vs +barrier |
| `test_stitch.py` | blocked-vs-monolithic equivalence on a 320³ cube |

## Ground rules

- Merges gate acceptance; NERL second. A NERL gain that adds a merge is a rejection.
- Training has priority on the box: `nice -n 19`, GPU 1, never touch `/root/surf/ffn_run5/`.
- One checkpoint (`ckpt_last.pt`) — by explicit decision, revisit ensembling only after this
  pipeline's numbers are in.
