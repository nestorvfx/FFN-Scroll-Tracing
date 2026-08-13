#!/bin/bash
# =============================================================================
# FFN bootstrap: BARE instance -> ready-to-train, one command.
#
#   HF_TOKEN=hf_...  bash scripts/bootstrap.sh
#
# Prereq: this FFN directory is on the box (scp/git). Everything else — venv,
# deps, all data (synth corpus + real crops + eval slab + preprocessed index) —
# is pulled from HF nestorvfx/ScrollData and verified. Idempotent: re-running
# skips whatever is already in place. Ends by printing the exact train command.
#
# Data layout produced (canonical paths, override via env):
#   $CORPUS  = /root/surf/data/synthfuse_corpus   400 synth + 160 real crops
#   $SLAB    = /root/data/slab                    held-out eval slab (Gate B ONLY)
#   $CORES   = /root/data/evalcubes               Scroll-3/4 core cubes, UNLABELLED
#   $WORK    = /root/surf/ffn_run5                preprocessed index + splits + checkpoints
#
# $CORES is not optional for a current run. The labelled val panel is the EASY regime --
# measured frac_1step 0.067 on it vs 0.556 on core cubes for the SAME weights -- so training
# steered on the panel alone cannot see a model that has stopped tracing. `--core-dirs` adds a
# label-free probe on the target regime. See FINDINGS.md 3.6.
#
# To CONTINUE an existing run on a new box, set RESUME=1 and the checkpoints are pulled too:
#   HF_TOKEN=hf_...  RESUME=1  bash scripts/bootstrap.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."                                    # -> FFN root

HF_TOKEN=${HF_TOKEN:-}
REPO=${REPO:-nestorvfx/ScrollData}
CORPUS=${CORPUS:-/root/surf/data/synthfuse_corpus}
SLAB=${SLAB:-/root/data/slab}
CORES=${CORES:-/root/data/evalcubes}
WORK=${WORK:-/root/surf/ffn_run5}
VENV=${VENV:-/root/surf/venv}
RESUME=${RESUME:-0}
PY="$VENV/bin/python"

echo "== [1/5] python env =="
if [ ! -x "$PY" ]; then
    python3 -m venv "$VENV"
    "$PY" -m pip install -q --upgrade pip
fi
"$PY" -c "import torch" 2>/dev/null || "$PY" -m pip install -q torch --index-url https://download.pytorch.org/whl/cu128
# connected-components-3d (cc3d) is REQUIRED by metrics.erl: NERL is computed as
# volumetric 26-connected run length (scikit-image's 3-D skeletonize is unusable on
# thin sheets -- issue #3757 -- so we do NOT rely on it for the metric or seeding).
"$PY" -m pip install -q numpy scipy scikit-image tifffile pynrrd "huggingface_hub[hf_transfer]" zstandard pytest connected-components-3d
command -v zstd >/dev/null || (apt-get update -qq && apt-get install -y -qq zstd)
"$PY" -m scripts.setup_instance --corpus "$CORPUS" --slab "$SLAB" --work "$WORK" || true

echo "== [2/5] pull data from HF ($REPO) =="
DL=/root/hf_dl && mkdir -p "$DL" "$CORPUS" "$SLAB" "$WORK" /root/data
hfget() {  # hfget <path_in_repo> — resumable, hf_transfer-accelerated
    HF_HUB_ENABLE_HF_TRANSFER=1 "$PY" - "$1" <<PYEOF
import sys, os
from huggingface_hub import hf_hub_download
token = "$HF_TOKEN" if "$HF_TOKEN" else None
p = hf_hub_download("$REPO", sys.argv[1], repo_type="dataset",
                    token=token, local_dir="$DL")
print(p)
PYEOF
}
# synth corpus (3.5 GB) — skip if the 400 synth cubes are already there
if [ "$(ls "$CORPUS/imagesTr" 2>/dev/null | grep -c '^synthfuse')" -lt 400 ]; then
    F=$(hfget 04_resources/corpora/synthfuse_corpus_v3.tar.zst | tail -1)
    TMP=$(mktemp -d)
    zstd -d -c "$F" | tar -xf - -C "$TMP"
    # archive may or may not have a single top-level dir -> find the imagesTr parent
    SRC=$(dirname "$(find "$TMP" -maxdepth 3 -type d -name imagesTr | head -1)")
    [ -d "$SRC/imagesTr" ] || { echo "FATAL: imagesTr not found in corpus archive"; exit 1; }
    cp -al "$SRC"/. "$CORPUS"/ 2>/dev/null || cp -a "$SRC"/. "$CORPUS"/
    rm -rf "$TMP"
else echo "  synth corpus present -> skip"; fi
# real crops (689 MB) -> overlay into the same corpus dir
if [ ! -f "$CORPUS/real_meta.json" ]; then
    F=$(hfget 05_ffn/ffn_real_crops.tar.zst | tail -1)
    zstd -d -c "$F" | tar -xf - -C "$CORPUS"
else echo "  real crops present -> skip"; fi
# eval slab (64 MB) -> /root/data/slab
if [ ! -f "$SLAB/volume.nrrd" ]; then
    F=$(hfget 05_ffn/ffn_slab_eval.tar.zst | tail -1)
    zstd -d -c "$F" | tar -xf - -C "$(dirname "$SLAB")"
else echo "  slab present -> skip"; fi
# preprocessed index (38 MB) -> $WORK (exact split/index of the 2026-07 run)
if [ ! -f "$WORK/train_index.npz" ]; then
    F=$(hfget 05_ffn/ffn_work_index.tar.zst | tail -1)
    zstd -d -c "$F" | tar -xf - -C "$WORK"
else echo "  work index present -> skip"; fi
# UNLABELLED Scroll-3/4 core cubes -> the label-free movement/core probe (--core-dirs).
# 100 cubes each. Without these the run is scored only on the easy labelled panel.
mkdir -p "$CORES"
for S in 3 4; do
    D="$CORES/Scroll${S}EvalCubes"
    if [ ! -d "$D/imagesTr" ]; then
        F=$(hfget "Scroll${S}EvalCubes.tar.zst" | tail -1)
        zstd -d -c "$F" | tar -xf - -C "$CORES"
        # archive top-level dir name varies -> normalise to Scroll<N>EvalCubes
        [ -d "$D/imagesTr" ] || { P=$(dirname "$(find "$CORES" -maxdepth 3 -type d -name imagesTr | grep -iv scroll$((7-S)) | head -1)"); [ "$P" = "$D" ] || mv "$P" "$D"; }
    else echo "  Scroll${S} core cubes present -> skip"; fi
done
# checkpoints, only when continuing an existing run
if [ "$RESUME" = 1 ]; then
    if ls "$WORK"/ckpt_*.pt >/dev/null 2>&1; then echo "  checkpoints present -> skip"
    else
        # run name is the work-dir basename, matching what scripts/ckpt_sync.py push writes.
        RUN=$(basename "$WORK")
        F=$(hfget "05_ffn/$RUN/${RUN}_ckpts.tar.zst" | tail -1)
        zstd -d -c "$(readlink -f "$F")" | tar -xf - -C "$WORK"   # readlink: zstd won't follow symlinks
    fi
fi

echo "== [3/5] verify =="
NS=$(ls "$CORPUS/imagesTr" | grep -c '^synthfuse' || true)
NR=$(ls "$CORPUS/imagesTr" | grep -c '^real_' || true)
[ "$NS" -eq 400 ] || { echo "FATAL: expected 400 synth cubes, got $NS"; exit 1; }
[ "$NR" -eq 160 ] || { echo "FATAL: expected 160 real crops, got $NR"; exit 1; }
[ -f "$SLAB/truth.nrrd" ] || { echo "FATAL: slab truth missing"; exit 1; }
[ -f "$WORK/splits.json" ] || { echo "FATAL: work index missing"; exit 1; }
for S in 3 4; do
    N=$(ls "$CORES/Scroll${S}EvalCubes/imagesTr" 2>/dev/null | wc -l)
    [ "$N" -gt 0 ] || { echo "FATAL: Scroll${S} core cubes missing"; exit 1; }
done
echo "  corpus: 400 synth + 160 real OK | slab OK | index OK | core cubes OK"

echo "== [4/5] unit tests (non-fatal; report only) =="
"$PY" -m pytest tests/ -q || echo "WARN: some tests failed (see above) -- setup continues"

echo "== [5/5] ready — run: =="
NGPU=$("$PY" - <<PYEOF
import torch; print(torch.cuda.device_count())
PYEOF
)
LAUNCH="nohup $VENV/bin/torchrun --nproc_per_node=$NGPU -m scripts.train"
[ "$NGPU" -gt 1 ] || LAUNCH="nohup $PY -m scripts.train"
CD="$CORES/Scroll3EvalCubes,$CORES/Scroll4EvalCubes"
echo "  # (re-preprocess only if you changed data:  $PY -m scripts.preprocess --work $WORK --corpus $CORPUS)"
if [ -f "$WORK/config.json" ] && grep -q '"move_gate": *"facemax"' "$WORK/config.json"; then
    # config.json already carries the tracer profile, so --tracer/--core-dirs are baked in and a
    # resume cannot silently revert to the run-2 settings.
    echo "  # FRESH:  $LAUNCH --work $WORK > $WORK/train.log 2>&1 &"
    echo "  # RESUME: $LAUNCH --work $WORK --resume >> $WORK/train.log 2>&1 &"
else
    echo "  # FRESH:  $LAUNCH --work $WORK --tracer --core-dirs $CD > $WORK/train.log 2>&1 &"
    echo "  # RESUME: $LAUNCH --work $WORK --tracer --core-dirs $CD --resume >> $WORK/train.log 2>&1 &"
    echo "  #   (--tracer MUST be repeated on resume until the profile is written into config.json;"
    echo "  #    scripts/ckpt_sync.py push does that for you when it uploads.)"
fi
echo "  # track: progress.json / train.log in $WORK  (or track_ffn.ps1 from Windows)"
echo "  # WATCH: core_score (maximize) and core_1step (minimize) -- see FINDINGS.md 3.6"
echo "BOOTSTRAP_DONE"
