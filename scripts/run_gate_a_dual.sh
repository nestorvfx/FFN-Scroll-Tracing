#!/bin/bash
# Fast Gate A: shard the val panel across many decode processes packed onto both
# GPUs, then merge + score once.
#
# Why packing helps: a single flood-fill decode is latency/CPU-bound (the serial
# BFS bookkeeping keeps GPU util at ~5-20%), so several decode processes share a
# GPU with near-linear throughput. With 64 CPU cores + 2x32GB this scales well.
# Each shard's cube decode is bit-identical to a serial run, so this is exact.
#
#   scripts/run_gate_a_dual.sh <ckpt> [n_cubes] [seed_sets]
#   PROCS=6 NGPU=2 scripts/run_gate_a_dual.sh <ckpt> 12 5
set -e
WORK=${WORK:-/root/surf/ffn_work}
CKPT=${1:?usage: run_gate_a_dual.sh <ckpt> [n_cubes] [seed_sets]}
CUBES=${2:-12}
SEEDS=${3:-}
PY=${PY:-/root/surf/venv/bin/python}
REPO=${REPO:-/root/surf/FFN}
PROCS=${PROCS:-6}          # total decode processes (>= NGPU); packs GPUs
NGPU=${NGPU:-2}
TAG=$(basename "$CKPT" .pt)
cd "$REPO"

ARGS="--gate A --work $WORK --ckpt $CKPT --cubes $CUBES"
[ -n "$SEEDS" ] && ARGS="$ARGS --seed-sets $SEEDS"

echo "[gateA-fast] $TAG cubes=$CUBES seeds=${SEEDS:-cfg} procs=$PROCS ngpu=$NGPU"
PIDS=(); OUTS=()
for ((i=0; i<PROCS; i++)); do
    G=$((i % NGPU))
    OUT=/tmp/gateA_${TAG}_s${i}.json
    OUTS+=("$OUT")
    CUDA_VISIBLE_DEVICES=$G $PY -m scripts.evaluate $ARGS \
        --n-shards "$PROCS" --shard "$i" --out "$OUT" \
        > /tmp/gateA_${TAG}_s${i}.log 2>&1 &
    PIDS+=($!)
done
FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || FAIL=1; done
[ $FAIL -ne 0 ] && echo "[gateA-fast] WARNING: a shard exited non-zero; check /tmp/gateA_${TAG}_s*.log"
$PY -m scripts.merge_gate_a "${OUTS[@]}" --ckpt "$CKPT" --out /tmp/gateA_${TAG}.json
