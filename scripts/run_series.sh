#!/usr/bin/env bash
# Unattended flywheel series: build + train + evaluate v0, gate on its success rate, then ROUNDS iterations.
#   SERIES=b25-v BASE=0-24 nohup caffeinate -i scripts/run_series.sh [rounds] [steps] &
# The gate keeps the flywheel inside the range where improvement is measurable (PLAN §4.4); out of range -> stop.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=scripts
export SERIES=${SERIES:?SERIES} BASE=${BASE:?BASE}
ROUNDS=${1:-3}; STEPS=${2:-20000}; GATE_LO=${GATE_LO:-0.2}; GATE_HI=${GATE_HI:-0.75}
S=$SERIES; HF=${HF_USER:-takerunakao}; ST=outputs/logs/series_${S}.status; mkdir -p outputs/logs
log() { echo "[$(date '+%F %T')] $*" >> "$ST"; }
fail() { log "FAILED: $*"; exit 1; }

log "build ${S}0 (base $BASE)"
python scripts/build_dataset.py --scene ${SCENE:-single} --base-seeds $BASE --repo-id $HF/tidy-${S}0 --root data/tidy-${S}0 --overwrite ${PUSH:+--push} \
  > outputs/logs/build_${S}0.log 2>&1 || fail "build ${S}0"
log "train ${S}0"
SAVE_FREQ=$STEPS scripts/train_local.sh data/tidy-${S}0 $HF/tidy-${S}0 outputs/act_${S}0 $STEPS mps > outputs/logs/train_${S}0.log 2>&1 \
  || { log "retry ${S}0 with NUM_WORKERS=0"; rm -rf outputs/act_${S}0
       SAVE_FREQ=$STEPS NUM_WORKERS=0 scripts/train_local.sh data/tidy-${S}0 $HF/tidy-${S}0 outputs/act_${S}0 $STEPS mps > outputs/logs/train_${S}0.log 2>&1; } \
  || fail "train ${S}0"
rm -rf outputs/act_${S}0/checkpoints/*/training_state  # optimizer state only serves resuming (393 MB)
log "eval ${S}0"
python scripts/evaluate.py --scene ${SCENE:-single} --checkpoint outputs/act_${S}0/checkpoints/last/pretrained_model --seeds 1000-1099 --tag ${S}0 \
  --out outputs/eval --success-videos 3 > outputs/logs/eval_${S}0.log 2>&1 || fail "eval ${S}0"
P=$(python -c "import json; print(json.load(open('outputs/eval/${S}0.json'))['success_rate'])")
python -c "import sys; sys.exit(0 if $GATE_LO <= $P <= $GATE_HI else 1)" || fail "GATE: ${S}0 success $P outside [$GATE_LO, $GATE_HI]"
log "GATE ok: ${S}0 success $P"

for ((K=0; K<ROUNDS; K++)); do
  log "round $K -> $((K+1))"
  scripts/pipeline_flywheel.sh $K $STEPS || fail "round $K (see pipeline_${S}$((K+1)).status)"
done
log "DONE"
