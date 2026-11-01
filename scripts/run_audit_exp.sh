#!/usr/bin/env bash
# Data-audit experiment (multi scene): clean base vs 30% mislabelled vs mislabelled-then-audited.
#   scripts/run_audit_exp.sh [steps]      (detached; status in outputs/logs/audit_exp.status)
# The clean arm is the m3-v0 policy of the m3-v series (same base seeds 0-99).
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=scripts
STEPS=${1:-20000}; HF=${HF_USER:-takerunakao}; BASE=${BASE:-0-99}; FRAC=${FRAC:-0.3}; P=${TAG:-m3-p30}
ST=outputs/logs/audit_exp.status; mkdir -p outputs/logs outputs/audit
log() { echo "[$(date '+%F %T')] $*" >> "$ST"; }
fail() { log "FAILED: $*"; exit 1; }
wait_gpu() { while [ "$(pgrep -f lerobot-train | wc -l)" -ge "${MAX_TRAIN:-3}" ]; do sleep 120; done; }
train() {  # train <tag>
  local tag=$1 out=outputs/act_$1
  wait_gpu
  SAVE_FREQ=$STEPS scripts/train_local.sh data/tidy-$tag $HF/tidy-$tag $out $STEPS mps > outputs/logs/train_$tag.log 2>&1 \
    || { log "train $tag failed with workers; retrying with NUM_WORKERS=0"; rm -rf "$out"
         SAVE_FREQ=$STEPS NUM_WORKERS=0 scripts/train_local.sh data/tidy-$tag $HF/tidy-$tag $out $STEPS mps > outputs/logs/train_$tag.log 2>&1; } \
    || return 1
  rm -rf "$out"/checkpoints/*/training_state
}
evaluate() {
  python scripts/evaluate.py --scene multi --checkpoint outputs/act_$1/checkpoints/last/pretrained_model --seeds 1000-1099 --tag $1 \
    --out outputs/eval --success-videos 3 > outputs/logs/eval_$1.log 2>&1
}

log "build $P (poison $FRAC of $BASE)"
python scripts/build_dataset.py --scene multi --base-seeds $BASE --poison-frac $FRAC --repo-id $HF/tidy-$P --root data/tidy-$P \
  --overwrite ${PUSH:+--push} > outputs/logs/build_$P.log 2>&1 || fail "build $P"
log "audit $P"
python scripts/audit_dataset.py --repo-id $HF/tidy-$P --root data/tidy-$P --out outputs/audit/$P.json > outputs/logs/audit_$P.log 2>&1 || fail "audit $P"
EXCL=$(grep '^EXCLUDE' outputs/logs/audit_$P.log | cut -d' ' -f2)
log "audit: $(python -c "import json; d=json.load(open('outputs/audit/$P.json')); print({k: d[k] for k in ('n','flagged','poisoned','precision','recall')})")"
log "build ${P}a (excluding ${EXCL:-nothing})"
python scripts/build_dataset.py --scene multi --base-seeds $BASE --poison-frac $FRAC --exclude-seeds "$EXCL" --repo-id $HF/tidy-${P}a \
  --root data/tidy-${P}a --overwrite ${PUSH:+--push} > outputs/logs/build_${P}a.log 2>&1 || fail "build ${P}a"
for tag in $P ${P}a; do
  log "train $tag"; train $tag || fail "train $tag"
  log "eval $tag"; evaluate $tag || fail "eval $tag"
  log "$tag: $(python -c "import json; d=json.load(open('outputs/eval/$tag.json')); print(d['success_rate'], d['failure_modes'])")"
done
log "DONE"
