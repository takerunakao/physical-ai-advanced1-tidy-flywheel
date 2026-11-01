#!/usr/bin/env bash
# Policy for whole-table tidying: teacher demos in mid-tidy-up states (scene "seq"), then evaluate it per instruction
# and on the full "片付けて" loop. Run detached (nohup). Status in outputs/logs/seq_<tag>.status.
#   scripts/run_seq.sh <tag> <base seeds> [steps]      e.g. scripts/run_seq.sh s200-v0 0-199 30000
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=scripts
T=${1:?tag}; BASE=${2:?base seeds}; STEPS=${3:-30000}; HF=${HF_USER:-takerunakao}
ST=outputs/logs/seq_$T.status; mkdir -p outputs/logs
log() { echo "[$(date '+%F %T')] $*" >> "$ST"; }
fail() { log "FAILED: $*"; exit 1; }
CK=outputs/act_$T/checkpoints/last/pretrained_model

log "build $T (seq scene, base $BASE)"
python scripts/build_dataset.py --scene seq --base-seeds $BASE --repo-id $HF/tidy-$T --root data/tidy-$T --overwrite \
  > outputs/logs/build_$T.log 2>&1 || fail "build $T"
log "train $T ($STEPS steps)"
SAVE_FREQ=$STEPS scripts/train_local.sh data/tidy-$T $HF/tidy-$T outputs/act_$T $STEPS mps > outputs/logs/train_$T.log 2>&1 \
  || { log "retry without workers"; rm -rf outputs/act_$T
       SAVE_FREQ=$STEPS NUM_WORKERS=0 scripts/train_local.sh data/tidy-$T $HF/tidy-$T outputs/act_$T $STEPS mps > outputs/logs/train_$T.log 2>&1; } \
  || fail "train $T"
rm -rf outputs/act_$T/checkpoints/*/training_state
for sc in seq multi; do
  log "eval $T on $sc"
  python scripts/evaluate.py --scene $sc --checkpoint $CK --seeds 1000-1099 --tag ${T}_$sc --out outputs/eval --success-videos 3 \
    > outputs/logs/eval_${T}_$sc.log 2>&1 || fail "eval $T $sc"
  log "$(python -c "import json; d=json.load(open('outputs/eval/${T}_$sc.json')); print('$sc', d['success_rate'], d['failure_modes'])")"
done
log "tidy_all $T"
python scripts/tidy_all.py --checkpoint $CK --snap-drop --seeds 1000-1019 --video-dir outputs/tidy_all/${T}_videos \
  --out outputs/tidy_all/$T.json > outputs/logs/tidy_all_$T.log 2>&1 || fail "tidy_all $T"
log "$(tail -1 outputs/logs/tidy_all_$T.log)"
log DONE
