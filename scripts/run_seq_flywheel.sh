#!/usr/bin/env bash
# VLM-detected flywheel on the mid-tidy-up scene, scored after every round on the full "片付けて" loop.
#   scripts/run_seq_flywheel.sh [rounds] [steps]      (needs scripts/run_seq.sh s200-v0 ... to have finished)
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=scripts SERIES=s200-v SCENE=seq DETECT=vlm BASE=0-199
R=${1:-3}; STEPS=${2:-30000}
ST=outputs/logs/series_s200-v.status
log() { echo "[$(date '+%F %T')] $*" >> "$ST"; }
until grep -q "DONE\|FAILED" outputs/logs/seq_s200-v0.status 2>/dev/null; do sleep 60; done
grep -q FAILED outputs/logs/seq_s200-v0.status && { log "FAILED: s200-v0"; exit 1; }
for ((K=0; K<R; K++)); do
  N=$((K+1))
  log "round $K -> $N"
  scripts/pipeline_flywheel.sh $K $STEPS || { log "FAILED: round $K (see pipeline_s200-v$N.status)"; exit 1; }
  log "s200-v$N: $(python -c "import json; d=json.load(open('outputs/eval/s200-v$N.json')); print(d['success_rate'], d['failure_modes'])")"
  python scripts/tidy_all.py --checkpoint outputs/act_s200-v$N/checkpoints/last/pretrained_model --snap-drop --seeds 1000-1019 \
    --video-dir outputs/tidy_all/s200-v${N}_videos --out outputs/tidy_all/s200-v$N.json > outputs/logs/tidy_all_s200-v$N.log 2>&1 \
    || log "tidy_all s200-v$N failed"
  log "tidy_all s200-v$N: $(tail -1 outputs/logs/tidy_all_s200-v$N.log)"
done
log DONE
