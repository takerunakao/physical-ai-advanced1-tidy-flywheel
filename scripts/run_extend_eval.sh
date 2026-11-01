#!/usr/bin/env bash
# After the deployment flywheel: score every round on 60 more held-out scenes (1040-1099), so each round is judged on
# 100 scenes in all (1000-1099) and the paired v0-vs-vN comparison has enough discordant scenes to test.
#   scripts/run_extend_eval.sh [rounds]
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=scripts
R=${1:-3}; ST=outputs/logs/series_d200-v.status
until grep -q "DONE\|FAILED" $ST; do sleep 300; done
log() { echo "[$(date '+%F %T')] $*" >> "$ST"; }
ck() { [ "$1" = 0 ] && echo outputs/act_s200-v0/checkpoints/last/pretrained_model || echo outputs/act_d200-v$1/checkpoints/last/pretrained_model; }
for ((K=0; K<=R; K++)); do
  [ -d "$(ck $K)" ] || continue
  for half in a b; do
    [ $half = a ] && S=1040-1069 || S=1070-1099
    python scripts/tidy_all.py --checkpoint $(ck $K) --snap-drop --seeds $S --out outputs/tidy_all/d200-v${K}_ext_$half.json \
      > outputs/logs/tidy_all_d200-v${K}_ext_$half.log 2>&1 &
  done
  wait
  log "d200-v$K extended (1040-1099): $(python -c "import json; r=[e for h in 'ab' for e in json.load(open(f'outputs/tidy_all/d200-v${K}_ext_{h}.json'))['episodes']]; print(sum(e['all_tidied'] for e in r), '/', len(r))")"
done
log "EXTENDED DONE"
