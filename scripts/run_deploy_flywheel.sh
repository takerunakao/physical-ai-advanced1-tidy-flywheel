#!/usr/bin/env bash
# Deployment flywheel on the whole "片付けて" task: deploy the policy on fresh scenes with the VLM judge watching every
# step; on a judged failure the teacher takes over and its demo is kept (HG-DAgger interventions). Retrain on the
# base demos + all interventions so far, then score the full loop on held-out scenes (no interventions there).
#   scripts/run_deploy_flywheel.sh [rounds] [steps]
# Seeds: base demos 0-199 (seq scene), deployment scenes 3000+100K.. (+39), held-out tidy-all scenes 1000-1039.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=scripts
R=${1:-3}; STEPS=${2:-30000}; HF=${HF_USER:-takerunakao}; EVAL=1000-1039
ST=outputs/logs/series_d200-v.status; mkdir -p outputs/logs outputs/interventions
log() { echo "[$(date '+%F %T')] $*" >> "$ST"; }
fail() { log "FAILED: $*"; exit 1; }
ck() { [ "$1" = 0 ] && echo outputs/act_s200-v0/checkpoints/last/pretrained_model || echo outputs/act_d200-v$1/checkpoints/last/pretrained_model; }
tidy() {  # tidy <tag> <checkpoint> <seeds> [extra args...]
  local tag=$1 c=$2 seeds=$3; shift 3
  python scripts/tidy_all.py --checkpoint $c --snap-drop --seeds $seeds --out outputs/tidy_all/$tag.json "$@" \
    > outputs/logs/tidy_all_$tag.log 2>&1
}
summary() { python -c "import json; d=json.load(open('outputs/tidy_all/$1.json')); print({k: d[k] for k in ('n', 'all_tidied_rate', 'mean_tidied', 'interventions') if d.get(k) is not None})"; }

IV=()
for ((K=0; K<R; K++)); do
  N=$((K+1)); C=$(ck $K); A=$((3000+100*K))
  log "round $K: deploy $(basename $(dirname $(dirname $(dirname $C)))) on $A-$((A+39)) with interventions"
  tidy d200-v${K}_deploy_a $C $A-$((A+19)) --intervene outputs/interventions/d200-v$K &
  P1=$!
  tidy d200-v${K}_deploy_b $C $((A+20))-$((A+39)) --intervene outputs/interventions/d200-v$K &
  P2=$!
  if [ $K = 0 ]; then tidy d200-v0 $C $EVAL --video-dir outputs/tidy_all/d200-v0_videos; log "d200-v0 (= s200-v0) held-out: $(summary d200-v0)"; fi
  wait $P1 $P2
  log "deploy a: $(summary d200-v${K}_deploy_a)"; log "deploy b: $(summary d200-v${K}_deploy_b)"
  IV+=(--interventions outputs/interventions/d200-v$K)
  log "build d200-v$N (base 0-199 + ${IV[*]})"
  python scripts/build_dataset.py --scene seq --base-seeds 0-199 "${IV[@]}" --repo-id $HF/tidy-d200-v$N --root data/tidy-d200-v$N \
    --overwrite > outputs/logs/build_d200-v$N.log 2>&1 || fail "build d200-v$N"
  log "$(python -c "import json; d=json.load(open('data/tidy-d200-v$N/build_info.json')); e=[m for m in d['episodes'] if m['source'].startswith('intervention')]; print(len(e), 'interventions,', sum(m['kept'] for m in e), 'kept')")"
  log "train d200-v$N"
  SAVE_FREQ=$STEPS scripts/train_local.sh data/tidy-d200-v$N $HF/tidy-d200-v$N outputs/act_d200-v$N $STEPS mps > outputs/logs/train_d200-v$N.log 2>&1 \
    || { log "retry without workers"; rm -rf outputs/act_d200-v$N
         SAVE_FREQ=$STEPS NUM_WORKERS=0 scripts/train_local.sh data/tidy-d200-v$N $HF/tidy-d200-v$N outputs/act_d200-v$N $STEPS mps > outputs/logs/train_d200-v$N.log 2>&1; } \
    || fail "train d200-v$N"
  rm -rf outputs/act_d200-v$N/checkpoints/*/training_state
  log "held-out tidy-all d200-v$N"
  tidy d200-v$N $(ck $N) $EVAL --video-dir outputs/tidy_all/d200-v${N}_videos || fail "tidy d200-v$N"
  log "d200-v$N held-out: $(summary d200-v$N)"
done
log DONE
