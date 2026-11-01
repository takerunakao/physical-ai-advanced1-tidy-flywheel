#!/usr/bin/env bash
# One unattended flywheel iteration K -> K+1, plus the control arm. Run detached (nohup). Logs in outputs/logs/.
#   scripts/pipeline_flywheel.sh <K> [steps]
# Env: SERIES (name prefix, default "v" -> tidy-v0, act_v0), BASE (base teacher seeds, default 0-99),
#      SCENE (single | multi, default single),
#      DETECT (privileged | vlm, default privileged): which signal finds the deployment failures to correct --
#        the simulator's outcome, or the VLM success judge looking at the camera frames (no privileged state;
#        the control arm is skipped since it does not depend on failure detection),
#      PUSH=1 to also push each dataset to the Hub (datasets are seed-reproducible, so pushing is optional).
# Requires outputs/act_<SERIES><K>/checkpoints/last/pretrained_model and data/tidy-<SERIES><K>/build_info.json.
#
# Seed sets never overlap: base demos BASE, control-arm demos 200+, "deployment" rollouts that find failures
# 2000+100K .. 2099+100K (fresh each round), held-out evaluation 1000-1099 (only ever evaluated, never trained on).
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=scripts
K=${1:?iteration}; STEPS=${2:-20000}; N=$((K+1))
S=${SERIES:-v}; BASE=${BASE:-0-99}; DETECT=${DETECT:-privileged}
JUDGE=; [ "$DETECT" = vlm ] && JUDGE=--judge
HF=${HF_USER:-takerunakao}; EVAL_SEEDS=1000-1099; COLLECT_SEEDS=$((2000+100*K))-$((2099+100*K))
ST=outputs/logs/pipeline_${S}${N}.status; mkdir -p outputs/logs
log() { echo "[$(date '+%F %T')] $*" >> "$ST"; }
fail() { log "FAILED: $*"; exit 1; }
CK=outputs/act_${S}${K}/checkpoints/last/pretrained_model

# Train with one checkpoint per run (disk), retrying once without DataLoader workers (macOS shm timeout).
# The optimizer state (393 MB of each 590 MB checkpoint) only serves resuming, so it is dropped after training.
train() {  # train <tag>
  local tag=$1 out=outputs/act_$1
  SAVE_FREQ=$STEPS scripts/train_local.sh data/tidy-$tag $HF/tidy-$tag $out $STEPS mps > outputs/logs/train_$tag.log 2>&1 \
    || { log "train $tag failed with workers; retrying with NUM_WORKERS=0"; rm -rf "$out"
         SAVE_FREQ=$STEPS NUM_WORKERS=0 scripts/train_local.sh data/tidy-$tag $HF/tidy-$tag $out $STEPS mps > outputs/logs/train_$tag.log 2>&1; } \
    || return 1
  rm -rf "$out"/checkpoints/*/training_state
}
evaluate() {  # evaluate <tag>   (held-out seeds)
  python scripts/evaluate.py --scene ${SCENE:-single} --checkpoint outputs/act_$1/checkpoints/last/pretrained_model --seeds $EVAL_SEEDS --tag $1 \
    --out outputs/eval --success-videos 3 > outputs/logs/eval_$1.log 2>&1
}

[ -f outputs/eval/${S}${K}.json ] || { log "eval ${S}${K}"; evaluate ${S}${K} || fail "eval ${S}${K}"; }
log "collect ${S}${K} on deployment seeds $COLLECT_SEEDS"
python scripts/evaluate.py --scene ${SCENE:-single} --checkpoint "$CK" --seeds $COLLECT_SEEDS --tag ${S}${K}_collect --out outputs/eval \
  --success-videos 0 --no-failure-videos $JUDGE > outputs/logs/collect_${S}${K}.log 2>&1 || fail "collect ${S}${K}"
NFAIL=$(python -c "import json; d=json.load(open('outputs/eval/${S}${K}_collect.json')); print(d['n']-d['success'], d.get('vlm_judge', ''))")
log "${S}${K}: $NFAIL deployment failures (privileged count, VLM confusion) -> corrections by $DETECT"

# corrections arm: base + recovery demos for the failed deployment seeds (previous rounds regenerated from build_info)
EXTRA_ARGS=$(python - <<PY
import json
b=json.load(open('data/tidy-${S}${K}/build_info.json'))['argv']
out=[]; i=0
while i < len(b):
    if b[i]=='--corrections': out += ['--corrections', b[i+1]]; i+=2
    elif b[i]=='--eval': ev=b[i+1]; i+=2
    elif b[i]=='--checkpoint': ck=b[i+1]; i+=2
    else: i+=1
if '--eval' in b: out += ['--corrections', f'{ev}={ck}']
print(' '.join(out))
PY
)
log "build ${S}${N} (extra: $EXTRA_ARGS)"
python scripts/build_dataset.py --scene ${SCENE:-single} --base-seeds $BASE $EXTRA_ARGS --eval outputs/eval/${S}${K}_collect.json --checkpoint "$CK" \
  --failure-source $DETECT --mode recovery --repo-id $HF/tidy-${S}${N} --root data/tidy-${S}${N} --overwrite ${PUSH:+--push} > outputs/logs/build_${S}${N}.log 2>&1 || fail "build ${S}${N}"

if [ "$DETECT" = vlm ]; then
  log "train ${S}${N}"; train ${S}${N} || fail "train ${S}${N}"
  log "eval ${S}${N}"; evaluate ${S}${N} || fail "eval ${S}${N}"
  log "DONE"; exit 0
fi

# control arm: base + the same number of fresh teacher demos (seeds 200+)
NKEPT=$(python -c "import json; m=json.load(open('data/tidy-${S}${N}/build_info.json'))['episodes']; print(sum(1 for e in m if e['kept'] and e['source']!='teacher'))")
CSTART=200; CEND=$((CSTART + NKEPT - 1))  # cumulative: the corrections arm regenerates all rounds
log "build ${S}${N}c with $NKEPT extra teacher demos (seeds $CSTART-$CEND)"
python scripts/build_dataset.py --scene ${SCENE:-single} --base-seeds $BASE --extra-seeds ${CSTART}-${CEND} --repo-id $HF/tidy-${S}${N}c --root data/tidy-${S}${N}c --overwrite ${PUSH:+--push} > outputs/logs/build_${S}${N}c.log 2>&1 || fail "build ${S}${N}c"

log "train ${S}${N}"; train ${S}${N} || fail "train ${S}${N}"
log "eval ${S}${N}"; evaluate ${S}${N} || fail "eval ${S}${N}"
log "train ${S}${N}c"; train ${S}${N}c || fail "train ${S}${N}c"
log "eval ${S}${N}c"; evaluate ${S}${N}c || fail "eval ${S}${N}c"
log "DONE"
