#!/usr/bin/env bash
# Detached pipeline: record dataset v0 -> push -> train ACT v0 (20k steps, MPS). Logs in outputs/logs/.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=scripts
mkdir -p outputs/logs
echo "[$(date '+%F %T')] build v0" > outputs/logs/pipeline_v0.status
python scripts/build_dataset.py --base-seeds 0-99 --repo-id takerunakao/tidy-v0 --root data/tidy-v0 --overwrite --push > outputs/logs/build_v0.log 2>&1 || { echo "[$(date '+%F %T')] BUILD FAILED" >> outputs/logs/pipeline_v0.status; exit 1; }
echo "[$(date '+%F %T')] train v0" >> outputs/logs/pipeline_v0.status
scripts/train_local.sh data/tidy-v0 takerunakao/tidy-v0 outputs/act_v0 20000 mps > outputs/logs/train_v0.log 2>&1 || { echo "[$(date '+%F %T')] TRAIN FAILED" >> outputs/logs/pipeline_v0.status; exit 1; }
echo "[$(date '+%F %T')] DONE" >> outputs/logs/pipeline_v0.status
