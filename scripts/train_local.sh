#!/usr/bin/env bash
# Train ACT locally (Apple Silicon: --policy.device=mps; CPU fallback). Same hyper-parameters as the Colab script.
#   scripts/train_local.sh data/tidy-v0 takerunakao/tidy-v0 outputs/act_v0 [steps] [device]
# Env: SAVE_FREQ (default 5000; checkpoints are 590 MB each), NUM_WORKERS (default 2; 0 avoids the macOS
# shared-memory timeout that killed train_v1c).
set -euo pipefail
ROOT=${1:?dataset root}; REPO=${2:?dataset repo id}; OUT=${3:?output dir}; STEPS=${4:-20000}; DEVICE=${5:-mps}
exec lerobot-train \
  --dataset.repo_id="$REPO" --dataset.root="$ROOT" \
  --policy.type=act --policy.chunk_size=20 --policy.n_action_steps=10 \
  --policy.device="$DEVICE" --policy.push_to_hub=false \
  --output_dir="$OUT" --job_name="$(basename "$OUT")" \
  --steps="$STEPS" --batch_size=32 --seed=1000 --save_freq="${SAVE_FREQ:-5000}" --log_freq=100 \
  --wandb.enable=false --num_workers="${NUM_WORKERS:-2}"
