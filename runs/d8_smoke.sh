#!/bin/bash

# Single-GPU d8 smoke test.
# Goal: verify the training path is healthy and loss decreases.
#
# Usage:
# bash runs/d8_smoke.sh
#
# Optional env vars:
# CUDA_VISIBLE_DEVICES=0
# NANOCHAT_BASE_DIR=$HOME/.cache/nanochat
# WANDB_RUN=dummy

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

source .venv/bin/activate

WANDB_RUN="${WANDB_RUN:-dummy}"

python -m scripts.base_train \
    --depth=8 \
    --model-tag=d8-smoke \
    --run="$WANDB_RUN" \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=1 \
    --total-batch-size=4096 \
    --num-iterations=60 \
    --eval-every=20 \
    --eval-tokens=8192 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=-1
