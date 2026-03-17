#!/bin/bash

# Single-GPU d8 baseline run.
# Goal: keep the setup conservative on 24GB cards while producing
# a useful checkpoint and a meaningful loss curve.
#
# Usage:
# bash runs/d8_single.sh
#
# Optional env vars:
# CUDA_VISIBLE_DEVICES=0
# NANOCHAT_BASE_DIR=$HOME/.cache/nanochat
# WANDB_RUN=d8-single

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

source .venv/bin/activate

WANDB_RUN="${WANDB_RUN:-d8-single}"

python -m scripts.base_train \
    --depth=8 \
    --model-tag=d8-single \
    --run="$WANDB_RUN" \
    --window-pattern=L \
    --max-seq-len=1024 \
    --device-batch-size=1 \
    --total-batch-size=16384 \
    --num-iterations=1500 \
    --eval-every=100 \
    --eval-tokens=131072 \
    --core-metric-every=-1 \
    --sample-every=500 \
    --save-every=500
