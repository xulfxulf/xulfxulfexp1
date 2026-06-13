#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export DATASET_NAME=${DATASET_NAME:-TAG-PEDES}
export DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/datasets}
export OUTPUT_DIR=${OUTPUT_DIR:-logs_4090}
export PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/irra190/bin/python}
export NUM_EPOCH=${NUM_EPOCH:-60}
export BATCH_SIZE=${BATCH_SIZE:-64}
export SEED=${SEED:-1}
export IMG_AUG=${IMG_AUG:-0}
export IRRA_LIGHT_MODE=single_pure
export PRETRAIN_CHOICE=${PRETRAIN_CHOICE:-/root/autodl-tmp/IRRA_light_baseline/pretrained/superclip_irra_vitb16.pt}
export EXP_NAME=${EXP_NAME:-irra_light_single_pure_superclip_4090_tagpedes_60e}

bash run_irra_light.sh
