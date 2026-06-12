#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

DATASET_NAME=${DATASET_NAME:-CUHK-PEDES}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/datasets}
OUTPUT_DIR=${OUTPUT_DIR:-logs_4090}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/irra190/bin/python}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NUM_EPOCH=${NUM_EPOCH:-2}
BATCH_SIZE=${BATCH_SIZE:-64}
SEED=${SEED:-1}
IMG_AUG=${IMG_AUG:-0}
SMOKE_MODES=${SMOKE_MODES:-"single_pure single_proj_pure split_pure"}

export DATASET_NAME DATA_ROOT OUTPUT_DIR PYTHON_BIN CUDA_VISIBLE_DEVICES
export NUM_EPOCH BATCH_SIZE SEED IMG_AUG

for MODE in ${SMOKE_MODES}; do
  export IRRA_LIGHT_MODE="${MODE}"
  export EXP_NAME="smoke_${DATASET_NAME}_${MODE}_e${NUM_EPOCH}_aug${IMG_AUG}_seed${SEED}"
  bash run_irra_light.sh
done
