#!/bin/bash
set -e

DATASET_NAME=${DATASET_NAME:-CUHK-PEDES}
DATA_ROOT=${DATA_ROOT:-/root/shared-nvme/zixiangwang/yxyx/RDE_3090/datasets}
OUTPUT_DIR=${OUTPUT_DIR:-logs}
EXP_NAME=${EXP_NAME:-irra_light_clean_two_head}
IRRA_LIGHT_MODE=${IRRA_LIGHT_MODE:-split_pure}
NUM_EPOCH=${NUM_EPOCH:-60}
BATCH_SIZE=${BATCH_SIZE:-64}
SEED=${SEED:-1}
PYTHON_BIN=${PYTHON_BIN:-python}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
"${PYTHON_BIN}" train.py \
  --name "${EXP_NAME}" \
  --irra_light \
  --irra_light_mode "${IRRA_LIGHT_MODE}" \
  --irra_light_identity_loss sdm \
  --img_aug \
  --batch_size "${BATCH_SIZE}" \
  --sampler random \
  --dataset_name "${DATASET_NAME}" \
  --root_dir "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --num_epoch "${NUM_EPOCH}"
