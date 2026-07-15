#!/bin/bash
set -euo pipefail

DATASET_NAME=${DATASET_NAME:-TAG-PEDES}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/datasets}
OUTPUT_DIR=${OUTPUT_DIR:-logs_4090/HIRE}
PYTHON_BIN=${PYTHON_BIN:-python}
NUM_EPOCH=${NUM_EPOCH:-60}
BATCH_SIZE=${BATCH_SIZE:-64}
SEED=${SEED:-1}
NUM_WORKERS=${NUM_WORKERS:-8}
SUPPORT_SIZE=${SUPPORT_SIZE:-3}
PRETRAIN_CHOICE=${PRETRAIN_CHOICE:-ViT-B/16}
EXP_NAME=${EXP_NAME:-hire_main_${DATASET_NAME}_60e_seed${SEED}}

if [ "${PRETRAIN_CHOICE}" != "ViT-B/16" ]; then
  echo "The delivered main HIRE version requires PRETRAIN_CHOICE=ViT-B/16"
  exit 1
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
"${PYTHON_BIN}" train.py \
  --name "${EXP_NAME}" \
  --hire \
  --pretrain_choice "${PRETRAIN_CHOICE}" \
  --dataset_name "${DATASET_NAME}" \
  --root_dir "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --test_batch_size 512 \
  --num_workers "${NUM_WORKERS}" \
  --num_epoch "${NUM_EPOCH}" \
  --seed "${SEED}" \
  --hire_support_size "${SUPPORT_SIZE}" \
  --sampler random \
  --val_dataset test
