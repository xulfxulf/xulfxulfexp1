#!/bin/bash
set -euo pipefail

DATASET_NAME=${DATASET_NAME:-TAG-PEDES}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/datasets}
OUTPUT_DIR=${OUTPUT_DIR:-logs_4090/HIRE_v2_identity_state}
PYTHON_BIN=${PYTHON_BIN:-python}
NUM_EPOCH=${NUM_EPOCH:-60}
BATCH_SIZE=${BATCH_SIZE:-64}
SEED=${SEED:-1}
NUM_WORKERS=${NUM_WORKERS:-8}
SUPPORT_SIZE=${SUPPORT_SIZE:-3}
AUX_WEIGHT=${AUX_WEIGHT:-0.1}
STATE_TOPK=${STATE_TOPK:-50}
STATE_IMAGE_TOKENS=${STATE_IMAGE_TOKENS:-16}
STATE_TEXT_TOKENS=${STATE_TEXT_TOKENS:-8}
PRETRAIN_CHOICE=${PRETRAIN_CHOICE:-ViT-B/16}
EXP_NAME=${EXP_NAME:-hire_v2_identity_state_${DATASET_NAME}_60e_seed${SEED}}

if [ "${PRETRAIN_CHOICE}" != "ViT-B/16" ]; then
  echo "HIRE-v2 v16.3.0 requires PRETRAIN_CHOICE=ViT-B/16"
  exit 1
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
"${PYTHON_BIN}" train.py \
  --name "${EXP_NAME}" \
  --hire_v2 \
  --hire_v2_mode identity_state \
  --hire_v2_support_size "${SUPPORT_SIZE}" \
  --hire_v2_aux_weight "${AUX_WEIGHT}" \
  --hire_v2_state_topk "${STATE_TOPK}" \
  --hire_v2_state_image_tokens "${STATE_IMAGE_TOKENS}" \
  --hire_v2_state_text_tokens "${STATE_TEXT_TOKENS}" \
  --pretrain_choice "${PRETRAIN_CHOICE}" \
  --dataset_name "${DATASET_NAME}" \
  --root_dir "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --test_batch_size 512 \
  --num_workers "${NUM_WORKERS}" \
  --num_epoch "${NUM_EPOCH}" \
  --seed "${SEED}" \
  --sampler random \
  --val_dataset test
