#!/bin/bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/datasets}
OUTPUT_DIR=${OUTPUT_DIR:-/root/autodl-tmp/HIRE_v2_phrase_route_v1670_logs}
PYTHON_BIN=${PYTHON_BIN:-python}
NUM_EPOCH=${NUM_EPOCH:-60}
BATCH_SIZE=${BATCH_SIZE:-64}
SEED=${SEED:-1}
NUM_WORKERS=${NUM_WORKERS:-8}
SUPPORT_SIZE=${SUPPORT_SIZE:-3}
AUX_WEIGHT=${AUX_WEIGHT:-0.1}
TRAIN_LABELS=${TRAIN_LABELS:?set TRAIN_LABELS to v16.7.0 comparative phrase labels JSONL}
VAL_SPANS=${VAL_SPANS:-${TEST_SPANS:-}}
TEST_SPANS=${TEST_SPANS:?set TEST_SPANS to TAG test phrase spans JSONL}
EXP_NAME=${EXP_NAME:-hire_v2_phrase_route_v1670_tagpedes_60e_seed${SEED}}

if [ -z "${VAL_SPANS}" ]; then
  VAL_SPANS="${TEST_SPANS}"
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
"${PYTHON_BIN}" train.py \
  --name "${EXP_NAME}" \
  --hire_v2 \
  --hire_v2_mode identity_phrase_route_cmp \
  --hire_v2_support_size "${SUPPORT_SIZE}" \
  --hire_v2_aux_weight "${AUX_WEIGHT}" \
  --hire_v2_phrase_train_labels "${TRAIN_LABELS}" \
  --hire_v2_phrase_val_spans "${VAL_SPANS}" \
  --hire_v2_phrase_test_spans "${TEST_SPANS}" \
  --pretrain_choice ViT-B/16 \
  --dataset_name TAG-PEDES \
  --root_dir "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --test_batch_size 512 \
  --num_workers "${NUM_WORKERS}" \
  --num_epoch "${NUM_EPOCH}" \
  --seed "${SEED}" \
  --sampler random \
  --val_dataset test
