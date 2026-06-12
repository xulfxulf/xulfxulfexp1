#!/bin/bash
set -e

DATASET_NAME=${DATASET_NAME:-CUHK-PEDES}
DATA_ROOT=${DATA_ROOT:-/root/shared-nvme/zixiangwang/yxyx/RDE_3090/datasets}
OUTPUT_DIR=${OUTPUT_DIR:-logs}
IRRA_LIGHT_MODE=${IRRA_LIGHT_MODE:-single_pure}
NUM_EPOCH=${NUM_EPOCH:-60}
BATCH_SIZE=${BATCH_SIZE:-64}
SEED=${SEED:-1}
IMG_AUG=${IMG_AUG:-0}
EXP_NAME=${EXP_NAME:-irra_light_${DATASET_NAME}_${IRRA_LIGHT_MODE}_aug${IMG_AUG}_seed${SEED}}
PYTHON_BIN=${PYTHON_BIN:-python}
AUG_ARGS=""

if [ -z "${OMP_NUM_THREADS:-}" ]; then
  export OMP_NUM_THREADS=4
fi

VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
if [[ "${VISIBLE_DEVICES}" == *,* ]]; then
  echo "First-round IRRA-light experiments must use a single GPU. Got CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES}"
  exit 1
fi

if [ "${IMG_AUG}" = "1" ]; then
  AUG_ARGS="--img_aug"
fi

CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} \
"${PYTHON_BIN}" train.py \
  --name "${EXP_NAME}" \
  --irra_light \
  --irra_light_mode "${IRRA_LIGHT_MODE}" \
  --irra_light_identity_loss sdm \
  ${AUG_ARGS} \
  --batch_size "${BATCH_SIZE}" \
  --sampler random \
  --dataset_name "${DATASET_NAME}" \
  --val_dataset test \
  --root_dir "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --num_epoch "${NUM_EPOCH}"
