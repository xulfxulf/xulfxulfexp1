#!/bin/bash
set -e

export DATASET_NAME=${DATASET_NAME:-TAG-PEDES}
export DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/datasets}
export OUTPUT_DIR=${OUTPUT_DIR:-logs_4090/TAG-PEDES}
export IRRA_LIGHT_MODE=${IRRA_LIGHT_MODE:-single_proj_bag}
export NUM_EPOCH=${NUM_EPOCH:-60}
export BATCH_SIZE=${BATCH_SIZE:-64}
export SEED=${SEED:-1}
export IMG_AUG=${IMG_AUG:-0}
export SUPPORT_SIZE=${SUPPORT_SIZE:-3}

case "${IRRA_LIGHT_MODE}" in
  single_proj_bag|split_bag)
    ;;
  *)
    echo "IRRA_LIGHT_MODE must be single_proj_bag or split_bag for v16 scheme1. Got ${IRRA_LIGHT_MODE}"
    exit 1
    ;;
esac

export EXP_NAME=${EXP_NAME:-irra_light_${IRRA_LIGHT_MODE}_v16_scheme1_tagpedes_60e_seed${SEED}}

bash run_irra_light.sh
