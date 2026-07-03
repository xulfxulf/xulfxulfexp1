#!/bin/bash
set -e

export DATASET_NAME=${DATASET_NAME:-TAG-PEDES}
export DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/datasets}
export OUTPUT_DIR=${OUTPUT_DIR:-logs_4090/TAG-PEDES}
export IRRA_LIGHT_MODE=${IRRA_LIGHT_MODE:-single_proj_bag_consistency}
export NUM_EPOCH=${NUM_EPOCH:-60}
export BATCH_SIZE=${BATCH_SIZE:-64}
export SEED=${SEED:-1}
export IMG_AUG=${IMG_AUG:-0}
export SUPPORT_SIZE=${SUPPORT_SIZE:-3}
export SUPPORT_CONSISTENCY_CSV=${SUPPORT_CONSISTENCY_CSV:-/root/autodl-tmp/IRRA_light_baseline/diagnostics/TAG-PEDES/scheme2_pre_analysis/support_conflict/intra_image_caption_consistency.csv}

case "${IRRA_LIGHT_MODE}" in
  single_proj_bag_consistency|split_bag_consistency)
    ;;
  *)
    echo "IRRA_LIGHT_MODE must be single_proj_bag_consistency or split_bag_consistency for v16 scheme2. Got ${IRRA_LIGHT_MODE}"
    exit 1
    ;;
esac

if [ ! -f "${SUPPORT_CONSISTENCY_CSV}" ]; then
  echo "Missing SUPPORT_CONSISTENCY_CSV=${SUPPORT_CONSISTENCY_CSV}"
  exit 1
fi

export EXP_NAME=${EXP_NAME:-irra_light_${IRRA_LIGHT_MODE}_v16_scheme2_tagpedes_60e_seed${SEED}}

bash run_irra_light.sh
