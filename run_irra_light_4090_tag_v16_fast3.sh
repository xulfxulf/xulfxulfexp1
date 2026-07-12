#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

export DATASET_NAME=${DATASET_NAME:-TAG-PEDES}
export DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/datasets}
export OUTPUT_DIR=${OUTPUT_DIR:-${SCRIPT_DIR}/logs_4090/TAG-PEDES}
export IRRA_LIGHT_MODE=${IRRA_LIGHT_MODE:-split_bag_safe}
export NUM_EPOCH=${NUM_EPOCH:-60}
export BATCH_SIZE=${BATCH_SIZE:-64}
export SEED=${SEED:-1}
export IMG_AUG=${IMG_AUG:-0}
export SUPPORT_SIZE=${SUPPORT_SIZE:-3}
export PRETRAIN_CHOICE=${PRETRAIN_CHOICE:-ViT-B/16}
export FAST3_INPUT_DIR=${FAST3_INPUT_DIR:-${SCRIPT_DIR}/diagnostics/TAG-PEDES/v16_fast3_inputs}
export SUPPORT_CONSISTENCY_CSV=${SUPPORT_CONSISTENCY_CSV:-${FAST3_INPUT_DIR}/support_reliability_hard_only.csv}
export SUPPORT_RELATION_CSV=${SUPPORT_RELATION_CSV:-${FAST3_INPUT_DIR}/support_hard_contradiction.csv}
export HARD_NEGATIVE_CSV=${HARD_NEGATIVE_CSV:-${FAST3_INPUT_DIR}/hard_negative_pool.csv}
export EXP_NAME=${EXP_NAME:-irra_light_${IRRA_LIGHT_MODE}_v16_fast3_tagpedes_${NUM_EPOCH}e_seed${SEED}}

case "${IRRA_LIGHT_MODE}" in
  split_bag_safe)
    required_inputs=("${SUPPORT_CONSISTENCY_CSV}")
    ;;
  split_bag_state)
    required_inputs=("${SUPPORT_CONSISTENCY_CSV}" "${SUPPORT_RELATION_CSV}")
    ;;
  split_bag_state_hn)
    required_inputs=(
      "${SUPPORT_CONSISTENCY_CSV}"
      "${SUPPORT_RELATION_CSV}"
      "${HARD_NEGATIVE_CSV}"
    )
    ;;
  *)
    echo "IRRA_LIGHT_MODE must be split_bag_safe, split_bag_state, or split_bag_state_hn. Got ${IRRA_LIGHT_MODE}" >&2
    exit 2
    ;;
esac

for input_path in "${required_inputs[@]}"; do
  if [ ! -f "${input_path}" ]; then
    echo "Missing required v16 fast3 input: ${input_path}" >&2
    exit 2
  fi
done

bash "${SCRIPT_DIR}/run_irra_light.sh"
