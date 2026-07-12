#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-${SCRIPT_DIR}}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/datasets}
FAST3_INPUT_DIR=${FAST3_INPUT_DIR:-${PROJECT_ROOT}/diagnostics/TAG-PEDES/v16_fast3_inputs}
RUN_LOG_ROOT=${RUN_LOG_ROOT:-${PROJECT_ROOT}/logs_4090/TAG-PEDES}
CODE_RECORD_ROOT=${CODE_RECORD_ROOT:-${PROJECT_ROOT}/run_records/v16_fast3}
VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
TIMESTAMP=${V16_FAST3_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}

if [ "${NUM_EPOCH:-60}" != "60" ]; then
  echo "run_v16_fast3_all.sh is intentionally full-run only: NUM_EPOCH must be 60" >&2
  exit 2
fi

if [ -n "$(git -C "${PROJECT_ROOT}" status --porcelain)" ]; then
  echo "Refusing to snapshot a dirty worktree. Commit intended fast3 code first." >&2
  exit 2
fi

COMMIT=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
mkdir -p "${CODE_RECORD_ROOT}" "${RUN_LOG_ROOT}"

for MODE in split_bag_safe split_bag_state split_bag_state_hn; do
  RECORD_DIR="${CODE_RECORD_ROOT}/${TIMESTAMP}_${MODE}"
  SNAPSHOT_DIR="${RECORD_DIR}/code"
  INPUT_DIR="${RECORD_DIR}/inputs"
  RUN_OUTPUT_DIR="${RUN_LOG_ROOT}/v16_fast3_${TIMESTAMP}_${MODE}"
  mkdir -p "${SNAPSHOT_DIR}" "${INPUT_DIR}" "${RUN_OUTPUT_DIR}"
  git -C "${PROJECT_ROOT}" archive --format=tar "${COMMIT}" | tar -xf - -C "${SNAPSHOT_DIR}"

  cp -p "${FAST3_INPUT_DIR}/support_reliability_hard_only.csv" "${INPUT_DIR}/"
  SUPPORT_CONSISTENCY_CSV="${INPUT_DIR}/support_reliability_hard_only.csv"
  SUPPORT_RELATION_CSV=""
  HARD_NEGATIVE_CSV=""
  if [ "${MODE}" != "split_bag_safe" ]; then
    cp -p "${FAST3_INPUT_DIR}/support_hard_contradiction.csv" "${INPUT_DIR}/"
    SUPPORT_RELATION_CSV="${INPUT_DIR}/support_hard_contradiction.csv"
  fi
  if [ "${MODE}" = "split_bag_state_hn" ]; then
    cp -p "${FAST3_INPUT_DIR}/hard_negative_pool.csv" "${INPUT_DIR}/"
    HARD_NEGATIVE_CSV="${INPUT_DIR}/hard_negative_pool.csv"
  fi

  {
    echo "commit=${COMMIT}"
    echo "mode=${MODE}"
    echo "timestamp=${TIMESTAMP}"
    echo "data_root=${DATA_ROOT}"
    echo "output_dir=${RUN_OUTPUT_DIR}"
    echo "cuda_visible_devices=${VISIBLE_DEVICES}"
    sha256sum "${INPUT_DIR}"/*.csv
  } > "${RECORD_DIR}/run_manifest.txt"

  (
    cd "${SNAPSHOT_DIR}"
    CUDA_VISIBLE_DEVICES="${VISIBLE_DEVICES}" \
    DATA_ROOT="${DATA_ROOT}" \
    OUTPUT_DIR="${RUN_OUTPUT_DIR}" \
    IRRA_LIGHT_MODE="${MODE}" \
    NUM_EPOCH=60 \
    BATCH_SIZE=64 \
    SEED=1 \
    IMG_AUG=0 \
    SUPPORT_SIZE=3 \
    PRETRAIN_CHOICE=ViT-B/16 \
    SUPPORT_CONSISTENCY_CSV="${SUPPORT_CONSISTENCY_CSV}" \
    SUPPORT_RELATION_CSV="${SUPPORT_RELATION_CSV}" \
    HARD_NEGATIVE_CSV="${HARD_NEGATIVE_CSV}" \
    EXP_NAME="irra_light_${MODE}_v16_fast3_tagpedes_60e_seed1" \
    bash "${SNAPSHOT_DIR}/run_irra_light_4090_tag_v16_fast3.sh"
  ) 2>&1 | tee "${RECORD_DIR}/nohup.log"
done
