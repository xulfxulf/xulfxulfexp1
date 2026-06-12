#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

DATASET_NAME=${DATASET_NAME:-CUHK-PEDES}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/datasets}
OUTPUT_DIR=${OUTPUT_DIR:-logs_4090}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/irra190/bin/python}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NUM_EPOCH=${NUM_EPOCH:-5}
BATCH_SIZE=${BATCH_SIZE:-64}
SEED=${SEED:-1}
IMG_AUG=${IMG_AUG:-0}
GPU_MONITOR_INTERVAL=${GPU_MONITOR_INTERVAL:-30}
MODES=${MODES:-"single_pure single_proj_pure split_pure"}
SUMMARY_FILE=${SUMMARY_FILE:-${OUTPUT_DIR}/${DATASET_NAME}/irra_light_mechanism_check_small_summary.tsv}

export DATASET_NAME DATA_ROOT OUTPUT_DIR PYTHON_BIN CUDA_VISIBLE_DEVICES
export NUM_EPOCH BATCH_SIZE SEED IMG_AUG

mkdir -p "${OUTPUT_DIR}/${DATASET_NAME}" "${OUTPUT_DIR}/gpu_monitors"

start_gpu_monitor() {
  local csv="$1"
  local stop_file="$2"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU_NOT_AVAILABLE" > "${csv}"
    return 0
  fi
  echo "timestamp,index,name,utilization.gpu,memory.used,memory.total" > "${csv}"
  (
    while [ ! -f "${stop_file}" ]; do
      nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits >> "${csv}" 2>/dev/null || true
      sleep "${GPU_MONITOR_INTERVAL}"
    done
  ) &
  echo $!
}

for MODE in ${MODES}; do
  export IRRA_LIGHT_MODE="${MODE}"
  export EXP_NAME="mechanism_check_small_${DATASET_NAME}_${MODE}_e${NUM_EPOCH}_aug${IMG_AUG}_seed${SEED}"
  GPU_CSV="${OUTPUT_DIR}/gpu_monitors/${EXP_NAME}_gpu.csv"
  STOP_FILE="${OUTPUT_DIR}/gpu_monitors/${EXP_NAME}.stop"
  rm -f "${STOP_FILE}"

  MONITOR_PID="$(start_gpu_monitor "${GPU_CSV}" "${STOP_FILE}")"
  set +e
  bash run_irra_light.sh
  TRAIN_STATUS=$?
  set -e
  touch "${STOP_FILE}"
  if [ -n "${MONITOR_PID}" ]; then
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
  rm -f "${STOP_FILE}"
  if [ "${TRAIN_STATUS}" -ne 0 ]; then
    echo "Training failed for ${MODE} with exit code ${TRAIN_STATUS}" >&2
    exit "${TRAIN_STATUS}"
  fi

  RUN_DIR="$(find "${OUTPUT_DIR}/${DATASET_NAME}" -maxdepth 1 -type d -name "*_${EXP_NAME}" -printf "%T@ %p\n" | sort -nr | head -1 | cut -d' ' -f2-)"
  if [ -z "${RUN_DIR}" ] || [ ! -d "${RUN_DIR}" ]; then
    echo "Could not locate run directory for ${MODE}" >&2
    exit 1
  fi
  cp "${GPU_CSV}" "${RUN_DIR}/gpu_monitor.csv" 2>/dev/null || true

  if [ ! -f "${RUN_DIR}/best.pth" ]; then
    echo "Missing best checkpoint for ${MODE}: ${RUN_DIR}/best.pth" >&2
    exit 1
  fi

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" test.py \
    --config_file "${RUN_DIR}/configs.yaml" 2>&1 | tee "${RUN_DIR}/final_test_stdout.log"

  "${PYTHON_BIN}" summarize_irra_light_run.py \
    --run_dir "${RUN_DIR}" \
    --gpu_csv "${RUN_DIR}/gpu_monitor.csv" \
    --summary_file "${SUMMARY_FILE}"
done

echo "Summary: ${SUMMARY_FILE}"
