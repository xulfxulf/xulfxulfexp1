#!/bin/bash
set -uo pipefail

CODE_ROOT=/root/autodl-tmp/HIRE_v2_RSTPReid_three_v16_e6b3159_run1
RUN_ROOT=/root/autodl-tmp/HIRE_v2_RSTPReid_run_records/e6b3159_full3x60_run1
OUTPUT_ROOT=/root/autodl-tmp/HIRE_v2_RSTPReid_logs/e6b3159
PYTHON_BIN=/root/miniconda3/envs/irra190/bin/python

mkdir -p "${RUN_ROOT}" "${OUTPUT_ROOT}"
cd "${CODE_ROOT}"

run_formal() {
  local label="$1"
  local script="$2"
  local experiment="$3"
  local evaluator="$4"
  local component_json="$5"
  local record_dir="${RUN_ROOT}/${label}"
  local output_dir="${OUTPUT_ROOT}/${label}"
  mkdir -p "${record_dir}" "${output_dir}"
  date -Iseconds > "${record_dir}/started_at.txt"

  env \
    CUDA_VISIBLE_DEVICES=0 \
    DATASET_NAME=RSTPReid \
    DATA_ROOT=/root/autodl-tmp/datasets \
    OUTPUT_DIR="${output_dir}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    NUM_EPOCH=60 \
    BATCH_SIZE=64 \
    SEED=1 \
    NUM_WORKERS=8 \
    SUPPORT_SIZE=3 \
    AUX_WEIGHT=0.1 \
    PRETRAIN_CHOICE=ViT-B/16 \
    EXP_NAME="${experiment}" \
    bash "${script}" > "${record_dir}/nohup.log" 2>&1
  local train_status=$?
  printf '%s\n' "${train_status}" > "${record_dir}/train_exit_code.txt"

  local experiment_dir=""
  experiment_dir=$(find "${output_dir}/RSTPReid" -name configs.yaml -type f -printf '%h\n' 2>/dev/null | sort | tail -n 1)
  printf '%s\n' "${experiment_dir}" > "${record_dir}/experiment_dir.txt"

  local eval_status=1
  if [ "${train_status}" -eq 0 ] && [ -n "${experiment_dir}" ] && [ -f "${experiment_dir}/best.pth" ]; then
    "${PYTHON_BIN}" "${evaluator}" \
      --config-file "${experiment_dir}/configs.yaml" \
      --checkpoint "${experiment_dir}/best.pth" \
      --output-json "${experiment_dir}/${component_json}" \
      > "${record_dir}/component_eval.log" 2>&1
    eval_status=$?
  fi
  printf '%s\n' "${eval_status}" > "${record_dir}/eval_exit_code.txt"
  date -Iseconds > "${record_dir}/finished_at.txt"

  if [ "${train_status}" -ne 0 ] || [ "${eval_status}" -ne 0 ]; then
    return 1
  fi
  return 0
}

overall=0
run_formal \
  v16.1.0_anchor \
  run_hire_v2_anchor_4090_tag.sh \
  hire_v2_anchor_v16.1.0_rstpreid_60e_seed1_e6b3159_run1 \
  tools/hire_v2/eval_anchor_components.py \
  hire_v2_anchor_components.json || overall=1
run_formal \
  v16.2.1_identity_balanced \
  run_hire_v2_identity_balanced_4090_tag.sh \
  hire_v2_identity_balanced_v16.2.1_rstpreid_60e_seed1_e6b3159_run1 \
  tools/hire_v2/eval_identity_balanced_components.py \
  hire_v2_identity_balanced_components.json || overall=1
run_formal \
  v16.4.0_identity_token_route \
  run_hire_v2_identity_token_route_4090_tag.sh \
  hire_v2_identity_token_route_v16.4.0_rstpreid_60e_seed1_e6b3159_run1 \
  tools/hire_v2/eval_identity_token_route_components.py \
  hire_v2_identity_token_route_components.json || overall=1

printf '%s\n' "${overall}" > "${RUN_ROOT}/queue_exit_code.txt"
date -Iseconds > "${RUN_ROOT}/queue_finished_at.txt"
exit "${overall}"
