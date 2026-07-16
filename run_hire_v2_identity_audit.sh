#!/bin/bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/IRRA_light_baseline}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/irra190/bin/python}
DEVICE=${DEVICE:-cuda}
NUM_WORKERS=${NUM_WORKERS:-8}
IMAGE_BATCH_SIZE=${IMAGE_BATCH_SIZE:-128}
TEXT_BATCH_SIZE=${TEXT_BATCH_SIZE:-512}
QUERY_CHUNK=${QUERY_CHUNK:-128}
GALLERY_CHUNK=${GALLERY_CHUNK:-1024}
SUPPORT_EPOCHS=${SUPPORT_EPOCHS:-0,15,30,45,54,59}
RETRIEVAL_EPOCHS=${RETRIEVAL_EPOCHS:-54}
MAX_TRAIN_QUERIES=${MAX_TRAIN_QUERIES:-0}

V162_CONFIG=${V162_CONFIG:-/root/autodl-tmp/HIRE_v2_identity_logs/82601f8/TAG-PEDES/20260716_201333_hire_v2_identity_tagpedes_60e_seed1_82601f8/configs.yaml}
V162_CHECKPOINT=${V162_CHECKPOINT:-/root/autodl-tmp/HIRE_v2_identity_logs/82601f8/TAG-PEDES/20260716_201333_hire_v2_identity_tagpedes_60e_seed1_82601f8/best.pth}
V161_CONFIG=${V161_CONFIG:-/root/autodl-tmp/HIRE_v2_anchor_logs/48e61f8_full60/TAG-PEDES/20260716_104228_hire_v2_anchor_tagpedes_60e_seed1_48e61f8/configs.yaml}
V161_CHECKPOINT=${V161_CHECKPOINT:-/root/autodl-tmp/HIRE_v2_anchor_logs/48e61f8_full60/TAG-PEDES/20260716_104228_hire_v2_anchor_tagpedes_60e_seed1_48e61f8/best.pth}
OUTPUT_DIR=${OUTPUT_DIR:-/root/autodl-tmp/HIRE_v2_identity_audit/v16.2.0_$(date +%Y%m%d_%H%M%S)}

cd "${PROJECT_ROOT}"

ARGS=(
  tools/hire_v2/audit_v162_identity.py
  --config-file "${V162_CONFIG}"
  --checkpoint "${V162_CHECKPOINT}"
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --num-workers "${NUM_WORKERS}"
  --image-batch-size "${IMAGE_BATCH_SIZE}"
  --text-batch-size "${TEXT_BATCH_SIZE}"
  --query-chunk "${QUERY_CHUNK}"
  --gallery-chunk "${GALLERY_CHUNK}"
  --support-epochs "${SUPPORT_EPOCHS}"
  --retrieval-epochs "${RETRIEVAL_EPOCHS}"
  --max-train-queries "${MAX_TRAIN_QUERIES}"
)

if [ -n "${V161_CONFIG}" ] || [ -n "${V161_CHECKPOINT}" ]; then
  if [ ! -f "${V161_CONFIG}" ] || [ ! -f "${V161_CHECKPOINT}" ]; then
    echo "v16.1 comparison requested but config/checkpoint is missing." >&2
    echo "Set both V161_CONFIG and V161_CHECKPOINT, or set both to empty strings." >&2
    exit 2
  fi
  ARGS+=(--anchor-config-file "${V161_CONFIG}" --anchor-checkpoint "${V161_CHECKPOINT}")
fi

"${PYTHON_BIN}" "${ARGS[@]}"

echo "Audit output: ${OUTPUT_DIR}"
