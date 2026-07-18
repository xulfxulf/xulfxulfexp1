#!/usr/bin/env bash
set -o pipefail
cd "/root/autodl-tmp/HIRE_v2_identity_state_v1630_overlay8cddc23_run1"
CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=/root/autodl-tmp/datasets \
OUTPUT_DIR="/root/autodl-tmp/HIRE_v2_identity_state_logs/6fd7272" \
PYTHON_BIN="/root/miniconda3/envs/irra190/bin/python" \
NUM_EPOCH=60 \
SEED=1 \
BATCH_SIZE=64 \
NUM_WORKERS=8 \
SUPPORT_SIZE=3 \
AUX_WEIGHT=0.1 \
STATE_TOPK=50 \
STATE_IMAGE_TOKENS=16 \
STATE_TEXT_TOKENS=8 \
PRETRAIN_CHOICE=ViT-B/16 \
EXP_NAME=hire_v2_identity_state_tagpedes_60e_seed1_6fd7272 \
bash run_hire_v2_identity_state_4090_tag.sh
rc=$?
printf '%s\n' "$rc" > "/root/autodl-tmp/HIRE_v2_identity_state_run_records/6fd7272_full60_run1/exit_code"
date --iso-8601=seconds > "/root/autodl-tmp/HIRE_v2_identity_state_run_records/6fd7272_full60_run1/finished_at"
exit "$rc"
