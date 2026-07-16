#!/bin/bash
set -u
printf "%s\n" "$$" > "/root/autodl-tmp/HIRE_v2_identity_run_records/82601f8/formal_20260716_201257/wrapper.pid"
printf "%s\n" "$(date -Is)" > "/root/autodl-tmp/HIRE_v2_identity_run_records/82601f8/formal_20260716_201257/started_at.txt"
status=0
cd "/root/autodl-tmp/HIRE_v2_identity_82601f8_run1" || status=$?
if [ "$status" -eq 0 ]; then
  CUDA_VISIBLE_DEVICES=0 \
  DATA_ROOT=/root/autodl-tmp/datasets \
  OUTPUT_DIR="/root/autodl-tmp/HIRE_v2_identity_logs/82601f8" \
  PYTHON_BIN=/root/miniconda3/envs/irra190/bin/python \
  NUM_EPOCH=60 \
  SEED=1 \
  BATCH_SIZE=64 \
  NUM_WORKERS=8 \
  SUPPORT_SIZE=3 \
  AUX_WEIGHT=0.1 \
  PRETRAIN_CHOICE=ViT-B/16 \
  EXP_NAME="hire_v2_identity_tagpedes_60e_seed1_82601f8" \
  bash run_hire_v2_identity_4090_tag.sh || status=$?
fi
printf "%s\n" "$status" > "/root/autodl-tmp/HIRE_v2_identity_run_records/82601f8/formal_20260716_201257/exit_code.txt"
printf "%s\n" "$(date -Is)" > "/root/autodl-tmp/HIRE_v2_identity_run_records/82601f8/formal_20260716_201257/finished_at.txt"
exit "$status"
