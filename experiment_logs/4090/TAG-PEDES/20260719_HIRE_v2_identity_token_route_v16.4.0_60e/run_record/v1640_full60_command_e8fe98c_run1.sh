#!/bin/bash
set -euo pipefail

cd /root/autodl-tmp/HIRE_v2_identity_token_route_v1640_e8fe98c_run1

exec env \
  CUDA_VISIBLE_DEVICES=0 \
  DATA_ROOT=/root/autodl-tmp/datasets \
  OUTPUT_DIR=/root/autodl-tmp/HIRE_v2_identity_token_route_logs/e8fe98c \
  PYTHON_BIN=/root/miniconda3/envs/irra190/bin/python \
  NUM_EPOCH=60 \
  BATCH_SIZE=64 \
  SEED=1 \
  NUM_WORKERS=8 \
  SUPPORT_SIZE=3 \
  AUX_WEIGHT=0.1 \
  PRETRAIN_CHOICE=ViT-B/16 \
  EXP_NAME=hire_v2_identity_token_route_v1640_e8fe98c_tagpedes_60e_seed1_run1 \
  bash run_hire_v2_identity_token_route_4090_tag.sh
