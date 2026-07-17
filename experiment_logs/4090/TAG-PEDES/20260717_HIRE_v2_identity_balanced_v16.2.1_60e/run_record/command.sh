#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/HIRE_v2_identity_balanced_9bbfc16_run1
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 DATA_ROOT=/root/autodl-tmp/datasets OUTPUT_DIR=/root/autodl-tmp/HIRE_v2_identity_balanced_logs/9bbfc16 PYTHON_BIN=/root/miniconda3/envs/irra190/bin/python NUM_EPOCH=60 SEED=1 BATCH_SIZE=64 NUM_WORKERS=8 SUPPORT_SIZE=3 AUX_WEIGHT=0.1 EXP_NAME=hire_v2_identity_balanced_tagpedes_60e_seed1_9bbfc16 bash run_hire_v2_identity_balanced_4090_tag.sh
