#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/HIRE_v2_anchor_48e61f8_run2
CUDA_VISIBLE_DEVICES=0 DATA_ROOT=/root/autodl-tmp/datasets OUTPUT_DIR=/root/autodl-tmp/HIRE_v2_anchor_logs/48e61f8_full60 PYTHON_BIN=/root/miniconda3/envs/irra190/bin/python SEED=1 BATCH_SIZE=64 NUM_EPOCH=60 EXP_NAME=hire_v2_anchor_tagpedes_60e_seed1_48e61f8 bash run_hire_v2_anchor_4090_tag.sh
