#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/HIRE_main_20260715_5aa9104_run2 && CUDA_VISIBLE_DEVICES=0 DATA_ROOT=/root/autodl-tmp/datasets OUTPUT_DIR=/root/autodl-tmp/HIRE_logs/5aa9104_full60 PYTHON_BIN=/root/miniconda3/envs/irra190/bin/python SEED=1 BATCH_SIZE=64 NUM_EPOCH=60 EXP_NAME=hire_main_tagpedes_60e_seed1_5aa9104 bash run_hire_4090_tag.sh
