#!/bin/bash
set -euo pipefail
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 PROJECT_ROOT=/root/autodl-tmp/HIRE_v2_identity_audit_code_82601f8_20260717_run1 PYTHON_BIN=/root/miniconda3/envs/irra190/bin/python OUTPUT_DIR=/root/autodl-tmp/HIRE_v2_identity_audit/v16.2.0_full_20260717_run1 bash /root/autodl-tmp/HIRE_v2_identity_audit_code_82601f8_20260717_run1/run_hire_v2_identity_audit.sh
