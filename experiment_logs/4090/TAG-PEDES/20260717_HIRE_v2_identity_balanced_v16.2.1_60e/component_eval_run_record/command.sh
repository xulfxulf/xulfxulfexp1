#!/usr/bin/env bash
set -o pipefail
cd "/root/autodl-tmp/HIRE_v2_identity_balanced_9bbfc16_run1"
"/root/miniconda3/envs/irra190/bin/python" tools/hire_v2/eval_identity_balanced_components.py \
  --config-file "/root/autodl-tmp/HIRE_v2_identity_balanced_logs/9bbfc16/TAG-PEDES/20260717_152943_hire_v2_identity_balanced_tagpedes_60e_seed1_9bbfc16/configs.yaml" \
  --checkpoint "/root/autodl-tmp/HIRE_v2_identity_balanced_logs/9bbfc16/TAG-PEDES/20260717_152943_hire_v2_identity_balanced_tagpedes_60e_seed1_9bbfc16/best.pth" \
  --output-json "/root/autodl-tmp/HIRE_v2_identity_balanced_logs/9bbfc16/TAG-PEDES/20260717_152943_hire_v2_identity_balanced_tagpedes_60e_seed1_9bbfc16/hire_v2_identity_balanced_components.json" \
  --query-chunk 128 \
  --gallery-chunk 512
rc=$?
printf '%s\n' "$rc" > "/root/autodl-tmp/HIRE_v2_identity_balanced_run_records/9bbfc16_component_eval_run1/exit_code"
date --iso-8601=seconds > "/root/autodl-tmp/HIRE_v2_identity_balanced_run_records/9bbfc16_component_eval_run1/finished_at"
exit "$rc"
