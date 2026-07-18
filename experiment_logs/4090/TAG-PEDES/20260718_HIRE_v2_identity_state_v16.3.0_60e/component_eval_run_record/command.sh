#!/usr/bin/env bash
set -o pipefail
ROOT=/root/autodl-tmp/HIRE_v2_identity_state_v1630_overlay8cddc23_run1
OUT=/root/autodl-tmp/HIRE_v2_identity_state_logs/6fd7272/TAG-PEDES/20260718_024215_hire_v2_identity_state_tagpedes_60e_seed1_6fd7272
REC=/root/autodl-tmp/HIRE_v2_identity_state_run_records/6fd7272_component_eval_run1
cd "$ROOT"
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/irra190/bin/python tools/hire_v2/eval_identity_state_components.py \
  --config-file "$OUT/configs.yaml" \
  --checkpoint "$OUT/best.pth" \
  --output-json "$OUT/hire_v2_identity_state_components.json" \
  --query-chunk 128
rc=$?
printf '%s\n' "$rc" > "$REC/exit_code"
date --iso-8601=seconds > "$REC/finished_at"
exit "$rc"
