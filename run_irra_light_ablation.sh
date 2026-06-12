#!/bin/bash
set -e

# First-round mechanism check:
# single_pure -> raw CLIP capacity baseline
# single_proj_pure -> single-head projection capacity control
# split_pure -> identity/state split-head method
# ID-classification modes remain available but are second-stage checks.

MODES=${MODES:-"single_pure single_proj_pure split_pure"}
BASE_EXP_NAME=${BASE_EXP_NAME:-irra_light_mechanism_check}

for MODE in ${MODES}; do
  export IRRA_LIGHT_MODE="${MODE}"
  export EXP_NAME="${BASE_EXP_NAME}_${MODE}"
  bash run_irra_light.sh
done
