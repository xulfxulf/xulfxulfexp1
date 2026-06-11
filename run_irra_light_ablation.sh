#!/bin/bash
set -e

# First-round IRRA-light ablations from simplest to most complex.

MODES=${MODES:-"single_pure single_proj_pure split_pure single_id single_proj_id split_id"}
BASE_EXP_NAME=${BASE_EXP_NAME:-irra_light_first_round}

for MODE in ${MODES}; do
  export IRRA_LIGHT_MODE="${MODE}"
  export EXP_NAME="${BASE_EXP_NAME}_${MODE}"
  bash run_irra_light.sh
done
