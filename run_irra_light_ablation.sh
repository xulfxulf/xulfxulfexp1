#!/bin/bash
set -e

# First-round IRRA-light ablations from the design note:
# A single_pure: single embedding, SDM + one-to-one ITC
# B split_pure : identity head SDM + state head one-to-one ITC
# C single_id  : single embedding, SDM + one-to-one ITC + ID classification
# D split_id   : identity head SDM + ID classification, state head one-to-one ITC

MODES=${MODES:-"single_pure split_pure single_id split_id"}
BASE_EXP_NAME=${BASE_EXP_NAME:-irra_light_first_round}

for MODE in ${MODES}; do
  export IRRA_LIGHT_MODE="${MODE}"
  export EXP_NAME="${BASE_EXP_NAME}_${MODE}"
  bash run_irra_light.sh
done
