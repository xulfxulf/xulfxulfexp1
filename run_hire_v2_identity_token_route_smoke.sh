#!/bin/bash
set -euo pipefail

NUM_EPOCH=${NUM_EPOCH:-1} \
EXP_NAME=${EXP_NAME:-hire_v2_identity_token_route_smoke} \
OUTPUT_DIR=${OUTPUT_DIR:-logs_4090/HIRE_v2_identity_token_route_smoke} \
bash run_hire_v2_identity_token_route_4090_tag.sh
