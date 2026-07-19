#!/bin/bash
set -euo pipefail
NUM_EPOCH=${NUM_EPOCH:-1} \
EXP_NAME=${EXP_NAME:-hire_v2_phrase_route_v1670_smoke} \
OUTPUT_DIR=${OUTPUT_DIR:-/root/autodl-tmp/HIRE_v2_phrase_route_v1670_smoke} \
bash run_hire_v2_phrase_route_v1670_4090_tag.sh
