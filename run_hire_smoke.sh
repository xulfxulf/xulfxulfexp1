#!/bin/bash
set -euo pipefail
NUM_EPOCH=1 EXP_NAME=${EXP_NAME:-hire_main_smoke} bash run_hire_4090_tag.sh
