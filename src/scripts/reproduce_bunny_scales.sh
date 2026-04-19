#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_ROOT="${DATASET_ROOT:-static/benchmarks}"
VIDEO="${VIDEO:-bunny}"
DEVICE="${DEVICE:-cuda:0}"
MAX_FRAMES="${MAX_FRAMES:-132}"

CONFIGS=(xxs xs small)

for config in "${CONFIGS[@]}"; do
  python3 "${SCRIPT_DIR}/train_nika.py" \
    --dataset-root "${DATASET_ROOT}" \
    --video "${VIDEO}" \
    --config "${config}" \
    --device "${DEVICE}" \
    --max-frames "${MAX_FRAMES}"
done
