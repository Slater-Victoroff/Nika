#!/usr/bin/env bash
# Batch-training wrapper for running one Nika config across the standard UVG sequences.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG="${1:-small}"
DATASET_ROOT="${DATASET_ROOT:-static/benchmarks/uvg}"
DEVICE="${DEVICE:-cuda:0}"
MAX_FRAMES="${MAX_FRAMES:-600}"

VIDEOS=(beauty bosphorus honey jockey ready shake yacht)

for video in "${VIDEOS[@]}"; do
  python3 "${SCRIPT_DIR}/train_nika.py" \
    --dataset-root "${DATASET_ROOT}" \
    --video "${video}" \
    --config "${CONFIG}" \
    --device "${DEVICE}" \
    --max-frames "${MAX_FRAMES}"
done
