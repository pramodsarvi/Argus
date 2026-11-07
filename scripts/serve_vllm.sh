#!/usr/bin/env bash
# One replica per GPU (PCIe, no NVLink -> no tensor parallel).
# Usage: GPU=0 PORT=8000 MODEL=Qwen/Qwen2.5-VL-7B-Instruct FRAMES=/mnt/nvme/frames scripts/serve_vllm.sh
set -euo pipefail
GPU=${GPU:-0}; PORT=${PORT:-8000}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
FRAMES=${FRAMES:-/mnt/nvme/frames}
MAX_FRAMES=${MAX_FRAMES:-64}
QUANT=${QUANT:-fp8}            # set QUANT=none for BF16 baseline
EXTRA=()
[[ "$QUANT" != "none" ]] && EXTRA+=(--quantization "$QUANT")

# DEV_MODE exposes /reset_prefix_cache for the spike. Do NOT enable in production.
CUDA_VISIBLE_DEVICES=$GPU VLLM_SERVER_DEV_MODE=${DEV_MODE:-1} exec vllm serve "$MODEL" \
  --port "$PORT" \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --allowed-local-media-path "$FRAMES" \
  --limit-mm-per-prompt "{\"image\": $MAX_FRAMES}" \
  --max-model-len "${MAX_LEN:-32768}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.85}" \
  "${EXTRA[@]}"
