#!/usr/bin/env bash
# One vLLM replica per GPU (PCIe, no NVLink -> no tensor parallel).
# Usage: GPU=0 PORT=8000 MODEL=Qwen/Qwen2.5-VL-7B-Instruct FRAMES=/mnt/nvme/frames scripts/serve_vllm.sh
#
# Production-safe defaults (override with env):
#   HOST=127.0.0.1      bind address; the worker runs on the same VM. Use 0.0.0.0 only behind a firewall.
#   VLLM_API_KEY=...    if set, the server requires it (clients read the same variable).
#   DEV_MODE=0          1 exposes /reset_prefix_cache and /reset_mm_cache; needed ONLY by bench/spike.py
#                       and bench/latency.py. Never in production.
#   MAX_SEQS=64         max concurrent sequences (~ cameras in flight x prompts); lower = less KV pressure.
#   MM_CACHE_GB=4       multimodal processor cache: avoids re-preprocessing the same image per prompt.
#   MAX_LEN=32768       context. 32 frames x 256 tokens is ~8k, so 16384 is plenty for camera chunks.
#   GPU_UTIL=0.85       leave headroom if ffmpeg/NVDEC shares the GPU.
#   SPEC='{"method":"ngram",...}'  optional speculative decoding. Measured: no gain here; off.
# Guided JSON is NOT the default (vlmpipe.bundle.STRUCTURED_MODE=prompt): it degraded Qwen2.5-VL
# output. The structured-outputs flag only prevents the infinite-whitespace loop if grammar mode
# is selected.
set -euo pipefail
GPU=${GPU:-0}; PORT=${PORT:-8000}; HOST=${HOST:-127.0.0.1}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
FRAMES=${FRAMES:-/mnt/nvme/frames}
MAX_FRAMES=${MAX_FRAMES:-64}
QUANT=${QUANT:-fp8}            # set QUANT=none for BF16 baseline
EXTRA=()
[[ "$QUANT" != "none" ]] && EXTRA+=(--quantization "$QUANT")
[[ -n "${SPEC:-}" ]] && EXTRA+=(--speculative-config "$SPEC")
[[ -n "${VLLM_API_KEY:-}" ]] && EXTRA+=(--api-key "$VLLM_API_KEY")

CUDA_VISIBLE_DEVICES=$GPU VLLM_SERVER_DEV_MODE=${DEV_MODE:-0} exec vllm serve "$MODEL" \
  --host "$HOST" --port "$PORT" \
  --enable-prefix-caching \
  --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}' \
  --enable-prompt-tokens-details \
  --allowed-local-media-path "$FRAMES" \
  --limit-mm-per-prompt "{\"image\": $MAX_FRAMES}" \
  --max-model-len "${MAX_LEN:-32768}" \
  --max-num-seqs "${MAX_SEQS:-64}" \
  --mm-processor-cache-gb "${MM_CACHE_GB:-4}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.85}" \
  "${EXTRA[@]}"
