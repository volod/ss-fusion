#!/bin/bash
# Adaptive vLLM entrypoint for Qwen2.5-VL-7B.
#
# - Kills any stale vLLM processes before starting so containers can be
#   restarted cleanly without orphaned processes holding GPU memory.
# - Queries nvidia-smi for TOTAL and USED VRAM (not just "free") so that
#   GPU memory held by other Ubuntu processes (X11, other ML jobs, etc.)
#   is accounted for when computing --cpu-offload-gb.
# - TurboQuant features: FP8 KV cache (--kv-cache-dtype fp8_e5m2),
#   chunked prefill, BF16 weights.
set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.5}"
MAX_SEQS="${VLLM_MAX_SEQS:-4}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-128000}"
# GB to keep reserved for Florence + CLIP + DINOv3 + OS overhead in the worker
WORKER_RESERVE_GB="${VLLM_WORKER_RESERVE_GB:-4}"

# --- Kill stale vLLM processes (safe to ignore if none exist) ---
echo "[vllm_entrypoint] Cleaning up any stale vLLM processes ..."
pkill -TERM -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 2
pkill -KILL -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true

# --- Detect GPU VRAM via nvidia-smi (total and currently used) ---
# "free" from nvidia-smi can be misleading when another process holds memory
# but has not yet released it.  Using total - used gives a stable picture that
# accounts for ALL GPU consumers (X11, other ML workers, etc.).
TOTAL_MIB=16384   # fallback: 16 GiB
USED_MIB=0

if command -v nvidia-smi &>/dev/null; then
    _total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
             | head -1 | tr -d '[:space:]') || true
    _used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
            | head -1 | tr -d '[:space:]') || true
    if [[ -n "$_total" && "$_total" =~ ^[0-9]+$ ]]; then
        TOTAL_MIB=$_total
    fi
    if [[ -n "$_used" && "$_used" =~ ^[0-9]+$ ]]; then
        USED_MIB=$_used
    fi
fi

# --- Compute cpu-offload-gb using Python to avoid bc/awk dependency ---
# available = total - used_by_others - worker_reserve
# offload   = max(0, model_fp16 - available + 1)   (+1 GB safety margin)
CPU_OFFLOAD_GB=$(python3 - "$TOTAL_MIB" "$USED_MIB" "$WORKER_RESERVE_GB" <<'PYEOF'
import sys
total_mib  = int(sys.argv[1])
used_mib   = int(sys.argv[2])
reserve_gb = float(sys.argv[3])
total_gb     = total_mib / 1024.0
used_gb      = used_mib  / 1024.0
available_gb = total_gb - used_gb - reserve_gb
model_gb     = 14.0   # FP16 Qwen2.5-VL-7B
offload = max(0, int(model_gb - available_gb) + 1) if available_gb < model_gb else 0
import sys as _sys
print(
    f"  total={total_gb:.1f}GB  used_by_others={used_gb:.1f}GB  "
    f"reserve={reserve_gb:.1f}GB  available={available_gb:.1f}GB  "
    f"→ cpu_offload={offload}GB",
    file=_sys.stderr,
)
print(offload)
PYEOF
)

echo "[vllm_entrypoint] GPU VRAM: total=${TOTAL_MIB}MiB  used_now=${USED_MIB}MiB"
echo "[vllm_entrypoint] cpu-offload-gb=${CPU_OFFLOAD_GB}  model=${MODEL}"
echo "[vllm_entrypoint] gpu-util=${GPU_UTIL}  max-seqs=${MAX_SEQS}  max-len=${MAX_LEN}"

# Build the argument list
ARGS=(
    --model "${MODEL}"
    --host 0.0.0.0
    --port 8000
    --dtype bfloat16
    --gpu-memory-utilization "${GPU_UTIL}"
    --kv-cache-dtype fp8_e5m2
    --enable-chunked-prefill
    --max-num-seqs "${MAX_SEQS}"
    --max-model-len "${MAX_LEN}"
    --limit-mm-per-prompt "image=1"
    --trust-remote-code
)

if [[ "${CPU_OFFLOAD_GB}" -gt 0 ]]; then
    ARGS+=(--cpu-offload-gb "${CPU_OFFLOAD_GB}")
    echo "[vllm_entrypoint] CPU offload active: ${CPU_OFFLOAD_GB}GB → CPU RAM"
fi

echo "[vllm_entrypoint] Starting vLLM ..."
exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
