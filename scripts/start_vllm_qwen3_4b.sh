#!/usr/bin/env bash

set -euo pipefail

# vLLM OpenAI-compatible server for the local Qwen3-4B checkpoint.
# Tool calls use the Hermes parser. Override any default with TAU2_VLLM_*.

TAU2_VLLM_MODEL_PATH="${TAU2_VLLM_MODEL_PATH:-/root/autodl-tmp/models/Qwen3-4B}"
TAU2_VLLM_ENV_DIR="${TAU2_VLLM_ENV_DIR:-/root/envs/toolcall}"
TAU2_VLLM_HOST="${TAU2_VLLM_HOST:-127.0.0.1}"
TAU2_VLLM_PORT="${TAU2_VLLM_PORT:-8000}"
TAU2_VLLM_SERVED_MODEL_NAME="${TAU2_VLLM_SERVED_MODEL_NAME:-Qwen3-4B}"
TAU2_VLLM_MAX_MODEL_LEN="${TAU2_VLLM_MAX_MODEL_LEN:-32768}"
TAU2_VLLM_GPU_MEMORY_UTILIZATION="${TAU2_VLLM_GPU_MEMORY_UTILIZATION:-0.85}"

TAU2_VLLM_BIN="${TAU2_VLLM_ENV_DIR}/bin/vllm"
TAU2_VLLM_PYTHON="${TAU2_VLLM_ENV_DIR}/bin/python"

if [[ ! -x "${TAU2_VLLM_BIN}" ]]; then
	echo "vLLM executable not found: ${TAU2_VLLM_BIN}" >&2
	exit 1
fi

if [[ ! -x "${TAU2_VLLM_PYTHON}" ]]; then
	echo "Python executable not found: ${TAU2_VLLM_PYTHON}" >&2
	exit 1
fi

if [[ ! -f "${TAU2_VLLM_MODEL_PATH}/config.json" ]]; then
	echo "Model config not found: ${TAU2_VLLM_MODEL_PATH}/config.json" >&2
	exit 1
fi

if ! "${TAU2_VLLM_PYTHON}" -c "import flash_attn"; then
	echo "flash_attn is not importable from ${TAU2_VLLM_PYTHON}" >&2
	exit 1
fi

# Ampere/Ada use FlashAttention-2. vLLM 0.12.0 reads these variables
# before selecting its attention backend.
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
export VLLM_FLASH_ATTN_VERSION="2"

echo "Starting vLLM with:"
echo "  model=${TAU2_VLLM_MODEL_PATH}"
echo "  served_model_name=${TAU2_VLLM_SERVED_MODEL_NAME}"
echo "  endpoint=http://${TAU2_VLLM_HOST}:${TAU2_VLLM_PORT}/v1"
echo "  attention_backend=${VLLM_ATTENTION_BACKEND}"
echo "  max_model_len=${TAU2_VLLM_MAX_MODEL_LEN}"
echo "  gpu_memory_utilization=${TAU2_VLLM_GPU_MEMORY_UTILIZATION}"
echo "  tool_call_parser=hermes"

exec "${TAU2_VLLM_BIN}" serve "${TAU2_VLLM_MODEL_PATH}" \
	--host "${TAU2_VLLM_HOST}" \
	--port "${TAU2_VLLM_PORT}" \
	--served-model-name "${TAU2_VLLM_SERVED_MODEL_NAME}" \
	--dtype bfloat16 \
	--tensor-parallel-size 1 \
	--max-model-len "${TAU2_VLLM_MAX_MODEL_LEN}" \
	--gpu-memory-utilization "${TAU2_VLLM_GPU_MEMORY_UTILIZATION}" \
	--generation-config vllm \
	--enable-auto-tool-choice \
	--tool-call-parser hermes \
	--reasoning-parser qwen3 \
	"$@"
