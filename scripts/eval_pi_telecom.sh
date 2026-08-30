#!/usr/bin/env bash
set -euo pipefail

# Sequential Pi + tau2 Telecom solo eval for local Qwen3 checkpoints.
# Default: thinking on (medium), small split, 0.6B then 4B then 8B.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f /root/autodl-tmp/env.sh ]]; then
  # shellcheck disable=SC1091
  source /root/autodl-tmp/env.sh >/dev/null
fi

TAU2_PYTHON="${TAU2_PI_PYTHON:-${TAU2_ENV:-/root/autodl-tmp/envs/tau2}/bin/python}"
if [[ ! -x "$TAU2_PYTHON" ]]; then
  TAU2_PYTHON="$(command -v python3 || command -v python)"
fi
export TAU2_PI_PYTHON="$TAU2_PYTHON"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${TAU2_ENV:-/root/autodl-tmp/envs/tau2}/bin:/root/autodl-tmp/envs/node-v22.19.0-linux-x64/bin:${PATH:-}"

SPLIT="${TAU2_PI_SPLIT:-small}"
THINKING="${TAU2_PI_THINKING:-medium}"
PROVIDER="${TAU2_PI_PROVIDER:-local-vllm}"
ENDPOINT="${TAU2_VLLM_ENDPOINT:-http://127.0.0.1:8000/v1}"
SKIP_VLLM="${TAU2_PI_SKIP_VLLM:-0}"
MAX_TASKS="${TAU2_PI_MAX_TASKS:-0}"
TASK_TIMEOUT="${TAU2_PI_TASK_TIMEOUT:-600}"
VLLM_WAIT="${TAU2_VLLM_WAIT_SECS:-300}"
STAMP="${TAU2_PI_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${TAU2_PI_OUTPUT_ROOT:-/root/autodl-tmp/logs/pi_tau2_eval/${STAMP}}"

if [[ -n "${TAU2_PI_MODELS:-}" ]]; then
  # shellcheck disable=SC2206
  MODELS=(${TAU2_PI_MODELS})
else
  MODELS=(Qwen3-0.6B Qwen3-4B Qwen3-8B)
fi

start_script_for_model() {
  case "$1" in
    Qwen3-0.6B) echo "$REPO_ROOT/scripts/start_vllm_qwen3_0_6b.sh" ;;
    Qwen3-4B) echo "$REPO_ROOT/scripts/start_vllm_qwen3_4b.sh" ;;
    Qwen3-8B) echo "$REPO_ROOT/scripts/start_vllm_qwen3_8b.sh" ;;
    *)
      echo "No vLLM start script mapped for model: $1" >&2
      return 1
      ;;
  esac
}

VLLM_PID=""
stop_vllm() {
  if [[ -z "${VLLM_PID}" ]]; then
    return 0
  fi
  if kill -0 "${VLLM_PID}" 2>/dev/null; then
    pkill -P "${VLLM_PID}" 2>/dev/null || true
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
  pkill -f "VLLM::EngineCore" 2>/dev/null || true
  VLLM_PID=""
  sleep 2
}

cleanup() {
  stop_vllm
}
trap cleanup EXIT INT TERM

start_vllm() {
  local model="$1"
  local log_path="$2"
  local script
  script="$(start_script_for_model "$model")"
  echo "Starting vLLM for ${model} via ${script}"
  bash "$script" >"$log_path" 2>&1 &
  VLLM_PID=$!
}

mkdir -p "$OUTPUT_ROOT"
echo "output_root=${OUTPUT_ROOT}"
echo "split=${SPLIT} thinking=${THINKING} models=${MODELS[*]}"

COMPARISON_PATH="${OUTPUT_ROOT}/comparison.json"
python_args=(
  "$REPO_ROOT/scripts/eval_pi_telecom.py"
  --provider "$PROVIDER"
  --split "$SPLIT"
  --thinking "$THINKING"
  --endpoint "$ENDPOINT"
  --timeout "$TASK_TIMEOUT"
  --vllm-wait "$VLLM_WAIT"
  --python "$TAU2_PYTHON"
)
if [[ "$MAX_TASKS" != "0" ]]; then
  python_args+=(--max-tasks "$MAX_TASKS")
fi

summaries=()
for model in "${MODELS[@]}"; do
  model_dir="${OUTPUT_ROOT}/${model}"
  mkdir -p "$model_dir"
  if [[ "$SKIP_VLLM" != "1" ]]; then
    stop_vllm
    start_vllm "$model" "${model_dir}/vllm.log"
  fi
  echo "Evaluating ${model}"
  "$TAU2_PYTHON" -u "${python_args[@]}" --model "$model" --output-dir "$model_dir"
  summaries+=("${model_dir}/summary.json")
  if [[ "$SKIP_VLLM" != "1" ]]; then
    stop_vllm
  fi
done

"$TAU2_PYTHON" - "$COMPARISON_PATH" "${summaries[@]}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for path in sys.argv[2:]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows.append(
        {
            "model": payload.get("model"),
            "split": payload.get("split"),
            "thinking": payload.get("thinking"),
            "n_tasks": payload.get("n_tasks"),
            "n_success": payload.get("n_success"),
            "success_rate": payload.get("success_rate"),
            "n_errors": payload.get("n_errors"),
            "mean_tool_calls": payload.get("mean_tool_calls"),
            "summary_path": path,
        }
    )
out.write_text(json.dumps({"runs": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"runs": rows}, ensure_ascii=False, indent=2))
PY

echo "wrote ${COMPARISON_PATH}"
