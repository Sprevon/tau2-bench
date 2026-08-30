#!/usr/bin/env bash
set -euo pipefail

# Pi Telecom solo eval through ~/.pi/agent/models.json provider openrouter-free.
# Does not start vLLM. Default model is minimax/minimax-m3:free.

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

PROVIDER="${TAU2_PI_PROVIDER:-openrouter-free}"
MODEL="${TAU2_PI_MODEL:-minimax/minimax-m3:free}"
SPLIT="${TAU2_PI_SPLIT:-test}"
# Catalog marks this model as non-reasoning. Override with TAU2_PI_THINKING if needed.
THINKING="${TAU2_PI_THINKING:-off}"
MAX_TASKS="${TAU2_PI_MAX_TASKS:-0}"
TASK_TIMEOUT="${TAU2_PI_TASK_TIMEOUT:-900}"
STAMP="${TAU2_PI_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
MODEL_SLUG="$(printf '%s' "$MODEL" | tr '/:' '__')"
OUTPUT_ROOT="${TAU2_PI_OUTPUT_ROOT:-/root/autodl-tmp/logs/pi_tau2_eval/${STAMP}}"
OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_SLUG}"

mkdir -p "$OUTPUT_DIR"
echo "output_root=${OUTPUT_ROOT}"
echo "provider=${PROVIDER} model=${MODEL} split=${SPLIT} thinking=${THINKING}"

python_args=(
  "$REPO_ROOT/scripts/eval_pi_telecom.py"
  --provider "$PROVIDER"
  --model "$MODEL"
  --split "$SPLIT"
  --thinking "$THINKING"
  --timeout "$TASK_TIMEOUT"
  --python "$TAU2_PYTHON"
  --skip-vllm-wait
  --output-dir "$OUTPUT_DIR"
)
if [[ "$MAX_TASKS" != "0" ]]; then
  python_args+=(--max-tasks "$MAX_TASKS")
fi

"$TAU2_PYTHON" -u "${python_args[@]}"
echo "wrote ${OUTPUT_DIR}/summary.json"
