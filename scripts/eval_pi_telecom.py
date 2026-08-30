#!/usr/bin/env python3
"""Run Pi coding-agent Telecom solo evals against a local vLLM server."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tau2.domains.telecom.environment import get_tasks  # noqa: E402

DEFAULT_PROVIDER = "local-vllm"
DEFAULT_THINKING = "medium"
DEFAULT_SPLIT = "small"
DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1"
SKILL_PROMPT_PREFIX = "/skill:telecom-solo-support"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Pi model id, e.g. Qwen3-0.6B")
    parser.add_argument("--provider", default=os.getenv("TAU2_PI_PROVIDER", DEFAULT_PROVIDER))
    parser.add_argument("--split", default=os.getenv("TAU2_PI_SPLIT", DEFAULT_SPLIT))
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--max-tasks", type=int, default=int(os.getenv("TAU2_PI_MAX_TASKS", "0")))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pi-bin", default=os.getenv("TAU2_PI_BIN", "pi"))
    parser.add_argument("--thinking", default=os.getenv("TAU2_PI_THINKING", DEFAULT_THINKING))
    parser.add_argument("--endpoint", default=os.getenv("TAU2_VLLM_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("TAU2_PI_TASK_TIMEOUT", "600")))
    parser.add_argument("--vllm-wait", type=int, default=int(os.getenv("TAU2_VLLM_WAIT_SECS", "180")))
    parser.add_argument("--skip-vllm-wait", action="store_true")
    parser.add_argument("--python", default=os.getenv("TAU2_PI_PYTHON", sys.executable))
    return parser.parse_args()


def _task_slug(task_id: str, index: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", task_id).strip("_")
    return f"{index:03d}_{safe[:96]}"


def _compose_prompt(task: Any) -> str:
    return (
        f"{SKILL_PROMPT_PREFIX} Task ID: {task.id}\n"
        f"Policy mode: workflow\n\n"
        f"{task.ticket}"
    )


def wait_for_vllm(endpoint: str, timeout: float) -> None:
    url = endpoint.rstrip("/") + "/models"
    deadline = time.time() + timeout
    last_error: Exception | None = None
    request = urllib.request.Request(url, method="GET")
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                if 200 <= response.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        time.sleep(2)
    raise SystemExit(f"vLLM was not ready at {url}: {last_error}")


def _select_tasks(split: str, task_ids: list[str] | None, max_tasks: int) -> list[Any]:
    catalog = get_tasks(task_split_name=split)
    by_id = {task.id: task for task in catalog}
    if task_ids:
        missing = [task_id for task_id in task_ids if task_id not in by_id]
        if missing:
            raise SystemExit(f"Unknown task ids for split {split}: {missing}")
        selected = [by_id[task_id] for task_id in task_ids]
    else:
        selected = catalog
    if max_tasks > 0:
        selected = selected[:max_tasks]
    if not selected:
        raise SystemExit(f"No tasks selected for split {split}")
    return selected


def _parse_pi_events(raw: str) -> dict[str, Any]:
    tool_names: list[str] = []
    texts: list[str] = []
    events = 0
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        events += 1
        event_type = payload.get("type")
        if event_type == "message_update":
            assistant = payload.get("assistantMessageEvent") or {}
            if assistant.get("type") == "toolcall_start" and assistant.get("toolName"):
                tool_names.append(str(assistant["toolName"]))
        if event_type == "message_end":
            message = payload.get("message") or {}
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        texts.append(str(item.get("text") or ""))
    return {
        "n_json_events": events,
        "pi_tool_names": tool_names,
        "final_text": texts[-1] if texts else "",
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _run_pi_task(
    *,
    task: Any,
    args: argparse.Namespace,
    task_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    task_dir.mkdir(parents=True, exist_ok=True)
    prompt = _compose_prompt(task)
    (task_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    eval_path = task_dir / "eval.json"
    session_dir = task_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = task_dir / "pi.stdout.jsonl"
    stderr_path = task_dir / "pi.stderr.log"

    env = os.environ.copy()
    source_path = str(repo_root / "src")
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        source_path if not pythonpath else f"{source_path}{os.pathsep}{pythonpath}"
    )
    env["TAU2_PI_PYTHON"] = args.python
    env["TAU2_TELECOM_TASK_ID"] = task.id
    env["TAU2_TELECOM_EVAL_OUT"] = str(eval_path)
    env.pop("TAU2_TELECOM_AUTO_PROMPT", None)

    command = [
        args.pi_bin,
        "--provider",
        args.provider,
        "--model",
        args.model,
        "--thinking",
        args.thinking,
        "--no-builtin-tools",
        "--approve",
        "--mode",
        "json",
        "--print",
        "--offline",
        "--session-dir",
        str(session_dir),
        prompt,
    ]
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd=str(repo_root),
            env=env,
            text=True,
            capture_output=True,
            timeout=args.timeout,
            check=False,
        )
        timed_out = False
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
    elapsed = time.time() - started
    stdout_path.write_text(stdout or "", encoding="utf-8")
    stderr_path.write_text(stderr or "", encoding="utf-8")
    eval_payload = _read_json(eval_path)
    pi_events = _parse_pi_events(stdout or "")
    error = None
    if timed_out:
        error = f"pi timed out after {args.timeout}s"
    elif returncode != 0:
        error = f"pi exited {returncode}"
    if eval_payload is None:
        error = error or "missing eval.json"
        reward = 0.0
        n_tool_calls = len(pi_events["pi_tool_names"])
        n_tool_errors = 0
    else:
        if eval_payload.get("error"):
            error = error or str(eval_payload["error"])
        reward = float(eval_payload.get("reward") or 0.0)
        n_tool_calls = int(eval_payload.get("n_tool_calls") or 0)
        n_tool_errors = int(eval_payload.get("n_tool_errors") or 0)

    result = {
        "task_id": task.id,
        "reward": reward,
        "success": bool(reward) and error is None,
        "error": error,
        "n_tool_calls": n_tool_calls,
        "n_tool_errors": n_tool_errors,
        "elapsed_sec": round(elapsed, 2),
        "returncode": returncode,
        "timed_out": timed_out,
        "final_text": pi_events["final_text"],
        "pi_tool_names": pi_events["pi_tool_names"],
        "eval": eval_payload,
    }
    (task_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    args = _parse_args()
    repo_root = _REPO_ROOT
    if not args.skip_vllm_wait:
        wait_for_vllm(args.endpoint, args.vllm_wait)

    tasks = _select_tasks(args.split, args.task_ids, args.max_tasks)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"model={args.model} split={args.split} n_tasks={len(tasks)}", flush=True)
    print(f"thinking={args.thinking} output={output_dir}", flush=True)

    results: list[dict[str, Any]] = []
    traces_path = output_dir / "results.jsonl"
    with traces_path.open("w", encoding="utf-8") as handle:
        for index, task in enumerate(tasks, 1):
            slug = _task_slug(task.id, index)
            task_dir = output_dir / "tasks" / slug
            print(f"\n==== {index}/{len(tasks)} {task.id} ====", flush=True)
            result = _run_pi_task(
                task=task,
                args=args,
                task_dir=task_dir,
                repo_root=repo_root,
            )
            results.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{args.model}] {index}/{len(tasks)} {task.id} "
                f"reward={result['reward']} tools={result['n_tool_calls']} "
                f"errors={result['n_tool_errors']} elapsed={result['elapsed_sec']}s "
                f"status={result['error'] or 'ok'}",
                flush=True,
            )

    n_success = sum(1 for item in results if item["success"])
    summary = {
        "model": args.model,
        "provider": args.provider,
        "split": args.split,
        "thinking": args.thinking,
        "n_tasks": len(results),
        "n_success": n_success,
        "success_rate": (n_success / len(results)) if results else 0.0,
        "n_errors": sum(1 for item in results if item["error"]),
        "mean_tool_calls": (
            sum(item["n_tool_calls"] for item in results) / len(results) if results else 0.0
        ),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "results": [
            {
                "task_id": item["task_id"],
                "reward": item["reward"],
                "success": item["success"],
                "error": item["error"],
                "n_tool_calls": item["n_tool_calls"],
                "n_tool_errors": item["n_tool_errors"],
                "elapsed_sec": item["elapsed_sec"],
            }
            for item in results
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {traces_path}", flush=True)
    print(f"wrote {summary_path}", flush=True)
    print(
        json.dumps(
            {
                "n_tasks": summary["n_tasks"],
                "n_success": summary["n_success"],
                "success_rate": summary["success_rate"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
