"""JSON-lines bridge exposing the solo Telecom environment to Pi.

The bridge keeps one :class:`TelecomEnvironment` alive for the lifetime of the
process.  Pi owns tool selection and invocation while tau2 remains the single
source of truth for tool schemas, state transitions, and task initialization.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.data_model.simulation import DBCheck, EnvAssertionCheck, RewardInfo
from tau2.data_model.tasks import RewardType, Task
from tau2.domains.telecom.environment import (
    TelecomEnvironment,
    get_environment,
    get_tasks,
)
from tau2.environment.tool import Tool
from tau2.environment.toolkit import ToolKitBase
from tau2.environment.toolkit import get_tool_types
from tau2.evaluator.evaluator_action import ActionEvaluator
from tau2.evaluator.evaluator_communicate import CommunicateEvaluator


@dataclass(frozen=True)
class PiToolDescriptor:
    """Serializable metadata needed to register one tau2 tool in Pi."""

    name: str
    description: str
    parameters: dict[str, Any]
    source: str
    tool_type: str
    mutates_state: bool

    def model_dump(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "source": self.source,
            "tool_type": self.tool_type,
            "mutates_state": self.mutates_state,
        }


class TelecomPiBridge:
    """Stateful adapter used by the project-local Pi extension."""

    def __init__(self) -> None:
        self.environment = self._new_environment()
        self._tasks: dict[str, Task] | None = None
        self.task: Task | None = None
        self.tool_calls: list[dict[str, Any]] = []
        self.assistant_texts: list[str] = []

    @property
    def tasks(self) -> dict[str, Task]:
        """Load the task catalog only when a task is selected."""
        if self._tasks is None:
            self._tasks = {
                task.id: task for task in get_tasks(task_split_name=None)
            }
        return self._tasks

    @staticmethod
    def _new_environment() -> TelecomEnvironment:
        # The workflow policy is the official policy variant written for an
        # agent that directly operates both the account and the phone.
        return get_environment(solo_mode=True, policy_type="workflow")

    @staticmethod
    def _apply_initial_state(environment: TelecomEnvironment, task: Task) -> None:
        initial_state = task.initial_state
        if initial_state is None:
            return
        environment.set_state(
            initialization_data=initial_state.initialization_data,
            initialization_actions=initial_state.initialization_actions,
            message_history=initial_state.message_history or [],
        )

    @staticmethod
    def _describe_tool(
        tool: Tool,
        toolkit: ToolKitBase,
        source: str,
    ) -> PiToolDescriptor:
        schema = tool.openai_schema["function"]
        return PiToolDescriptor(
            name=tool.name,
            description=schema["description"],
            parameters=schema["parameters"],
            source=source,
            tool_type=toolkit.tool_type(tool.name).value,
            mutates_state=toolkit.tool_mutates_state(tool.name),
        )

    def describe_tools(self) -> list[dict[str, Any]]:
        """Return all assistant and device tool definitions for solo mode."""
        assistant_tools = [
            self._describe_tool(tool, self.environment.tools, "assistant")
            for tool in self.environment.get_tools()
        ]
        device_tools = [
            self._describe_tool(tool, self.environment.user_tools, "device")
            for tool in self.environment.get_user_tools()
        ]
        descriptors = sorted(assistant_tools + device_tools, key=lambda item: item.name)
        return [descriptor.model_dump() for descriptor in descriptors]

    def load_task(self, task_id: str) -> dict[str, Any]:
        """Reset the environment and apply the requested task's initial state."""
        task = self.tasks.get(task_id)
        if task is None:
            raise ValueError(f"Unknown telecom task id: {task_id}")
        if not task.ticket:
            raise ValueError(f"Telecom task {task_id} has no solo ticket")

        environment = self._new_environment()
        self._apply_initial_state(environment, task)
        self.environment = environment
        self.task = task
        self.tool_calls = []
        self.assistant_texts = []
        return {
            "task_id": task.id,
            "ticket": task.ticket,
            "policy_type": "workflow",
            "tool_count": len(self.describe_tools()),
        }

    def record_assistant_text(self, content: str) -> dict[str, Any]:
        """Record a final assistant message for communication evaluation."""
        if self.task is None:
            raise ValueError("No telecom task loaded. Run /telecom-task <task-id> first.")
        normalized = content.strip()
        if normalized:
            self.assistant_texts.append(normalized)
        return {"recorded": bool(normalized), "task_id": self.task.id}

    def call_tool(
        self,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute one Pi tool call against the active solo environment."""
        if self.task is None:
            raise ValueError("No telecom task loaded. Run /telecom-task <task-id> first.")

        response = self.environment.get_response(
            ToolCall(
                id=tool_call_id,
                name=tool_name,
                arguments=arguments,
                requestor="assistant",
            )
        )
        record = {
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "content": response.content,
            "error": bool(response.error),
            "task_id": self.task.id,
        }
        self.tool_calls.append(record)
        return {
            "content": record["content"],
            "error": record["error"],
            "task_id": record["task_id"],
            "tool_name": tool_name,
        }

    def status(self) -> dict[str, Any]:
        """Return non-sensitive bridge state for the interactive command."""
        return {
            "task_id": self.task.id if self.task is not None else None,
            "loaded": self.task is not None,
            "policy_type": "workflow",
            "tool_count": len(self.describe_tools()),
            "n_tool_calls": len(self.tool_calls),
        }

    def list_tasks(self, split: str | None = None) -> dict[str, Any]:
        """Return task ids for a named split, or the full catalog."""
        if split is None:
            tasks = list(self.tasks.values())
            split_name = "all"
        else:
            tasks = get_tasks(task_split_name=split)
            split_name = split
        return {
            "split": split_name,
            "count": len(tasks),
            "task_ids": [task.id for task in tasks],
        }

    def evaluate(self) -> dict[str, Any]:
        """Score the live solo environment with the official reward components."""
        if self.task is None:
            raise ValueError("No telecom task loaded. Run /telecom-task <task-id> first.")

        task = self.task
        criteria = task.evaluation_criteria
        if criteria is None:
            reward_info = RewardInfo(
                reward=1.0,
                info={"note": "No evaluation criteria"},
            )
            return self._evaluation_payload(reward_info)

        env_assertions = criteria.env_assertions or []
        env_assertion_checks: list[EnvAssertionCheck] = []
        env_assertion_reward = 1.0
        for assertion in env_assertions:
            success = self.environment.run_env_assertion(
                assertion,
                raise_assertion_error=False,
            )
            check = EnvAssertionCheck(
                env_assertion=assertion,
                met=success,
                reward=1.0 if success else 0.0,
            )
            env_assertion_checks.append(check)
            env_assertion_reward *= check.reward

        gold = self._new_environment()
        self._apply_initial_state(gold, task)
        for action in criteria.actions or []:
            gold.get_response(
                ToolCall(
                    id=action.action_id,
                    name=action.name,
                    arguments=action.arguments,
                    requestor=action.requestor,
                )
            )

        agent_db_match = self.environment.get_db_hash() == gold.get_db_hash()
        user_db_match = self.environment.get_user_db_hash() == gold.get_user_db_hash()
        db_match = bool(agent_db_match and user_db_match)
        db_reward = 1.0 if db_match else 0.0
        db_check = DBCheck(db_match=db_match, db_reward=db_reward)

        reward = 1.0
        reward_breakdown: dict[RewardType, float] = {}
        basis = criteria.reward_basis or []

        trajectory = [
            AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id=call["tool_call_id"],
                        name=call["tool_name"],
                        arguments=call["arguments"],
                        requestor="assistant",
                    )
                ],
            )
            for call in self.tool_calls
        ]
        trajectory.extend(
            AssistantMessage(role="assistant", content=content)
            for content in self.assistant_texts
        )
        tool_types = get_tool_types(self.environment.tools)
        if self.environment.user_tools is not None:
            tool_types.update(get_tool_types(self.environment.user_tools))
        action_info = ActionEvaluator.calculate_reward(
            task=task,
            full_trajectory=trajectory,
            tool_types=tool_types,
        )
        communicate_info = CommunicateEvaluator.calculate_reward(
            task=task,
            full_trajectory=trajectory,
        )

        if RewardType.DB in basis:
            reward_breakdown[RewardType.DB] = db_reward
            reward *= db_reward
        if RewardType.ENV_ASSERTION in basis:
            reward_breakdown[RewardType.ENV_ASSERTION] = env_assertion_reward
            reward *= env_assertion_reward
        if RewardType.ACTION in basis:
            reward_breakdown[RewardType.ACTION] = action_info.reward
            reward *= action_info.reward
        if RewardType.COMMUNICATE in basis:
            reward_breakdown[RewardType.COMMUNICATE] = communicate_info.reward
            reward *= communicate_info.reward
        unsupported = set(basis) - {
            RewardType.DB,
            RewardType.ENV_ASSERTION,
            RewardType.ACTION,
            RewardType.COMMUNICATE,
        }
        if unsupported:
            raise ValueError(
                f"Pi Telecom bridge cannot evaluate reward bases: {sorted(item.value for item in unsupported)}"
            )

        reward_info = RewardInfo(
            reward=reward,
            db_check=db_check,
            env_assertions=env_assertion_checks,
            action_checks=action_info.action_checks,
            communicate_checks=communicate_info.communicate_checks,
            reward_basis=basis,
            reward_breakdown=reward_breakdown,
        )
        return self._evaluation_payload(reward_info)

    def _evaluation_payload(self, reward_info: RewardInfo) -> dict[str, Any]:
        n_errors = sum(1 for call in self.tool_calls if call["error"])
        payload = reward_info.model_dump(mode="json")
        payload.update(
            {
                "task_id": self.task.id if self.task is not None else None,
                "policy_type": "workflow",
                "n_tool_calls": len(self.tool_calls),
                "n_tool_errors": n_errors,
                "tool_calls": self.tool_calls,
                "assistant_texts": self.assistant_texts,
            }
        )
        return payload

    def handle(self, method: str, params: dict[str, Any]) -> Any:
        """Dispatch one protocol request."""
        if method == "describe_tools":
            return self.describe_tools()
        if method == "load_task":
            return self.load_task(task_id=str(params["task_id"]))
        if method == "call_tool":
            arguments = params.get("arguments", {})
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object")
            return self.call_tool(
                tool_call_id=str(params["tool_call_id"]),
                tool_name=str(params["tool_name"]),
                arguments=arguments,
            )
        if method == "record_assistant_text":
            return self.record_assistant_text(content=str(params.get("content", "")))
        if method == "status":
            return self.status()
        if method == "list_tasks":
            split = params.get("split")
            if split is not None:
                split = str(split)
            return self.list_tasks(split=split)
        if method == "evaluate":
            return self.evaluate()
        raise ValueError(f"Unknown bridge method: {method}")


def serve() -> None:
    """Serve newline-delimited JSON requests on stdin/stdout."""
    bridge = TelecomPiBridge()
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        request_id: Any = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("Bridge request must be a JSON object")
            request_id = request.get("id")
            method = request.get("method")
            params = request.get("params", {})
            if not isinstance(method, str):
                raise ValueError("Bridge request method must be a string")
            if not isinstance(params, dict):
                raise ValueError("Bridge request params must be a JSON object")
            response = {
                "id": request_id,
                "ok": True,
                "result": bridge.handle(method, params),
            }
        except Exception as exc:
            response = {"id": request_id, "ok": False, "error": str(exc)}

        sys.stdout.write(json.dumps(response, default=str) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    serve()
