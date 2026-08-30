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

from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import Task
from tau2.domains.telecom.environment import (
    TelecomEnvironment,
    get_environment,
    get_tasks,
)
from tau2.environment.tool import Tool
from tau2.environment.toolkit import ToolKitBase


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
        initial_state = task.initial_state
        if initial_state is not None:
            environment.set_state(
                initialization_data=initial_state.initialization_data,
                initialization_actions=initial_state.initialization_actions,
                message_history=initial_state.message_history or [],
            )

        self.environment = environment
        self.task = task
        return {
            "task_id": task.id,
            "ticket": task.ticket,
            "policy_type": "workflow",
            "tool_count": len(self.describe_tools()),
        }

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
        return {
            "content": response.content,
            "error": response.error,
            "task_id": self.task.id,
            "tool_name": tool_name,
        }

    def status(self) -> dict[str, Any]:
        """Return non-sensitive bridge state for the interactive command."""
        return {
            "task_id": self.task.id if self.task is not None else None,
            "loaded": self.task is not None,
            "policy_type": "workflow",
            "tool_count": len(self.describe_tools()),
        }

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
        if method == "status":
            return self.status()
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
