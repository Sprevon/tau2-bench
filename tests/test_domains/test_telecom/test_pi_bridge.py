import pytest

from tau2.domains.telecom.environment import get_tasks
from tau2.domains.telecom.pi_bridge import TelecomPiBridge


def test_bridge_describes_every_solo_tool() -> None:
    bridge = TelecomPiBridge()

    tools = bridge.describe_tools()
    names = {tool["name"] for tool in tools}

    assert len(tools) == 43
    assert len(names) == 43
    assert "get_customer_by_phone" in names
    assert "check_status_bar" in names
    assert "transfer_to_human_agents" in names
    assert {tool["source"] for tool in tools} == {"assistant", "device"}
    assert all(tool["parameters"]["type"] == "object" for tool in tools)


def test_bridge_loads_solo_task_and_routes_device_tool() -> None:
    bridge = TelecomPiBridge()
    task = get_tasks("small")[0]

    loaded = bridge.load_task(task.id)
    result = bridge.call_tool(
        tool_call_id="test-call",
        tool_name="check_status_bar",
        arguments={},
    )

    assert loaded["task_id"] == task.id
    assert loaded["ticket"] == task.ticket
    assert loaded["policy_type"] == "workflow"
    assert loaded["tool_count"] == 43
    assert result["error"] is False
    assert "Status Bar:" in result["content"]


def test_bridge_requires_task_before_tool_call() -> None:
    bridge = TelecomPiBridge()

    with pytest.raises(ValueError, match="/telecom-task"):
        bridge.call_tool("test-call", "check_status_bar", {})
