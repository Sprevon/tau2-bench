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


def test_bridge_requires_task_before_evaluate() -> None:
    bridge = TelecomPiBridge()

    with pytest.raises(ValueError, match="/telecom-task"):
        bridge.evaluate()


def test_bridge_lists_small_split() -> None:
    bridge = TelecomPiBridge()

    listed = bridge.list_tasks(split="small")

    assert listed["split"] == "small"
    assert listed["count"] == 20
    assert listed["task_ids"][0] == get_tasks("small")[0].id


def test_bridge_evaluate_records_read_tool_and_scores_unsolved_task() -> None:
    bridge = TelecomPiBridge()
    task = get_tasks("small")[0]

    bridge.load_task(task.id)
    bridge.call_tool("test-call", "check_status_bar", {})
    result = bridge.evaluate()

    assert result["task_id"] == task.id
    assert result["n_tool_calls"] == 1
    assert result["n_tool_errors"] == 0
    assert result["tool_calls"][0]["tool_name"] == "check_status_bar"
    assert result["reward"] == 0.0
    assert result["reward_basis"] == ["ENV_ASSERTION"]
    assert any(not item["met"] for item in result["env_assertions"])


def test_bridge_evaluate_passes_after_golden_actions() -> None:
    bridge = TelecomPiBridge()
    task = get_tasks("small")[0]
    actions = task.evaluation_criteria.actions or []
    assert actions, "small[0] should have a reference action list"

    bridge.load_task(task.id)
    for index, action in enumerate(actions):
        result = bridge.call_tool(f"gold-{index}", action.name, action.arguments)
        assert result["error"] is False, result["content"]

    scored = bridge.evaluate()
    assert scored["reward"] == 1.0
    assert all(item["met"] for item in scored["env_assertions"])
