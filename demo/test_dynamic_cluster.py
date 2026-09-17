"""Tests for the demonstrable dynamic-agent protocol."""

import pytest
import httpx

from agent_protocol import AgentSpec, AgentTask, ToolCall, WorkerResult
import dynamic_cluster
from dynamic_cluster import PROJECT_CONTEXT, _split_rules_for_worker_budget, _validate_specs, create_ephemeral_agent, run_dynamic_cluster, send_task


@pytest.mark.asyncio
async def test_dynamic_cluster_creates_independent_specialists():
    report = await run_dynamic_cluster("测试 item API 的功能、边界和状态一致性；自动创建临时探测 Agent", force_offline=True)
    assert report["mode"] == "offline"
    assert report["status"] == "passed"
    assert report["rounds"] == 1
    assert sorted(report["agents_created"]) == ["boundary", "functional", "state", "temporary_contract_probe"]
    assert report["missing_rules"] == []
    assert report["metrics"]["scenario_coverage"]["percentage"] == 100.0
    assert report["metrics"]["exception_coverage"]["percentage"] == 100.0
    assert report["metrics"]["code_coverage"]["total_lines"] > 0
    assert report["metrics"]["code_coverage"]["covered_lines"] > 0
    probe = next(result for result in report["results"] if result["agent_type"] == "temporary_contract_probe")
    assert [entry["action"]["method"] for entry in probe["trace"]] == ["GET"]


def test_protocol_allows_ephemeral_agent_but_rejects_arbitrary_tool():
    spec = _validate_specs([{
        "agent_type": "temporary_probe_01", "objective": "probe", "rules": ["read_existing_item"]
    }])[0]
    spec = spec.model_copy(update={"project": PROJECT_CONTEXT})
    assert create_ephemeral_agent(spec).agent_type == "temporary_probe_01"
    with pytest.raises(ValueError):
        ToolCall.model_validate({"tool": "run_shell", "reason": "execute"})


def test_protocol_has_structured_result_models():
    spec = AgentSpec(agent_type="state", objective="check state", rules=["invalid_update_preserves_state"])
    result = WorkerResult(task_id="task-1", agent_type=spec.agent_type, objective=spec.objective,
                          status="incomplete", rules=[], trace=[])
    assert result.model_dump()["status"] == "incomplete"


@pytest.mark.asyncio
async def test_supervisor_can_send_a_follow_up_to_an_existing_worker():
    spec = AgentSpec(agent_type="functional", objective="initial objective", project=PROJECT_CONTEXT,
                     rules=["read_existing_item", "valid_update_persists"])
    task = AgentTask(task_id="follow-up-1", agent_type="functional", objective="only recheck read",
                     rules=["read_existing_item"])
    async with __import__("httpx").AsyncClient() as client:
        result = await send_task(spec, task, client, online=False)
    assert result.task_id == "follow-up-1"
    assert result.objective == "only recheck read"
    assert result.status == "passed"


@pytest.mark.asyncio
async def test_worker_cannot_use_a_write_tool_without_write_permission():
    spec = AgentSpec(agent_type="functional", objective="read only", project=PROJECT_CONTEXT,
                     permissions=["read"], rules=["valid_update_persists"])
    task = AgentTask(task_id="read-only-1", agent_type="functional", objective="attempt update",
                     rules=["valid_update_persists"])
    async with __import__("httpx").AsyncClient() as client:
        result = await send_task(spec, task, client, online=False)
    assert result.status == "failed"
    assert any(entry["result"]["result"].get("error") == "Permission not granted"
               for entry in result.trace)


@pytest.mark.asyncio
async def test_online_supervisor_transport_failure_is_a_structured_error(monkeypatch):
    monkeypatch.setenv("PROJECT_GUIDE_LLM_API_KEY", "fake-test-key")

    async def unavailable(*args, **kwargs):
        raise httpx.ConnectError("network unavailable")

    monkeypatch.setattr(dynamic_cluster, "_supervisor_specs", unavailable)
    report = await run_dynamic_cluster("test online error")
    assert report["mode"] == "online"
    assert report["status"] == "error"
    assert report["error"] == "Supervisor request failed: ConnectError"


def test_supervisor_splits_exception_rules_to_fit_worker_budget():
    assert _split_rules_for_worker_budget([
        "read_missing_item", "invalid_update_rejected", "invalid_update_preserves_state",
    ]) == [
        ["invalid_update_rejected", "invalid_update_preserves_state"],
        ["read_missing_item"],
    ]
