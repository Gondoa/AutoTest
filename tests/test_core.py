# tests/test_core.py

import pytest
from core.protocol import (
    AgentHandle, AgentSpec, AgentStatus,
    AgentTask, TaskResult, ToolCall, ToolResult,
)
from core.registry import AgentRegistry


# ── protocol 测试 ──────────────────────────────────────────────────────────────

def test_tool_call_rejects_unknown_tool():
    """ToolCall 拒绝白名单之外的工具名"""
    with pytest.raises(Exception):
        ToolCall(tool="shell_exec", arguments={})


def test_tool_call_requires_method_for_http():
    """http_request 工具必须带 method 和 path"""
    with pytest.raises(Exception):
        ToolCall(tool="http_request", arguments={"path": "/items/foo"})


def test_agent_spec_rejects_invalid_type():
    """agent_type 不能包含特殊字符"""
    with pytest.raises(Exception):
        AgentSpec(
            agent_type    = "../../etc/passwd",
            objective     = "test",
            prompt        = "test",
        )


def test_agent_handle_is_available():
    """新创建的 handle 默认是 IDLE 状态"""
    spec   = AgentSpec(agent_type="functional", objective="test", prompt="test")
    handle = AgentHandle(spec=spec)
    assert handle.is_available() is True
    handle.status = AgentStatus.RUNNING
    assert handle.is_available() is False


# ── registry 测试 ──────────────────────────────────────────────────────────────

class FakeAgent:
    """最小化的 fake Agent，用于测试 Registry 逻辑"""
    def __init__(self, handle):
        self.handle = handle

    async def run(self, task):
        self.handle.status = AgentStatus.FINISHED
        return TaskResult(
            task_id  = task.task_id,
            agent_id = self.handle.agent_id,
            status   = "passed",
        )

    async def teardown(self):
        pass


def make_fake_agent(agent_type: str = "functional") -> FakeAgent:
    spec   = AgentSpec(agent_type=agent_type, objective="test", prompt="test")
    handle = AgentHandle(spec=spec)
    return FakeAgent(handle)


def test_registry_register_and_get():
    registry = AgentRegistry()
    agent    = make_fake_agent()
    handle   = registry.register(agent)

    assert registry.get_handle(handle.agent_id) is handle


def test_registry_list_idle():
    registry = AgentRegistry()
    agent    = make_fake_agent()
    registry.register(agent)

    assert len(registry.list_idle()) == 1


@pytest.mark.asyncio
async def test_registry_send_task():
    registry = AgentRegistry()
    agent    = make_fake_agent()
    handle   = registry.register(agent)

    task = AgentTask(agent_id=handle.agent_id, instruction="test")
    result = await registry.send_task(handle.agent_id, task)

    assert result.status == "passed"
    assert len(handle.task_history)   == 1
    assert len(handle.result_history) == 1


@pytest.mark.asyncio
async def test_registry_rejects_busy_agent():
    """向 RUNNING 状态的 Agent 发任务应该报错"""
    registry = AgentRegistry()
    agent    = make_fake_agent()
    handle   = registry.register(agent)
    handle.status = AgentStatus.RUNNING

    task = AgentTask(agent_id=handle.agent_id, instruction="test")
    with pytest.raises(RuntimeError):
        await registry.send_task(handle.agent_id, task)