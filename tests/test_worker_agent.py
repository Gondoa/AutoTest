# tests/test_worker_agent.py

import pytest
from core.protocol import AgentHandle, AgentSpec, AgentTask
from agents.worker_agent import WorkerAgent
from targets.fastapi_app.app import create_app
import os

SOURCE_PATH = os.path.abspath("targets/fastapi_app/app.py")


def make_worker(agent_type="functional", rules=None, offline=True) -> WorkerAgent:
    spec = AgentSpec(
        agent_type    = agent_type,
        objective     = "测试 item API",
        prompt        = "你是测试 Agent。",
        allowed_paths = ["/items/foo", "/items/bar", "/items/nonexistent"],
        allowed_tools = ["http_request", "finish"],
        rules         = rules or ["GET /items/foo 返回 200"],
        max_steps     = 6,
    )
    handle = AgentHandle(spec=spec)
    return WorkerAgent(
        handle      = handle,
        app         = create_app(),
        source_path = SOURCE_PATH,
        offline     = offline,
    )


@pytest.mark.asyncio
async def test_worker_returns_task_result():
    """子 Agent 执行后必须返回 TaskResult"""
    worker = make_worker()
    task   = AgentTask(agent_id=worker.agent_id, instruction="验证 GET /items/foo")
    result = await worker.run(task)

    assert result.task_id  == task.task_id
    assert result.agent_id == worker.agent_id
    assert result.status   in ("passed", "failed", "incomplete", "error")


@pytest.mark.asyncio
async def test_worker_status_lifecycle():
    """子 Agent 执行完后状态应该变为 FINISHED"""
    from core.protocol import AgentStatus
    worker = make_worker()
    task   = AgentTask(agent_id=worker.agent_id, instruction="test")

    assert worker.status == AgentStatus.IDLE
    await worker.run(task)
    assert worker.status == AgentStatus.FINISHED


@pytest.mark.asyncio
async def test_worker_isolation():
    """两个子 Agent 的写操作互不影响"""
    worker1 = make_worker("w1")
    worker2 = make_worker("w2")

    # worker1 写入新数据
    task1 = AgentTask(
        agent_id    = worker1.agent_id,
        instruction = "PUT /items/foo with price=999",
    )
    # worker2 读取，应该看到原始数据
    task2 = AgentTask(
        agent_id    = worker2.agent_id,
        instruction = "GET /items/foo，验证 price 是 50.2",
    )

    import asyncio
    r1, r2 = await asyncio.gather(worker1.run(task1), worker2.run(task2))

    # 两个都应该完成，不报 error
    assert r1.status != "error"
    assert r2.status != "error"


@pytest.mark.asyncio
async def test_worker_reusable():
    """同一个子 Agent 可以被调用多次"""
    worker = make_worker()

    task1 = AgentTask(agent_id=worker.agent_id, instruction="第一个任务")
    task2 = AgentTask(agent_id=worker.agent_id, instruction="第二个任务")

    r1 = await worker.run(task1)
    r2 = await worker.run(task2)   # 第二次调用同一个 worker

    assert r1.status != "error"
    assert r2.status != "error"


@pytest.mark.asyncio
async def test_worker_permission_blocks_path():
    """子 Agent 不能访问白名单外的路径"""
    spec = AgentSpec(
        agent_type    = "restricted",
        objective     = "test",
        prompt        = "test",
        allowed_paths = ["/items/foo"],   # 只允许 /items/foo
        allowed_tools = ["http_request", "finish"],
        rules         = [],
        max_steps     = 3,
    )
    handle = AgentHandle(spec=spec)
    worker = WorkerAgent(
        handle=handle, app=create_app(), source_path=SOURCE_PATH, offline=True
    )

    from tools.http_tool import HttpTool
    from core.protocol import ToolCall
    http_tool = HttpTool(worker.http_tool.app, ["/items/foo"])
    call      = ToolCall(
        tool="http_request",
        arguments={"method": "GET", "path": "/items/bar"},   # bar 不在白名单
    )
    result = await http_tool.execute(call)
    assert result.success is False
    assert "不在允许列表" in result.error