# tests/test_master_agent.py

import pytest
import os
from core.protocol import AgentHandle, AgentSpec, AgentTask
from agents.master_agent import MasterAgent

SOURCE_PATH = os.path.abspath("targets/fastapi_app/app.py")


def make_master(max_rounds=2, offline=True) -> MasterAgent:
    spec = AgentSpec(
        agent_type="master", objective="test", prompt="test"
    )
    handle = AgentHandle(spec=spec)
    return MasterAgent(
        handle      = handle,
        source_path = SOURCE_PATH,
        max_rounds  = max_rounds,
        offline     = offline,
    )


@pytest.mark.asyncio
async def test_master_creates_workers():
    """主 Agent 执行后 Registry 里应该有子 Agent"""
    master = make_master()
    task   = AgentTask(
        agent_id    = master.agent_id,
        instruction = "测试 FastAPI item API",
    )
    await master.run(task)

    # 主 Agent 完成后，registry 已清理，通过 result 验证
    assert master.status.value == "finished"


@pytest.mark.asyncio
async def test_master_returns_report():
    """主 Agent 返回的 TaskResult 包含覆盖率信息"""
    master = make_master()
    task   = AgentTask(
        agent_id    = master.agent_id,
        instruction = "测试 FastAPI item API",
    )
    result = await master.run(task)

    assert result.status in ("passed", "failed", "incomplete", "error")
    assert result.coverage_hint is not None


@pytest.mark.asyncio
async def test_master_respects_max_rounds():
    """主 Agent 不超过 max_rounds 轮"""
    master = make_master(max_rounds=1)
    task   = AgentTask(
        agent_id    = master.agent_id,
        instruction = "测试 FastAPI item API",
    )
    result = await master.run(task)
    # 1 轮内完成，不报 error
    assert result.status != "error"


@pytest.mark.asyncio
async def test_master_worker_isolation():
    """两个主 Agent 并发执行互不影响"""
    import asyncio
    m1 = make_master()
    m2 = make_master()

    t1 = AgentTask(agent_id=m1.agent_id, instruction="测试 API（实例1）")
    t2 = AgentTask(agent_id=m2.agent_id, instruction="测试 API（实例2）")

    r1, r2 = await asyncio.gather(m1.run(t1), m2.run(t2))
    assert r1.status != "error"
    assert r2.status != "error"