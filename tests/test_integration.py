# tests/test_integration.py

import pytest
import asyncio
from core.protocol import AgentHandle, AgentSpec, AgentTask
from agents.master_agent import MasterAgent
from target.fastapi_app.app import create_app
from tools.http_tool import HttpTool
from core.protocol import ToolCall
import os

SOURCE_PATH = os.path.abspath("targets/fastapi_app/app.py")


@pytest.mark.asyncio
async def test_full_cluster_offline():
    """
    完整离线集群测试：
    从 MasterAgent.run() 开始，验证整个链路能跑通。
    """
    spec   = AgentSpec(agent_type="master", objective="test", prompt="test")
    handle = AgentHandle(spec=spec)
    master = MasterAgent(
        handle=handle, source_path=SOURCE_PATH, max_rounds=2, offline=True
    )
    task   = AgentTask(
        agent_id    = master.agent_id,
        instruction = "测试 FastAPI item CRUD API",
    )
    result = await master.run(task)

    # 整条链路不报 error
    assert result.status in ("passed", "failed", "incomplete")


@pytest.mark.asyncio
async def test_target_app_factory_isolation():
    """
    create_app() 工厂函数的隔离性：
    两个实例的写操作互不干扰。
    """
    app1 = create_app()
    app2 = create_app()

    tool1 = HttpTool(app1, [])
    tool2 = HttpTool(app2, [])

    # 在 app1 里修改 foo 的价格
    await tool1.execute(ToolCall(
        tool="http_request",
        arguments={"method": "PUT", "path": "/items/foo",
                   "body": {"name": "Modified", "price": 999.0}},
    ))

    # app2 里的 foo 应该不受影响
    result = await tool2.execute(ToolCall(
        tool="http_request",
        arguments={"method": "GET", "path": "/items/foo"},
    ))

    assert result.status_code == 200
    assert result.body["price"] == 50.2   # 原始值


@pytest.mark.asyncio
async def test_llm_client_offline_mode():
    """LLMClient 离线模式不发网络请求"""
    from core.llm_client import LLMClient, Message
    llm    = LLMClient(offline=True)
    result = await llm.chat(
        messages      = [Message(role="user", content="test")],
        offline_reply = "offline_response",
    )
    assert result == "offline_response"


@pytest.mark.asyncio
async def test_worker_sends_results_to_master():
    """
    子 Agent 的结果最终能被主 Agent 收集到：
    验证 Registry 的 result_history 被正确填充。
    """
    from core.registry import AgentRegistry
    from agents.worker_agent import WorkerAgent
    from core.protocol import AgentHandle, AgentSpec

    registry = AgentRegistry()
    spec     = AgentSpec(
        agent_type    = "functional",
        objective     = "test",
        prompt        = "test",
        allowed_paths = ["/items/foo"],
        allowed_tools = ["http_request", "finish"],
        rules         = ["GET /items/foo 返回 200"],
        max_steps     = 5,
    )
    handle = AgentHandle(spec=spec)
    worker = WorkerAgent(
        handle=handle, app=create_app(), source_path=SOURCE_PATH, offline=True
    )
    registry.register(worker)

    task   = AgentTask(agent_id=handle.agent_id, instruction="验证 GET /items/foo")
    result = await registry.send_task(handle.agent_id, task)

    assert len(handle.result_history) == 1
    assert handle.result_history[0].task_id == task.task_id