# core/registry.py

from __future__ import annotations
import asyncio
from typing import Dict, List, Optional, TYPE_CHECKING

from core.protocol import (
    AgentHandle, AgentSpec, AgentStatus, AgentTask, TaskResult,
)

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent


class AgentRegistry:
    """
    子 Agent 的生命周期管理器。

    职责：
      1. 注册新 Agent（create）
      2. 按 ID 查找 Agent（get）
      3. 按状态筛选 Agent（list_idle / list_finished）
      4. 向已有 Agent 派发任务（send_task）
      5. 收集已完成 Agent 的结果（collect_results）
      6. 清理 Agent 资源（teardown_all）
    """

    def __init__(self):
        # handle 表：agent_id → AgentHandle（轻量数据，供主 Agent 读取）
        self._handles: Dict[str, AgentHandle] = {}
        # agent 表：agent_id → BaseAgent（实际对象，供 Registry 调用）
        self._agents:  Dict[str, "BaseAgent"] = {}

    # ── 注册 ───────────────────────────────────────────────────────────────────

    def register(self, agent: "BaseAgent") -> AgentHandle:
        """
        注册一个已创建的 Agent。

        注意：Registry 不负责创建 Agent 对象，
        创建由 MasterAgent 完成，创建后调用 register() 登记。
        这样 Registry 不需要知道 WorkerAgent 的构造细节。
        """
        handle = agent.handle
        self._handles[handle.agent_id] = handle
        self._agents[handle.agent_id]  = agent
        return handle

    # ── 查找 ───────────────────────────────────────────────────────────────────

    def get_handle(self, agent_id: str) -> AgentHandle:
        """按 ID 获取 handle，找不到就报错"""
        if agent_id not in self._handles:
            raise KeyError(f"Agent {agent_id!r} 不在 Registry 中")
        return self._handles[agent_id]

    def get_agent(self, agent_id: str) -> "BaseAgent":
        """按 ID 获取 Agent 对象"""
        if agent_id not in self._agents:
            raise KeyError(f"Agent {agent_id!r} 不在 Registry 中")
        return self._agents[agent_id]

    # ── 筛选 ───────────────────────────────────────────────────────────────────

    def list_all(self) -> List[AgentHandle]:
        return list(self._handles.values())

    def list_by_status(self, status: AgentStatus) -> List[AgentHandle]:
        return [h for h in self._handles.values() if h.status == status]

    def list_idle(self) -> List[AgentHandle]:
        return self.list_by_status(AgentStatus.IDLE)

    def list_finished(self) -> List[AgentHandle]:
        return self.list_by_status(AgentStatus.FINISHED)

    # ── 派发任务 ───────────────────────────────────────────────────────────────

    async def send_task(self, agent_id: str, task: AgentTask) -> TaskResult:
        """
        向指定 Agent 派发任务并等待结果。

        这是"主 Agent 向已有子 Agent 发送任务"的核心实现。

        流程：
          1. 找到 Agent 对象
          2. 检查它是否空闲
          3. 把任务记录进 handle.task_history
          4. 调用 agent.run(task)
          5. 把结果记录进 handle.result_history
          6. 返回结果
        """
        handle = self.get_handle(agent_id)
        agent  = self.get_agent(agent_id)

        if not handle.is_available():
            raise RuntimeError(
                f"Agent {agent_id!r} 当前状态为 {handle.status.value!r}，"
                f"无法接收新任务"
            )

        # 记录任务历史（在 run 之前记录，方便审查）
        handle.task_history.append(task)

        # 执行任务（agent.run() 内部会改 handle.status）
        result = await agent.run(task)

        # 记录结果历史
        handle.result_history.append(result)

        return result

    # ── 并发派发 ───────────────────────────────────────────────────────────────

    async def dispatch_all(
        self,
        assignments: List[tuple[str, AgentTask]],
    ) -> List[TaskResult]:
        """
        并发向多个 Agent 派发任务，等待全部完成。

        assignments: [(agent_id, task), (agent_id, task), ...]

        用 asyncio.gather() 实现并发，所有子 Agent 同时跑，
        不是串行等待。
        """
        coroutines = [
            self.send_task(agent_id, task)
            for agent_id, task in assignments
        ]
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # 把异常转换成 error 状态的 TaskResult，不让单个失败影响整体
        cleaned: List[TaskResult] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                agent_id, task = assignments[i]
                cleaned.append(TaskResult(
                    task_id       = task.task_id,
                    agent_id      = agent_id,
                    status        = "error",
                    error_message = str(r),
                ))
            else:
                cleaned.append(r)

        return cleaned

    # ── 收集结果 ───────────────────────────────────────────────────────────────

    def collect_results(self) -> List[TaskResult]:
        """
        收集所有已完成 Agent 的最新结果。

        只看 result_history 的最后一条——
        同一个 Agent 可能执行了多个任务，主 Agent 关心最近一次。
        """
        results = []
        for handle in self.list_finished():
            if handle.result_history:
                results.append(handle.result_history[-1])
        return results

    # ── 清理 ───────────────────────────────────────────────────────────────────

    async def teardown_all(self) -> None:
        """
        销毁所有 Agent，释放资源（如临时覆盖率文件）。
        主 Agent 在整个测试流程结束后调用。
        """
        for agent in self._agents.values():
            await agent.teardown()
        self._handles.clear()
        self._agents.clear()

    # ── 调试用 ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status_count: Dict[str, int] = {}
        for h in self._handles.values():
            key = h.status.value
            status_count[key] = status_count.get(key, 0) + 1
        return f"AgentRegistry({status_count})"