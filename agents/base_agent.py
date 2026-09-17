# agents/base_agent.py

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from core.protocol import AgentHandle, AgentSpec, AgentTask, AgentStatus, TaskResult


class BaseAgent(ABC):
    """
    所有 Agent 的抽象基类。

    定义两件事：
      1. 所有 Agent 共有的属性（handle、spec）
      2. 子类必须实现的方法（run）
    """

    def __init__(self, handle: AgentHandle):
        """
        handle 由 Registry 创建并传入，
        Agent 不自己生成 agent_id，由 Registry 统一管理。
        """
        self.handle: AgentHandle = handle

    # ── 便捷属性 ───────────────────────────────────────────────────────────────

    @property
    def agent_id(self) -> str:
        return self.handle.agent_id

    @property
    def spec(self) -> AgentSpec:
        return self.handle.spec

    @property
    def status(self) -> AgentStatus:
        return self.handle.status

    # ── 状态管理 ───────────────────────────────────────────────────────────────

    def _set_status(self, status: AgentStatus) -> None:
        """
        更新自身状态，同时写回 handle。
        handle 是 Registry 里存的同一个对象，
        所以 Registry 能直接看到状态变化。
        """
        self.handle.status = status

    # ── 子类必须实现 ───────────────────────────────────────────────────────────

    @abstractmethod
    async def run(self, task: AgentTask) -> TaskResult:
        """
        接收一个任务，执行，返回结果。

        实现要求：
          - 执行前把状态设为 RUNNING
          - 执行后把状态设为 FINISHED 或 FAILED
          - 无论成功失败都要返回 TaskResult，不能抛异常
          - 子 Agent 可以多次被调用（可重用）
        """
        ...

    # ── 可选覆盖 ───────────────────────────────────────────────────────────────

    async def teardown(self) -> None:
        """
        Agent 生命周期结束时的清理工作。
        子类按需覆盖，比如 WorkerAgent 用它来调用 coverage_tool.cleanup()。
        默认什么都不做。
        """
        pass

    # ── 调试用 ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.agent_id[:8]}..., "
            f"type={self.spec.agent_type!r}, "
            f"status={self.status.value!r})"
        )