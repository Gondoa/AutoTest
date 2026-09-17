# agents/master_agent.py

from __future__ import annotations
import asyncio
import json
import os
from typing import Annotated, Any, Dict, List, Optional

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from core.protocol import (
    AgentHandle, AgentSpec, AgentStatus,
    AgentTask, TaskResult, DecisionContext,
)
from core.llm_client import LLMClient, Message
from core.registry import AgentRegistry
from agents.base_agent import BaseAgent
from agents.worker_agent import WorkerAgent
from targets.fastapi_app.app import create_app
from tools.coverage_tool import CoverageTool


class MasterState(TypedDict):
    """主 Agent LangGraph 图的状态"""

    # 整体测试目标（自然语言，从 run() 传入）
    objective: str

    # 当前轮次
    round_number: int
    max_rounds:   int

    # 本轮 LLM 规划出的 AgentSpec 列表（create 节点消费）
    pending_specs: List[Dict]

    # 本轮待派发的任务列表：[(agent_id, AgentTask), ...]
    pending_assignments: List[Dict]

    # 本轮执行结果
    latest_results: List[Dict]

    # 决策信号："need_more" | "need_new" | "done"
    decision: str

    # 最终报告（summarize 节点生成）
    final_report: Optional[Dict]

class MasterAgent(BaseAgent):
    """
    主 Agent：控制测试流程的 LangGraph 图。

    职责：
      1. 规划测试策略（plan）
      2. 创建子 Agent（create）
      3. 向已有子 Agent 派任务（dispatch）
      4. 并发运行子 Agent（run_workers）
      5. 审查结果（review）
      6. 决策下一步（decide）
    """

    def __init__(
        self,
        handle:      AgentHandle,
        source_path: str,           # 被测文件路径
        max_rounds:  int  = 5,
        offline:     bool = False,
    ):
        super().__init__(handle)

        self.source_path = source_path
        self.max_rounds  = max_rounds
        self.offline     = offline

        self.registry = AgentRegistry()
        self.llm      = LLMClient(offline=offline)

        self._graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(MasterState)

        builder.add_node("plan",        self._plan_node)
        builder.add_node("create",      self._create_node)
        builder.add_node("dispatch",    self._dispatch_node)
        builder.add_node("run_workers", self._run_workers_node)
        builder.add_node("review",      self._review_node)
        builder.add_node("decide",      self._decide_node)
        builder.add_node("summarize",   self._summarize_node)

        builder.add_edge(START,        "plan")
        builder.add_edge("plan",       "create")
        builder.add_edge("create",     "dispatch")
        builder.add_edge("dispatch",   "run_workers")
        builder.add_edge("run_workers","review")
        builder.add_edge("review",     "decide")

        # 条件边：decide 节点决定走哪条路
        builder.add_conditional_edges(
            "decide",
            self._route,
            {
                "need_more": "dispatch",   # 向已有 Agent 补派任务
                "need_new":  "create",     # 创建新 Agent
                "done":      "summarize",
            },
        )

        builder.add_edge("summarize", END)

        return builder.compile()

    async def _plan_node(self, state: MasterState) -> Dict:
        """
        让 LLM 根据测试目标，规划需要哪些 Agent 类型和规则。
        输出 pending_specs 供 create 节点消费。
        """
        prompt = (
            f"你是一个软件测试主管。\n"
            f"测试目标：{state['objective']}\n\n"
            f"请规划需要创建哪些测试 Agent，每个 Agent 负责不同的测试维度。\n"
            f"返回 JSON 数组，每个元素包含：\n"
            f"  agent_type: 类型标识（如 functional/boundary/state）\n"
            f"  objective:  该 Agent 的具体目标\n"
            f"  prompt:     注入给 Agent 的系统提示\n"
            f"  allowed_paths: 允许访问的路径列表\n"
            f"  rules:      该 Agent 必须验证的规则列表\n"
            f"  max_steps:  最大步数（建议 5-15）\n"
        )

        offline_reply = [
            {
                "agent_type":    "functional",
                "objective":     "验证正常读写功能",
                "prompt":        "你是功能测试 Agent，验证 API 的正常读写行为。",
                "allowed_paths": ["/items/foo", "/items/bar"],
                "rules": [
                    "GET 已有 item 返回 200 和正确数据",
                    "PUT 有效数据后 GET 能读回更新后的值",
                ],
                "max_steps": 8,
            },
            {
                "agent_type":    "boundary",
                "objective":     "验证边界和错误处理",
                "prompt":        "你是边界测试 Agent，验证 API 的错误处理行为。",
                "allowed_paths": ["/items/foo", "/items/nonexistent"],
                "rules": [
                    "GET 不存在的 item 返回 404",
                    "PUT 无效数据（price 为字符串）返回 422",
                ],
                "max_steps": 8,
            },
            {
                "agent_type":    "state",
                "objective":     "验证状态一致性",
                "prompt":        "你是状态测试 Agent，验证无效操作不会破坏已有数据。",
                "allowed_paths": ["/items/foo"],
                "rules": [
                    "PUT 无效数据后，原数据保持不变",
                ],
                "max_steps": 6,
            },
        ]

        specs = await self.llm.chat_json(
            messages=[Message(role="user", content=prompt)],
            offline_reply=offline_reply,
        )

        return {"pending_specs": specs, "round_number": 1}

    async def _create_node(self, state: MasterState) -> Dict:
        """
        根据 pending_specs 创建 WorkerAgent，注册进 Registry。

        这是"主 Agent 创建子 Agent，分配权限、工具列表和项目信息"的实现。
        """
        for spec_dict in state["pending_specs"]:
            # 构造 AgentSpec（权限和规则在这里确定，子 Agent 无法自己修改）
            spec = AgentSpec(
                agent_type    = spec_dict["agent_type"],
                objective     = spec_dict["objective"],
                prompt        = spec_dict["prompt"],
                allowed_paths = spec_dict.get("allowed_paths", []),
                allowed_tools = ["http_request", "finish"],   # 主 Agent 决定工具列表
                rules         = spec_dict.get("rules", []),
                max_steps     = spec_dict.get("max_steps", 10),
            )

            # 构造 AgentHandle
            handle = AgentHandle(spec=spec)

            # 为这个 Agent 创建独立的 App 实例（测试隔离）
            app = create_app()

            # 创建 WorkerAgent，传入项目信息（source_path）和权限（spec）
            worker = WorkerAgent(
                handle      = handle,
                app         = app,
                source_path = self.source_path,
                offline     = self.offline,
            )

            # 注册进 Registry
            self.registry.register(worker)

        # 消费完毕，清空 pending_specs
        return {"pending_specs": []}

    async def _dispatch_node(self, state: MasterState) -> Dict:
        """
        向 Registry 里所有 idle Agent 派发任务。

        这是"主 Agent 向已有子 Agent 发送任务"的实现。
        """
        idle_handles = self.registry.list_idle()

        assignments = []
        for handle in idle_handles:
            task = AgentTask(
                agent_id    = handle.agent_id,
                instruction = (
                    f"请验证以下规则：{json.dumps(handle.spec.rules, ensure_ascii=False)}\n"
                    f"测试目标：{handle.spec.objective}"
                ),
                context = {
                    "round":       state["round_number"],
                    "objective":   state["objective"],
                },
            )
            assignments.append({
                "agent_id": handle.agent_id,
                "task":     task.model_dump(),
            })

        return {"pending_assignments": assignments}

    async def _run_workers_node(self, state: MasterState) -> Dict:
        """
        并发执行所有待派发的任务，等待全部完成。
        """
        assignments = [
            (a["agent_id"], AgentTask(**a["task"]))
            for a in state["pending_assignments"]
        ]

        # asyncio.gather 并发运行，Registry 处理异常转换
        results = await self.registry.dispatch_all(assignments)

        return {
            "latest_results":    [r.model_dump() for r in results],
            "pending_assignments": [],    # 消费完毕
        }

    async def _review_node(self, state: MasterState) -> Dict:
        """
        让 LLM 审查本轮结果，判断覆盖是否足够，指出缺口。
        """
        results      = state["latest_results"]
        all_handles  = self.registry.list_all()

        # 汇总覆盖率
        coverage_reports = [
            r.get("coverage_hint", {})
            for r in results
            if r.get("coverage_hint")
        ]
        avg_line = (
            sum(r.get("line_coverage", 0) for r in coverage_reports)
            / len(coverage_reports)
        ) if coverage_reports else 0.0

        all_missing = []
        for r in coverage_reports:
            all_missing.extend(r.get("missing_lines", []))

        review_prompt = (
            f"你是测试主管，请审查本轮测试结果：\n"
            f"测试目标：{state['objective']}\n"
            f"本轮结果：{json.dumps(results, ensure_ascii=False)}\n"
            f"行覆盖率：{avg_line:.1%}\n"
            f"未覆盖行：{sorted(set(all_missing))}\n"
            f"当前轮次：{state['round_number']} / {state['max_rounds']}\n\n"
            f"请判断：\n"
            f"1. 测试是否全面？\n"
            f"2. 有哪些规则未验证或验证不充分？\n"
            f"3. 建议下一步：done / need_more（复用已有 Agent）/ need_new（创建新 Agent）\n"
            f"4. 如果 need_more：给出具体补充指令（发给哪个 agent_type 的 Agent）\n"
            f"4. 如果 need_new：给出新 AgentSpec（同 plan 节点的格式）\n\n"
            f"返回 JSON：\n"
            f'{{"decision": "...", "reason": "...", '
            f'"supplemental_instructions": [...], "new_specs": [...]}}'
        )

        offline_reply = {
            "decision": "done",
            "reason":   "离线模式，单轮即完成",
            "supplemental_instructions": [],
            "new_specs": [],
        }

        review = await self.llm.chat_json(
            messages=[Message(role="user", content=review_prompt)],
            offline_reply=offline_reply,
        )

        return {"_review": review}

    async def _decide_node(self, state: MasterState) -> Dict:
        """
        根据 review 结果决定下一步。
        同时处理补充任务的准备工作。
        """
        review       = state.get("_review", {})
        decision     = review.get("decision", "done")
        round_number = state["round_number"]

        # 预算耗尽强制结束
        if round_number >= state["max_rounds"]:
            return {"decision": "done"}

        if decision == "need_more":
            # 向已有 idle Agent 补派任务
            supplemental = review.get("supplemental_instructions", [])
            assignments  = []

            for instruction in supplemental:
                target_type = instruction.get("agent_type")
                # 找到对应类型的 idle Agent
                idle = [
                    h for h in self.registry.list_idle()
                    if h.spec.agent_type == target_type
                ]
                if idle:
                    task = AgentTask(
                        agent_id    = idle[0].agent_id,
                        instruction = instruction.get("instruction", ""),
                        context     = {"round": round_number + 1},
                    )
                    assignments.append({
                        "agent_id": idle[0].agent_id,
                        "task":     task.model_dump(),
                    })

            return {
                "decision":           "need_more",
                "pending_assignments": assignments,
                "round_number":        round_number + 1,
            }

        elif decision == "need_new":
            # 准备新的 AgentSpec，交给 create 节点处理
            new_specs = review.get("new_specs", [])
            return {
                "decision":     "need_new",
                "pending_specs": new_specs,
                "round_number":  round_number + 1,
            }

        else:
            return {"decision": "done"}

    def _route(self, state: MasterState) -> str:
        return state.get("decision", "done")

    async def _summarize_node(self, state: MasterState) -> Dict:
        """
        汇总所有轮次的结果，生成最终测试报告。
        """
        all_results = []
        for handle in self.registry.list_all():
            all_results.extend(handle.result_history)

        # 统计
        passed   = sum(1 for r in all_results if r.status == "passed")
        failed   = sum(1 for r in all_results if r.status == "failed")
        errors   = sum(1 for r in all_results if r.status == "error")

        all_rule_results = [
            rr for r in all_results for rr in r.rule_results
        ]
        rules_passed = sum(1 for rr in all_rule_results if rr.passed)
        rules_failed = sum(1 for rr in all_rule_results if not rr.passed)

        coverage_reports = [
            r.coverage_hint for r in all_results if r.coverage_hint
        ]
        avg_line = (
            sum(r["line_coverage"] for r in coverage_reports)
            / len(coverage_reports)
        ) if coverage_reports else 0.0

        final_report = {
            "status":          "passed" if rules_failed == 0 else "failed",
            "rounds":          state["round_number"],
            "agents_created":  len(self.registry.list_all()),
            "tasks_executed":  len(all_results),
            "rules_passed":    rules_passed,
            "rules_failed":    rules_failed,
            "task_passed":     passed,
            "task_failed":     failed,
            "task_errors":     errors,
            "avg_line_coverage": round(avg_line, 2),
            "agent_details":   [
                {
                    "agent_id":   h.agent_id[:8],
                    "type":       h.spec.agent_type,
                    "tasks_done": len(h.result_history),
                    "rules":      h.spec.rules,
                }
                for h in self.registry.list_all()
            ],
        }

        return {"final_report": final_report}

    async def run(self, task: AgentTask) -> TaskResult:
        """
        BaseAgent 要求实现的方法。
        对主 Agent 来说，task.instruction 就是整体测试目标。
        """
        self._set_status(AgentStatus.RUNNING)

        try:
            initial_state: MasterState = {
                "objective":           task.instruction,
                "round_number":        0,
                "max_rounds":          self.max_rounds,
                "pending_specs":       [],
                "pending_assignments": [],
                "latest_results":      [],
                "decision":            "need_new",
                "final_report":        None,
            }

            final_state = await self._graph.ainvoke(initial_state)
            report      = final_state.get("final_report", {})

            result = TaskResult(
                task_id      = task.task_id,
                agent_id     = self.agent_id,
                status       = report.get("status", "error"),
                tool_trace   = [],
                coverage_hint = {
                    "line_coverage": report.get("avg_line_coverage", 0.0),
                },
            )

        except Exception as e:
            result = TaskResult(
                task_id       = task.task_id,
                agent_id      = self.agent_id,
                status        = "error",
                error_message = str(e),
            )

        finally:
            await self.registry.teardown_all()

        self._set_status(AgentStatus.FINISHED)
        return result

    async def teardown(self) -> None:
        await self.registry.teardown_all()