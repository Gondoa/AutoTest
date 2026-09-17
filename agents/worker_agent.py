# agents/worker_agent.py

from __future__ import annotations
import json
from typing import Annotated, Any, Dict, List, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from core.protocol import (
    AgentHandle, AgentTask, AgentStatus,
    TaskResult, RuleResult, ToolCall, ToolResult,
)
from core.llm_client import LLMClient, Message
from core.registry import AgentRegistry
from tools.http_tool import HttpTool
from tools.coverage_tool import CoverageTool
from agents.base_agent import BaseAgent


class WorkerState(TypedDict):
    """
    子 Agent LangGraph 图的状态。
    每个节点读取这个状态，返回要更新的字段。
    """
    # 当前任务
    task:          Optional[AgentTask]

    # LLM 对话历史（Annotated + add_messages 让 LangGraph 自动追加而非覆盖）
    messages:      Annotated[List[Dict], add_messages]

    # 工具调用记录（用于生成 TaskResult.tool_trace）
    tool_trace:    List[Dict[str, Any]]

    # 规则验证结果
    rule_results:  List[RuleResult]

    # 已用步数
    steps:         int

    # 最终动作信号："continue" | "finish" | "budget_exceeded"
    next_action:   str

    # finish 工具带来的最终结论
    final_status:  str   # "passed" | "failed" | "incomplete"

class WorkerAgent(BaseAgent):
    """
    子 Agent：独立的 LangGraph 图实例。

    每个实例绑定：
      - 独立的 FastAPI app（测试隔离）
      - 独立的 HttpTool（权限控制）
      - 独立的 CoverageTool（覆盖率测量）
      - 独立的 LLMClient（离线/在线模式）
    """

    def __init__(
        self,
        handle:        AgentHandle,
        app,                          # FastAPI 实例，从 targets/ 创建
        source_path:   str,           # 被测文件路径，给 CoverageTool 用
        offline:       bool = False,
    ):
        super().__init__(handle)

        self.http_tool     = HttpTool(app, handle.spec.allowed_paths)
        self.coverage_tool = CoverageTool(source_path)
        self.llm           = LLMClient(offline=offline)
        self.offline       = offline

        # 编译 LangGraph 图（只编译一次，可重复执行）
        self._graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(WorkerState)

        # 注册节点
        builder.add_node("receive", self._receive_node)
        builder.add_node("think",   self._think_node)
        builder.add_node("act",     self._act_node)
        builder.add_node("observe", self._observe_node)
        builder.add_node("report",  self._report_node)

        # 连接边
        builder.add_edge(START,     "receive")
        builder.add_edge("receive", "think")
        builder.add_edge("think",   "act")
        builder.add_edge("act",     "observe")

        # 条件边：observe 之后决定继续还是结束
        builder.add_conditional_edges(
            "observe",
            self._route,          # 路由函数
            {
                "continue": "think",     # 继续循环
                "finish":   "report",    # 完成
                "budget_exceeded": "report",  # 预算耗尽
            },
        )

        builder.add_edge("report", END)

        return builder.compile()
    async def _receive_node(self, state: WorkerState) -> Dict:
        """
        接收任务，构造初始对话（system prompt + 第一条 user 消息）。
        不调用 LLM，只是初始化状态。
        """
        task = state["task"]

        system_prompt = self.spec.prompt + "\n\n" + "\n".join([
            "你是一个软件测试 Agent。",
            f"你的目标：{self.spec.objective}",
            f"你需要验证以下规则：{json.dumps(self.spec.rules, ensure_ascii=False)}",
            f"你可以使用的工具：{self.spec.allowed_tools}",
            f"你可以访问的路径：{self.spec.allowed_paths}",
            "",
            "每次回复必须是 JSON 格式，包含 tool 和 arguments 字段：",
            '{"tool": "http_request", "arguments": {"method": "GET", "path": "/items/foo"}}',
            "或者当所有规则验证完毕后：",
            '{"tool": "finish", "arguments": {"status": "passed", "summary": "..."}}',
        ])

        first_user_message = (
            f"任务指令：{task.instruction}\n"
            f"附加上下文：{json.dumps(task.context, ensure_ascii=False)}"
        )

        return {
            "messages": [
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": first_user_message},
            ],
            "tool_trace":   [],
            "rule_results": [],
            "steps":        0,
            "next_action":  "continue",
            "final_status": "incomplete",
        }
    async def _think_node(self, state: WorkerState) -> Dict:
        """
        调用 LLM，决定下一步工具调用。
        返回 LLM 的回复，追加到消息历史。
        """
        messages = [
            Message(role=m["role"], content=m["content"])
            for m in state["messages"]
        ]

        # 离线模式：按规则脚本化决策
        offline_reply = self._offline_think(state)

        raw = await self.llm.chat(
            messages      = messages,
            offline_reply = offline_reply,
        )

        # LLM 的回复追加到对话历史
        return {
            "messages": [{"role": "assistant", "content": raw}],
        }

    async def _act_node(self, state: WorkerState) -> Dict:
        """
        解析 LLM 的最新回复，执行工具调用。
        """
        last_message = state["messages"][-1]["content"]

        # 解析 LLM 输出的 JSON
        try:
            raw_call = json.loads(last_message)
            tool_call = ToolCall(**raw_call)
        except Exception as e:
            # LLM 输出格式不对：记录错误，下一轮让它重试
            error_result = ToolResult(
                tool    = "unknown",
                success = False,
                error   = f"无法解析工具调用：{e}，原始内容：{last_message!r}",
            )
            return {
                "messages": [{
                    "role":    "user",
                    "content": f"工具调用格式错误，请重新输出 JSON：{error_result.error}",
                }],
                "tool_trace": state["tool_trace"] + [{"error": str(e)}],
            }

        # 执行工具
        if tool_call.tool == "finish":
            # finish 不调用 HttpTool，直接提取 status
            result = ToolResult(tool="finish", success=True)
            final_status = tool_call.arguments.get("status", "incomplete")
            return {
                "next_action":  "finish",
                "final_status": final_status,
                "tool_trace":   state["tool_trace"] + [raw_call],
            }
        else:
            result = await self.http_tool.execute(tool_call)
            return {
                "tool_trace": state["tool_trace"] + [{
                    "call":   raw_call,
                    "result": result.model_dump(),
                }],
            }

    async def _observe_node(self, state: WorkerState) -> Dict:
        """
        把工具执行结果追加到消息历史，供 LLM 在下一轮 think 时参考。
        同时更新步数。
        """
        # 取最近一次工具结果
        if state["tool_trace"]:
            last_trace = state["tool_trace"][-1]
            observation = json.dumps(last_trace, ensure_ascii=False)
        else:
            observation = "（无工具结果）"

        return {
            "messages": [{
                "role":    "user",
                "content": f"工具执行结果：{observation}\n请继续，或调用 finish 结束。",
            }],
            "steps": state["steps"] + 1,
        }

    def _route(self, state: WorkerState) -> str:
        """
        决定 observe 之后走哪条边。
        返回值必须是 add_conditional_edges 字典里的 key。
        """
        if state["next_action"] == "finish":
            return "finish"
        if state["steps"] >= self.spec.max_steps:
            return "budget_exceeded"
        return "continue"

    async def _report_node(self, state: WorkerState) -> Dict:
        """
        汇总执行结果，生成 TaskResult。
        结果存入 state，run() 方法从 state 里取出返回。
        """
        # 从 tool_trace 里提取规则验证结果
        # （简化实现：让 LLM 从对话历史里总结规则结论）
        rule_results = await self._extract_rule_results(state)

        # 停止覆盖率测量
        self.coverage_tool.stop()
        coverage_hint = self.coverage_tool.get_report()

        status = state["final_status"]
        if state["next_action"] == "budget_exceeded":
            status = "incomplete"

        result = TaskResult(
            task_id       = state["task"].task_id,
            agent_id      = self.agent_id,
            status        = status,
            rule_results  = rule_results,
            tool_trace    = state["tool_trace"],
            coverage_hint = coverage_hint,
        )

        # 把结果存入状态，run() 从这里取
        return {"_result": result}

    async def _extract_rule_results(
        self, state: WorkerState
    ) -> List[RuleResult]:
        """
        让 LLM 根据完整对话历史，判断每条规则是否通过。
        离线模式：根据 tool_trace 的 status_code 做简单推断。
        """
        if self.offline:
            return self._offline_rule_results(state)

        summary_prompt = (
            f"根据以下工具调用记录，判断这些规则是否通过：\n"
            f"规则：{json.dumps(self.spec.rules, ensure_ascii=False)}\n"
            f"记录：{json.dumps(state['tool_trace'], ensure_ascii=False)}\n"
            f"返回 JSON 数组：["
            f'{{"rule": "...", "passed": true/false, "evidence": "..."}}]'
        )

        response = await self.llm.chat_json(
            messages=[Message(role="user", content=summary_prompt)],
            offline_reply=[],
        )

        return [RuleResult(**r) for r in response]

    async def run(self, task: AgentTask) -> TaskResult:
        """
        BaseAgent 要求实现的方法。
        执行一次任务，返回结果。可以多次调用（可重用）。
        """
        self._set_status(AgentStatus.RUNNING)
        self.coverage_tool.start()

        try:
            # 构造初始状态
            initial_state: WorkerState = {
                "task":         task,
                "messages":     [],
                "tool_trace":   [],
                "rule_results": [],
                "steps":        0,
                "next_action":  "continue",
                "final_status": "incomplete",
            }

            # 运行 LangGraph 图
            final_state = await self._graph.ainvoke(initial_state)

            result = final_state.get("_result") or TaskResult(
                task_id  = task.task_id,
                agent_id = self.agent_id,
                status   = "error",
                error_message = "图执行完成但未生成结果",
            )

        except Exception as e:
            self.coverage_tool.stop()
            result = TaskResult(
                task_id       = task.task_id,
                agent_id      = self.agent_id,
                status        = "error",
                error_message = str(e),
            )

        self._set_status(AgentStatus.FINISHED)
        return result

    async def teardown(self) -> None:
        """BaseAgent 要求的清理方法"""
        self.coverage_tool.cleanup()

    def _offline_think(self, state: WorkerState) -> str:
        """
        离线模式下的确定性决策脚本。
        根据已执行步数返回预设的工具调用。
        """
        steps = state["steps"]
        paths = self.spec.allowed_paths or ["/items/foo"]

        script = [
            # step 0：GET 第一个路径
            {"tool": "http_request",
             "arguments": {"method": "GET", "path": paths[0]}},
            # step 1：PUT 有效数据
            {"tool": "http_request",
             "arguments": {"method": "PUT", "path": paths[0],
                           "body": {"name": "Test", "price": 9.9}}},
            # step 2：GET 验证写入
            {"tool": "http_request",
             "arguments": {"method": "GET", "path": paths[0]}},
            # step 3：PUT 无效数据（边界测试）
            {"tool": "http_request",
             "arguments": {"method": "PUT", "path": paths[0],
                           "body": {"name": "Test", "price": "not_a_number"}}},
            # step 4：GET 不存在的路径
            {"tool": "http_request",
             "arguments": {"method": "GET", "path": "/items/nonexistent"}},
        ]

        if steps < len(script):
            return json.dumps(script[steps], ensure_ascii=False)

        # 超出脚本范围：结束
        return json.dumps({
            "tool": "finish",
            "arguments": {"status": "passed", "summary": "离线脚本执行完毕"},
        })

    def _offline_rule_results(self, state: WorkerState) -> List[RuleResult]:
        """离线模式下从 tool_trace 简单推断规则结果"""
        results = []
        for rule in self.spec.rules:
            # 检查是否有任何成功的 HTTP 调用作为证据
            has_evidence = any(
                t.get("result", {}).get("success")
                for t in state["tool_trace"]
                if isinstance(t, dict) and "result" in t
            )
            results.append(RuleResult(
                rule     = rule,
                passed   = has_evidence,
                evidence = f"离线模式，基于 {len(state['tool_trace'])} 次工具调用推断",
            ))
        return results