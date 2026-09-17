# core/protocol.py

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator
import re
import uuid

class AgentStatus(str, Enum):
    IDLE     = "idle"      # 创建完毕，等待任务
    RUNNING  = "running"   # 正在执行任务
    FINISHED = "finished"  # 任务完成，结果可读取
    FAILED   = "failed"    # 执行出错

class ToolCall(BaseModel):
    """子Agent请求执行某个工具"""
    tool: Literal["http_request", "finish"]
    arguments: Dict[str, Any]

    @field_validator("arguments")
    @classmethod
    def validate_http_args(cls, v, info):
        if info.data.get("tool") == "http_request":
            assert "method" in v, "http_request 必须有 method"
            assert "path" in v,   "http_request 必须有 path"
            assert v["method"] in ("GET", "POST", "PUT", "DELETE", "PATCH")
        return v

class ToolResult(BaseModel):
    """工具执行后返回给子Agent的结果"""
    tool: str
    success: bool
    status_code: Optional[int] = None   # HTTP工具专用
    body: Optional[Any]        = None   # HTTP响应体
    error: Optional[str]       = None   # 出错时的错误信息


class AgentSpec(BaseModel):
    """主Agent创建子Agent时的配置蓝图"""
    agent_type: str = Field(
        ...,
        description="Agent类型标识，如 functional / boundary / state"
    )
    objective: str = Field(..., description="这个Agent的测试目标，自然语言描述")
    prompt: str = Field(..., description="注入给LLM的系统提示词")

    allowed_paths: List[str] = Field(
        default_factory=list,
        description="允许访问的HTTP路径白名单，如 ['/items/foo', '/items/bar']"
    )
    allowed_tools: List[str] = Field(
        default_factory=lambda: ["http_request", "finish"],
        description="允许调用的工具名列表"
    )
    max_steps: int = Field(default=10, ge=1, le=50)
    rules: List[str] = Field(
        default_factory=list,
        description="此Agent必须验证的规则列表，每条是一句自然语言断言"
    )

    @field_validator("agent_type")
    @classmethod
    def validate_agent_type(cls, v: str) -> str:
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("agent_type 只能包含字母、数字、下划线、连字符")
        return v

class AgentTask(BaseModel):
    """主Agent派发给子Agent的一个任务"""
    task_id:    str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id:   str = Field(..., description="目标子Agent的ID")
    instruction: str = Field(..., description="任务指令，自然语言")
    context:    Dict[str, Any] = Field(
        default_factory=dict,
        description="附加上下文，如目标URL、认证信息等"
    )

class RuleResult(BaseModel):
    """子Agent对单条规则的验证结论"""
    rule:    str
    passed:  bool
    evidence: str = Field(..., description="支持结论的证据，如HTTP请求/响应摘要")

class TaskResult(BaseModel):
    """子Agent完成任务后返回给主Agent的完整报告"""
    task_id:      str
    agent_id:     str
    status:       Literal["passed", "failed", "incomplete", "error"]
    rule_results: List[RuleResult] = Field(default_factory=list)
    tool_trace:   List[Dict[str, Any]] = Field(
        default_factory=list,
        description="完整的工具调用记录，用于主Agent审查"
    )
    error_message: Optional[str] = None
    coverage_hint: Optional[Dict[str, float]] = Field(
        default=None,
        description="覆盖率提示，如 {'line_coverage': 0.85}"
    )

class AgentHandle(BaseModel):
    """主Agent持有的子Agent"句柄"——子Agent存活期间的引用"""
    agent_id:       str = Field(default_factory=lambda: str(uuid.uuid4()))
    spec:           AgentSpec
    status:         AgentStatus = AgentStatus.IDLE
    task_history:   List[AgentTask]   = Field(default_factory=list)
    result_history: List[TaskResult]  = Field(default_factory=list)

    def is_available(self) -> bool:
        """子Agent是否可以接收新任务"""
        return self.status == AgentStatus.IDLE

class DecisionContext(BaseModel):
    """主Agent在 review/decide 节点看到的全局视图"""
    round_number:  int
    max_rounds:    int
    all_handles:   List[AgentHandle]
    latest_results: List[TaskResult]

    def coverage_summary(self) -> Dict[str, int]:
        """统计各状态的规则数量，辅助主Agent判断测试完整性"""
        passed   = sum(1 for r in self.latest_results for rr in r.rule_results if rr.passed)
        failed   = sum(1 for r in self.latest_results for rr in r.rule_results if not rr.passed)
        return {"passed": passed, "failed": failed, "total": passed + failed}

    def has_idle_agents(self) -> bool:
        return any(h.is_available() for h in self.all_handles)

    def budget_exhausted(self) -> bool:
        return self.round_number >= self.max_rounds

