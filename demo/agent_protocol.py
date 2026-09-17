"""Small, explicit protocol shared by the supervisor and worker agents."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class AgentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_type: str = Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")
    objective: str = Field(min_length=1, max_length=300)
    prompt: str = Field(default="You are a careful software tester.", max_length=2000)
    project: Dict[str, Any] = Field(default_factory=dict)
    permissions: List[Literal["read", "write"]] = Field(default_factory=lambda: ["read", "write"])
    tools: List[Literal["http_request", "finish"]] = Field(default_factory=lambda: ["http_request", "finish"])
    rules: List[str] = Field(min_length=1, max_length=10)
    max_steps: int = Field(default=3, ge=1, le=10)


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: Literal["http_request", "finish"]
    method: Optional[Literal["GET", "PUT"]] = None
    path: Optional[str] = None
    body: Optional[Dict[str, Any]] = None
    reason: str = Field(min_length=1, max_length=300)


class ToolResult(BaseModel):
    call_id: str
    status: Literal["success", "error"]
    result: Dict[str, Any]
    retryable: bool = False


class AgentTask(BaseModel):
    """A supervisor message addressed to one previously-created worker."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1, max_length=120)
    agent_type: str = Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")
    objective: str = Field(min_length=1, max_length=300)
    rules: List[str] = Field(min_length=1, max_length=10)


class RuleResult(BaseModel):
    rule: str
    passed: bool
    expected: Any = None
    actual: Any = None
    evidence_ids: List[str] = []


class WorkerResult(BaseModel):
    task_id: str
    agent_type: str
    objective: str
    status: Literal["passed", "failed", "incomplete", "error"]
    rules: List[RuleResult]
    trace: List[Dict[str, Any]]
    next_suggestions: List[str] = []
