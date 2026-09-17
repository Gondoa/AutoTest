"""Minimal Supervisor/Workers test-agent cluster.

The supervisor owns planning and review. Workers own a narrow test concern and
return structured evidence; they never decide the overall release result.
"""

from operator import add
from typing import Any, Dict, List, Optional

import httpx
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

from target_app import create_app


class ClusterState(TypedDict):
    requirement: str
    tasks: List[Dict[str, str]]
    results: Annotated[List[Dict[str, Any]], add]
    missing_rules: List[str]
    round: int
    max_rounds: int
    status: str


RULES = [
    "read_existing_item",
    "read_missing_item",
    "valid_update_persists",
    "invalid_update_rejected",
    "invalid_update_preserves_state",
]


def _task(agent: str, objective: str, rules: List[str]) -> Dict[str, str]:
    return {"agent": agent, "objective": objective, "rules": ",".join(rules)}


def _initial_tasks(requirement: str) -> List[Dict[str, str]]:
    return [
        _task("functional", "验证正常读取和合法更新", ["read_existing_item", "valid_update_persists"]),
        _task("boundary", "验证不存在资源和缺少必填字段", ["read_missing_item", "invalid_update_rejected"]),
        _task("state", "验证失败请求不会破坏已保存状态", ["invalid_update_preserves_state"]),
    ]


def _follow_up_tasks(missing_rules: List[str]) -> List[Dict[str, str]]:
    """Turn review gaps into the smallest next batch of worker tasks."""
    tasks = []
    functional_rules = {"read_existing_item", "valid_update_persists"}
    boundary_rules = {"read_missing_item", "invalid_update_rejected"}
    state_rules = {"invalid_update_preserves_state"}
    if functional_rules.intersection(missing_rules):
        tasks.append(_task("functional", "补测正常读取和合法更新", list(functional_rules.intersection(missing_rules))))
    if boundary_rules.intersection(missing_rules):
        tasks.append(_task("boundary", "补测不存在资源和非法输入", list(boundary_rules.intersection(missing_rules))))
    if state_rules.intersection(missing_rules):
        tasks.append(_task("state", "补测失败请求后的状态一致性", list(state_rules.intersection(missing_rules))))
    return tasks


async def _request(client: httpx.AsyncClient, method: str, path: str,
                   payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    response = await client.request(method, path, json=payload)
    try:
        body = response.json()
    except ValueError:
        body = response.text
    return {"method": method, "path": path, "payload": payload,
            "status": response.status_code, "body": body}


async def _run_worker(task: Dict[str, str]) -> Dict[str, Any]:
    """Run one worker in an isolated target-data snapshot."""
    agent = task["agent"]
    app = create_app()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        evidence: List[Dict[str, Any]] = []
        checks: List[Dict[str, Any]] = []

        if agent == "functional":
            evidence.append(await _request(client, "GET", "/items/foo"))
            evidence.append(await _request(
                client, "PUT", "/items/foo",
                {"name": "Cluster Updated", "price": 88.8, "tax": 7.5},
            ))
            evidence.append(await _request(client, "GET", "/items/foo"))
            checks.extend([
                {"rule": "read_existing_item", "passed": evidence[0]["status"] == 200},
                {"rule": "valid_update_persists", "passed": (
                    evidence[1]["status"] == 200
                    and evidence[2]["body"] == {"name": "Cluster Updated", "price": 88.8, "tax": 7.5}
                )},
            ])
        elif agent == "boundary":
            evidence.append(await _request(client, "GET", "/items/missing"))
            evidence.append(await _request(client, "PUT", "/items/foo", {"name": "Invalid"}))
            checks.extend([
                {"rule": "read_missing_item", "passed": evidence[0]["status"] == 404},
                {"rule": "invalid_update_rejected", "passed": evidence[1]["status"] == 422},
            ])
        elif agent == "state":
            evidence.append(await _request(client, "GET", "/items/foo"))
            evidence.append(await _request(client, "PUT", "/items/foo", {"name": "Invalid"}))
            evidence.append(await _request(client, "GET", "/items/foo"))
            checks.append({
                "rule": "invalid_update_preserves_state",
                "passed": evidence[1]["status"] == 422 and evidence[2]["body"] == evidence[0]["body"],
            })
        else:
            raise ValueError("Unknown worker: " + agent)

    return {
        "agent": agent,
        "objective": task["objective"],
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
        "checks": checks,
        "evidence": evidence,
    }


async def _supervisor_decision(state: ClusterState) -> Dict[str, Any]:
    """Create an initial plan or a targeted follow-up plan."""
    tasks = (_initial_tasks(state["requirement"])
             if state["round"] == 0
             else _follow_up_tasks(state["missing_rules"]))
    return {"tasks": tasks, "round": state["round"] + 1, "status": "running"}


async def run_cluster(requirement: str, max_rounds: int = 3) -> Dict[str, Any]:
    if not 1 <= max_rounds <= 10:
        raise ValueError("max_rounds must be between 1 and 10")

    async def plan(state: ClusterState):
        return await _supervisor_decision(state)

    async def functional_worker(state: ClusterState):
        task = next((task for task in state["tasks"] if task["agent"] == "functional"), None)
        return {"results": [] if task is None else [await _run_worker(task)]}

    async def boundary_worker(state: ClusterState):
        task = next((task for task in state["tasks"] if task["agent"] == "boundary"), None)
        return {"results": [] if task is None else [await _run_worker(task)]}

    async def state_worker(state: ClusterState):
        task = next((task for task in state["tasks"] if task["agent"] == "state"), None)
        return {"results": [] if task is None else [await _run_worker(task)]}

    def review(state: ClusterState):
        checks = {
            check["rule"]: check["passed"]
            for result in state["results"]
            for check in result["checks"]
        }
        missing = [rule for rule in RULES if not checks.get(rule, False)]
        if not missing:
            status = "passed"
        elif state["round"] >= state["max_rounds"]:
            status = "incomplete"
        else:
            status = "running"
        return {"missing_rules": missing, "status": status}

    graph = StateGraph(ClusterState)
    graph.add_node("supervisor_plan", plan)
    graph.add_node("functional_worker", functional_worker)
    graph.add_node("boundary_worker", boundary_worker)
    graph.add_node("state_worker", state_worker)
    graph.add_node("supervisor_review", review)
    graph.add_edge(START, "supervisor_plan")
    graph.add_edge("supervisor_plan", "functional_worker")
    graph.add_edge("supervisor_plan", "boundary_worker")
    graph.add_edge("supervisor_plan", "state_worker")
    graph.add_edge("functional_worker", "supervisor_review")
    graph.add_edge("boundary_worker", "supervisor_review")
    graph.add_edge("state_worker", "supervisor_review")
    graph.add_conditional_edges(
        "supervisor_review",
        lambda state: "supervisor_plan" if state["status"] == "running" else END,
    )
    result = await graph.compile().ainvoke({
        "requirement": requirement, "tasks": [], "results": [],
        "missing_rules": RULES.copy(), "round": 0, "max_rounds": max_rounds,
        "status": "created",
    })
    return {
        "architecture": "supervisor-workers",
        "supervisor": {"planned": [task["agent"] for task in result["tasks"]],
                       "rounds": result["round"], "max_rounds": max_rounds},
        "status": result["status"],
        "covered_rules": [
            check["rule"] for worker in result["results"]
            for check in worker["checks"] if check["passed"]
        ],
        "missing_rules": result["missing_rules"],
        "workers": sorted(result["results"], key=lambda worker: worker["agent"]),
    }
