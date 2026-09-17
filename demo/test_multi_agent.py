"""Tests for supervisor planning, isolated workers and deterministic review."""

import pytest

import multi_agent
from multi_agent import RULES, run_cluster


@pytest.mark.asyncio
async def test_supervisor_runs_specialists_and_aggregates_rules():
    report = await run_cluster("测试 item API")
    assert report["architecture"] == "supervisor-workers"
    assert report["supervisor"] == {
        "planned": ["functional", "boundary", "state"], "rounds": 1,
        "max_rounds": 3,
    }
    assert report["status"] == "passed"
    assert report["missing_rules"] == []
    assert sorted(report["covered_rules"]) == sorted(RULES)
    assert [worker["status"] for worker in report["workers"]] == ["passed"] * 3


@pytest.mark.asyncio
async def test_workers_do_not_share_mutable_target_data():
    report = await run_cluster("测试数据隔离")
    workers = {worker["agent"]: worker for worker in report["workers"]}
    functional = workers["functional"]
    boundary = workers["boundary"]
    state = workers["state"]
    assert functional["evidence"][-1]["body"]["name"] == "Cluster Updated"
    assert boundary["evidence"][1]["status"] == 422
    assert state["evidence"][0]["body"]["name"] == "Foo"


@pytest.mark.asyncio
async def test_cluster_rejects_invalid_round_budget():
    with pytest.raises(ValueError):
        await run_cluster("测试", max_rounds=0)


@pytest.mark.asyncio
async def test_supervisor_can_stop_after_one_round_with_missing_rules():
    report = await run_cluster("测试", max_rounds=1)
    assert report["status"] == "passed"
    assert report["supervisor"]["rounds"] == 1


@pytest.mark.asyncio
async def test_supervisor_reassigns_only_missing_rule(monkeypatch):
    calls = []

    async def flaky_worker(task):
        calls.append((task["agent"], task["rules"]))
        passed = task["agent"] != "state" or len(calls) > 3
        return {
            "agent": task["agent"], "objective": task["objective"],
            "status": "passed" if passed else "failed",
            "checks": [{"rule": rule, "passed": passed}
                       for rule in task["rules"].split(",")],
            "evidence": [],
        }

    monkeypatch.setattr(multi_agent, "_run_worker", flaky_worker)
    report = await run_cluster("测试反馈补测", max_rounds=2)
    assert report["status"] == "passed"
    assert report["supervisor"]["rounds"] == 2
    assert sorted(calls) == sorted([
        ("functional", "read_existing_item,valid_update_persists"),
        ("boundary", "read_missing_item,invalid_update_rejected"),
        ("state", "invalid_update_preserves_state"),
        ("state", "invalid_update_preserves_state"),
    ])
