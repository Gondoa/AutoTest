"""Exercise feedback, false-success detection, isolation and stopping rules."""

from copy import deepcopy
import json

import httpx
import pytest

from target_app import INITIAL_ITEMS, create_app
from user_agent import REQUIRED, TARGET, offline_decision, run_agent


@pytest.mark.asyncio
async def test_editor_uses_observations_and_runs_in_isolation():
    observations = []

    async def decision(state):
        observations.append(deepcopy(state["trace"]))
        return await offline_decision(state)

    first = await run_agent(decide=decision)
    second = await run_agent(force_offline=True)
    assert first["status"] == second["status"] == "passed"
    assert first["completed_checks"] == REQUIRED
    assert first["steps"] == 5
    assert observations[0] == []
    assert observations[2][-1]["response"]["body"] == TARGET
    assert observations[4][-1]["response"]["status"] == 422
    assert second["trace"][0]["response"]["body"] == INITIAL_ITEMS["foo"]


@pytest.mark.asyncio
@pytest.mark.parametrize("bug,failed_step", [("lost_write", 3), ("invalid_mutation", 5)])
async def test_detects_bugs_that_status_only_tests_miss(bug, failed_step):
    app = create_app()

    @app.middleware("http")
    async def inject_bug(request, call_next):
        before = deepcopy(app.state.items)
        response = await call_next(request)
        if request.method == "PUT":
            if bug == "lost_write" and response.status_code == 200:
                app.state.items.clear()
                app.state.items.update(before)
            elif bug == "invalid_mutation" and response.status_code == 422:
                app.state.items["foo"]["name"] = "Corrupted"
        return response

    result = await run_agent(force_offline=True, app=app)
    assert result["status"] == "failed"
    assert result["steps"] == failed_step
    checks = result["trace"][-1]["checks"]
    assert checks[0]["passed"]  # HTTP status alone would pass.
    assert checks[1]["rule"] == "item_values"
    assert not checks[1]["passed"]


@pytest.mark.asyncio
async def test_budget_exhaustion_is_not_success():
    result = await run_agent(force_offline=True, max_steps=2)
    assert result["status"] == "incomplete"
    assert result["steps"] == 2
    assert "read_back" in result["missing_checks"]


@pytest.mark.asyncio
@pytest.mark.parametrize("action,status", [
    ({"tool": "finish", "reason": "I think everything passed"}, "incomplete"),
    ({"tool": "shell", "reason": "Run arbitrary code"}, "error"),
    ({"tool": "read_item", "reason": "Read", "url": "https://example.com"}, "error"),
])
async def test_agent_cannot_bypass_verification_or_tool_schema(action, status):
    async def decision(state):
        return action

    result = await run_agent(decide=decision)
    assert result["status"] == status
    assert result["steps"] == 0
    assert result["missing_checks"] == REQUIRED


@pytest.mark.asyncio
async def test_online_adapter_receives_feedback(monkeypatch):
    monkeypatch.setenv("PROJECT_GUIDE_LLM_API_KEY", "fake-test-key")
    monkeypatch.setenv("PROJECT_GUIDE_LLM_URL", "https://provider.test/chat/completions")
    monkeypatch.setenv("PROJECT_GUIDE_LLM_MODEL", "test-model")
    prompts = []

    async def respond(request):
        payload = json.loads(request.content)
        assert payload["model"] == "test-model"
        prompt = json.loads(payload["messages"][1]["content"])
        prompts.append(prompt)
        action = await offline_decision({"completed": prompt["completed_checks"]})
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(action)}}]})

    original_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        if "transport" not in kwargs:
            kwargs["transport"] = httpx.MockTransport(respond)
        return original_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    result = await run_agent()
    assert result["mode"] == "online"
    assert result["status"] == "passed"
    assert len(prompts) == 5
    assert prompts[-1]["observations"][-1]["response"]["status"] == 422


@pytest.mark.asyncio
async def test_provider_failure_is_reported_without_fallback(monkeypatch):
    monkeypatch.setenv("PROJECT_GUIDE_LLM_API_KEY", "fake-test-key")

    async def unavailable(state):
        raise httpx.ConnectError("provider unavailable")

    monkeypatch.setattr("user_agent.model_decision", unavailable)
    result = await run_agent()
    assert result["mode"] == "online"
    assert result["status"] == "error"
    assert result["steps"] == 0


@pytest.mark.asyncio
async def test_non_json_server_failure_keeps_response_evidence():
    app = create_app()

    @app.middleware("http")
    async def crash(request, call_next):
        raise RuntimeError("injected server failure")

    result = await run_agent(force_offline=True, app=app)
    assert result["status"] == "failed"
    assert result["trace"][0]["response"]["status"] == 500
    assert isinstance(result["trace"][0]["response"]["body"], str)
