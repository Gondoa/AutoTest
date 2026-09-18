# AutoTest

A multi-agent software testing framework built on [LangGraph](https://github.com/langchain-ai/langgraph). A master agent orchestrates a cluster of worker agents to automatically test HTTP APIs — planning test strategies, spawning agents with scoped permissions, dispatching tasks, and making round-by-round decisions based on results.

---

## How It Works

```
MasterAgent (LangGraph graph)
  │
  ├─ plan        LLM decides what test types and rules are needed
  ├─ create      Spawns WorkerAgents, each with isolated app instance + permissions
  ├─ dispatch    Sends tasks to idle agents in the registry
  ├─ run         asyncio.gather — all workers run concurrently
  ├─ review      LLM audits results and coverage gaps
  └─ decide ─────► need_more  → re-dispatch to existing idle agents
             ├───► need_new   → spawn additional agents
             └───► done       → generate final report

WorkerAgent (LangGraph graph, per agent)
  │
  ├─ receive     Initialize state from task + AgentSpec
  ├─ think       LLM decides next tool call
  ├─ act         Execute http_request or finish
  ├─ observe     Feed result back into conversation history
  └─ decide ─────► continue → think (loop)
             └───► finish / budget_exceeded → report
```

Key properties:

- **Master creates workers at runtime** — agent type, objectives, allowed paths, tool list, and rules are all decided by the master LLM (or the offline script).
- **Master can re-task existing agents** — after a review round the master dispatches supplemental tasks to already-registered idle agents without spawning new ones.
- **Workers are independent** — each worker has its own LLM call history, its own in-memory app instance, and its own coverage measurement. One worker's writes cannot affect another's reads.
- **Workers are reusable** — the LangGraph graph is compiled once per worker; each `run(task)` call starts from a fresh `initial_state`, so the same worker object can receive multiple sequential tasks.

---

## Project Structure

```
AutoTest/
├── core/
│   ├── protocol.py          # All Pydantic models (AgentSpec, AgentHandle, TaskResult, …)
│   ├── registry.py          # AgentRegistry — lifecycle management for worker agents
│   └── llm_client.py        # LLM client (OpenRouter) with offline/online switch
│
├── agents/
│   ├── base_agent.py        # Abstract base class — run() contract + status lifecycle
│   ├── master_agent.py      # Master agent LangGraph graph
│   └── worker_agent.py      # Worker agent LangGraph graph
│
├── tools/
│   ├── http_tool.py         # HTTP request tool with path-whitelist enforcement
│   └── coverage_tool.py     # Runtime line + branch coverage via coverage.py
│
├── targets/
│   └── fastapi_app/
│       └── app.py           # Target FastAPI app (factory function, isolated per agent)
│
├── tests/
│   ├── test_core.py         # Protocol models + registry unit tests
│   ├── test_worker_agent.py # Worker isolation, reusability, permission tests
│   ├── test_master_agent.py # Master orchestration + round budget tests
│   └── test_integration.py  # End-to-end offline cluster tests
│
├── runners/
│   └── run_cluster.py       # CLI entry point
│
└── demo/                    # Original prototype code (reference only)
```

---

## Requirements

**Python 3.9+**

```
fastapi==0.115.12
httpx==0.28.1
pytest==8.3.5
pytest-asyncio==0.25.3
langgraph==0.6.11
coverage==7.10.7
```

Install:

```bash
pip install fastapi httpx pytest pytest-asyncio langgraph coverage
```

No external services are required to run in offline mode. For online (LLM-driven) mode, set the following environment variables:

```bash
export PROJECT_GUIDE_LLM_API_KEY=sk-...          # required
export PROJECT_GUIDE_LLM_MODEL=deepseek/deepseek-chat   # optional, this is the default
export PROJECT_GUIDE_LLM_URL=https://openrouter.ai/api/v1/chat/completions  # optional
```

The API endpoint is [OpenRouter](https://openrouter.ai)-compatible. Any provider that follows the OpenAI chat completions format will work.

---

## Usage

### Run the agent cluster

```bash
# Offline mode — deterministic, no API key needed
python runners/run_cluster.py --offline

# Online mode — LLM-driven planning and decision making
python runners/run_cluster.py

# Custom objective and round limit
python runners/run_cluster.py --objective "Test the user registration and login API" --max-rounds 3

# Exit code: 0 = passed, 1 = failed / incomplete / error
```

### Run the tests

```bash
# Full test suite (offline, no API key needed)
pytest tests/ -v

# Single layer
pytest tests/test_core.py -v
pytest tests/test_worker_agent.py -v
pytest tests/test_master_agent.py -v
pytest tests/test_integration.py -v

# With coverage report
pytest tests/ --cov=. --cov-report=term-missing
```

---

## What Gets Tested

The default target is a FastAPI item CRUD API (`GET /items/{id}`, `PUT /items/{id}`). The master agent plans and assigns tests across three dimensions:

| Agent Type | What It Verifies |
|---|---|
| `functional` | Normal read/write — GET returns correct data, PUT persists correctly |
| `boundary` | Error handling — 404 for missing items, 422 for invalid payloads |
| `state` | State consistency — invalid PUT does not corrupt existing data |

### Offline mode output (example)

```json
{
  "status": "passed",
  "rounds": 1,
  "agents_created": 3,
  "tasks_executed": 3,
  "rules_passed": 5,
  "rules_failed": 0,
  "avg_line_coverage": 0.88,
  "agent_details": [
    { "type": "functional", "tasks_done": 1, "rules": ["GET /items/foo returns 200", "PUT persists correctly"] },
    { "type": "boundary",   "tasks_done": 1, "rules": ["GET missing item returns 404", "PUT invalid data returns 422"] },
    { "type": "state",      "tasks_done": 1, "rules": ["Invalid PUT does not corrupt existing data"] }
  ]
}
```

### Online mode behaviour

In online mode the master LLM:

1. Reads the test objective and generates a tailored set of `AgentSpec` definitions — agent types, rules, allowed paths, and step budgets.
2. After each round, reviews the `tool_trace` and coverage report from every worker, identifies gaps, and chooses one of three next steps: re-task an existing idle agent (`need_more`), spawn a new specialist agent (`need_new`), or conclude (`done`).
3. Workers independently call the LLM to decide each HTTP request, self-correct when they produce malformed JSON, and summarise rule outcomes from their own conversation history.

---

## Extending to Other Targets

The framework is not tied to the bundled FastAPI app. To test a different target:

**In-process ASGI app** — replace `targets/fastapi_app/app.py` with your own `create_app()` factory:

```python
# targets/myapp/app.py
def create_app():
    app = FastAPI()
    # ... your routes ...
    return app
```

Update `source_path` in `runners/run_cluster.py` to point at the new file.

**Live HTTP service** — swap `httpx.ASGITransport` in `tools/http_tool.py` for a real base URL:

```python
# tools/http_tool.py — replace the transport block with:
async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
    response = await _dispatch(client, method, path, body)
```

**New tool capabilities** — add entries to `ToolCall.tool`'s `Literal` type in `core/protocol.py`, implement the execution logic in a new file under `tools/`, and wire it into `WorkerAgent._act_node()`.

---

## Architecture Notes

**Permission model** — each worker's `AgentSpec` carries `allowed_paths` and `allowed_tools`. The `HttpTool` enforces the path whitelist at call time; the `ToolCall` type uses a `Literal` union to reject unknown tool names at the Pydantic validation layer. Workers cannot escalate their own permissions.

**Test isolation** — `create_app()` uses a Python closure so each call produces a completely independent in-memory database. Workers running concurrently cannot interfere with each other's state.

**Coverage isolation** — each `CoverageTool` instance writes to a unique `tempfile` path. Concurrent workers do not share a `.coverage` file.

**Offline / CI mode** — every LLM call site provides an `offline_reply`. With `--offline` the entire cluster runs deterministically without any network calls, making it suitable for CI environments without API keys.

---

## Limitations

- Only HTTP APIs are supported out of the box. Non-HTTP targets (gRPC, message queues, CLI tools) require new tool implementations.
- Code coverage measurement only works for Python targets using `coverage.py`.
- Test quality in online mode depends on the LLM's understanding of the rules and its ability to reason about HTTP responses. Complex multi-step race conditions are unlikely to be discovered automatically.
- There is no assertion library — pass/fail judgements are made by the LLM from raw HTTP response data.
