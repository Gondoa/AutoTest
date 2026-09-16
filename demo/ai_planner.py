"""Minimal AI test planner.

The offline planner keeps the demo runnable without credentials. The online
path uses the project's OpenRouter-compatible environment variables. The
important boundary is that the model returns a validated test plan, not
executable shell commands.
"""

import json
import os
import re
from typing import Any, Dict, List

import httpx


def _offline_plan(requirement: str) -> List[Dict[str, Any]]:
    """Provide a deterministic plan for local demos and CI."""
    return [
        {
            "id": "get-existing-item",
            "title": "读取已存在的 item",
            "method": "GET",
            "path": "/items/foo",
            "expected_status": 200,
            "assertions": ["name", "price", "tax"],
        },
        {
            "id": "get-missing-item",
            "title": "读取不存在的 item 返回 404",
            "method": "GET",
            "path": "/items/missing",
            "expected_status": 404,
            "assertions": ["detail"],
        },
        {
            "id": "update-item",
            "title": "使用合法数据更新 item",
            "method": "PUT",
            "path": "/items/foo",
            "payload": {"name": "Updated", "price": 99.9, "tax": 8.5},
            "expected_status": 200,
            "assertions": ["name", "price", "tax"],
        },
        {
            "id": "reject-invalid-item",
            "title": "缺少必填字段时拒绝更新",
            "method": "PUT",
            "path": "/items/foo",
            "payload": {"name": "Invalid"},
            "expected_status": 422,
            "assertions": ["detail"],
        },
    ]


def _validate_plan(plan: Any) -> List[Dict[str, Any]]:
    if not isinstance(plan, list) or not plan:
        raise ValueError("AI must return a non-empty test plan")
    required = {"id", "method", "path", "expected_status", "assertions"}
    for case in plan:
        if not isinstance(case, dict) or not required.issubset(case):
            raise ValueError("Invalid test plan schema")
        if case["method"] not in {"GET", "PUT"}:
            raise ValueError("Unsupported HTTP method")
        if not isinstance(case["expected_status"], int):
            raise ValueError("expected_status must be an integer")
    return plan


def _parse_model_content(content: Any) -> Any:
    if isinstance(content, (list, dict)):
        return content
    if not isinstance(content, str):
        raise ValueError("AI response content must be JSON text")
    cleaned = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    return json.loads(cleaned)


def create_test_plan(requirement: str, force_offline: bool = False) -> List[Dict[str, Any]]:
    api_url = os.getenv(
        "PROJECT_GUIDE_LLM_URL", "https://openrouter.ai/api/v1/chat/completions"
    )
    api_key = os.getenv("PROJECT_GUIDE_LLM_API_KEY")
    model = os.getenv("PROJECT_GUIDE_LLM_MODEL", "deepseek/deepseek-chat")
    if force_offline or not api_key:
        return _offline_plan(requirement)

    schema = {
        "id": "string",
        "title": "string",
        "method": "GET or PUT",
        "path": "string starting with /",
        "payload": "optional JSON object",
        "expected_status": "integer",
        "assertions": "array of response field names",
    }
    prompt = (
        "You are an API test planner. Read the API contract and requirement, then return only a "
        "JSON array of test cases. Do not return Markdown, Python, shell commands, "
        "or explanations. Each object must match this schema: "
        + json.dumps(schema, ensure_ascii=False)
        + "\nAPI contract:\n"
        + json.dumps(
            {
                "GET /items/{item_id}": {
                    "200": {"name": "string", "price": "number", "tax": "number"},
                    "404": {"detail": "Item not found"},
                },
                "PUT /items/{item_id}": {
                    "request": {"name": "string", "price": "number", "tax": "number"},
                    "200": {"name": "string", "price": "number", "tax": "number"},
                    "422": {"detail": "validation errors"},
                },
                "fixtures": {
                    "existing_item_ids": ["foo", "bar"],
                    "missing_item_id": "missing",
                    "tax_is_optional": True,
                    "default_tax": 10.5,
                },
            },
            ensure_ascii=False,
        )
        + "\nRequirement:\n"
        + requirement
    )
    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }
    response = httpx.post(api_url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return _validate_plan(_parse_model_content(content))
