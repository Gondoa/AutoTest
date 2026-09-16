"""Generated API tests accepted from the AI test plan."""

import pytest
import httpx

from ai_planner import create_test_plan
from target_app import app


PLAN = create_test_plan("测试 item API 的读取、更新和参数校验", force_offline=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", PLAN, ids=[case["id"] for case in PLAN])
async def test_ai_generated_api_case(case):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.request(
            case["method"], case["path"], json=case.get("payload")
        )

    assert response.status_code == case["expected_status"]
    body = response.json()
    for field in case["assertions"]:
        assert field in body
