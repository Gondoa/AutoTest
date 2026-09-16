"""Run the smallest AI-assisted API testing demo."""

import asyncio
import json

import httpx

from ai_planner import create_test_plan
from target_app import app


REQUIREMENT = """测试 item API：读取已有和不存在的资源，更新 item，并拒绝缺少必填字段的数据。"""


async def run() -> None:
    plan = create_test_plan(REQUIREMENT)
    transport = httpx.ASGITransport(app=app)
    results = []
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for case in plan:
            response = await client.request(
                case["method"], case["path"], json=case.get("payload")
            )
            body = response.json()
            results.append(
                {
                    "id": case["id"],
                    "status": "passed"
                    if response.status_code == case["expected_status"]
                    else "failed",
                    "actual_status": response.status_code,
                    "expected_status": case["expected_status"],
                    "body": body,
                }
            )

    passed = sum(result["status"] == "passed" for result in results)
    print(json.dumps({"plan": plan, "passed": passed, "total": len(results), "results": results}, indent=2))
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(run())
