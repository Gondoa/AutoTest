# tools/http_tool.py

from __future__ import annotations
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI

from core.protocol import ToolCall, ToolResult


class HttpTool:
    """
    子 Agent 的 HTTP 请求工具。

    每个子 Agent 持有一个 HttpTool 实例，
    绑定到该 Agent 专属的 FastAPI app 实例和权限白名单。
    """

    def __init__(
        self,
        app: FastAPI,
        allowed_paths: List[str],
    ):
        self.app           = app
        self.allowed_paths = allowed_paths

    # ── 核心方法 ───────────────────────────────────────────────────────────────

    async def execute(self, call: ToolCall) -> ToolResult:
        """
        执行一次 HTTP 请求。

        先做权限检查，再真正发请求，最后把结果包装成 ToolResult。
        """
        path   = call.arguments.get("path", "")
        method = call.arguments.get("method", "GET").upper()
        body   = call.arguments.get("body", None)

        # ── 权限检查 ────────────────────────────────────────────────────────────
        if not self._is_allowed(path):
            return ToolResult(
                tool        = "http_request",
                success     = False,
                error       = f"路径 {path!r} 不在允许列表中：{self.allowed_paths}",
            )

        # ── 发请求 ──────────────────────────────────────────────────────────────
        transport = httpx.ASGITransport(app=self.app)

        async with httpx.AsyncClient(
            transport = transport,
            base_url  = "http://testserver",
        ) as client:
            response = await _dispatch(client, method, path, body)

        # ── 包装结果 ────────────────────────────────────────────────────────────
        try:
            body_parsed = response.json()
        except Exception:
            body_parsed = response.text

        return ToolResult(
            tool        = "http_request",
            success     = True,           # 请求成功送达（不代表业务成功）
            status_code = response.status_code,
            body        = body_parsed,
        )

    # ── 辅助方法 ───────────────────────────────────────────────────────────────

    def _is_allowed(self, path: str) -> bool:
        """
        检查 path 是否在白名单里。

        白名单为空 → 允许所有路径（主Agent不限制时的默认行为）。
        """
        if not self.allowed_paths:
            return True
        return path in self.allowed_paths


# ── 模块级辅助函数 ────────────────────────────────────────────────────────────

async def _dispatch(
    client: httpx.AsyncClient,
    method: str,
    path:   str,
    body:   Optional[Any],
) -> httpx.Response:
    """
    根据 HTTP 方法分发请求。
    单独提取成函数，避免 execute() 里一堆 if/elif。
    """
    kwargs: Dict[str, Any] = {}
    if body is not None:
        kwargs["json"] = body

    method_map = {
        "GET":    client.get,
        "POST":   client.post,
        "PUT":    client.put,
        "DELETE": client.delete,
        "PATCH":  client.patch,
    }

    func = method_map.get(method)
    if func is None:
        raise ValueError(f"不支持的 HTTP 方法：{method}")

    return await func(path, **kwargs)