# core/llm_client.py

from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel


# ── 常量 ──────────────────────────────────────────────────────────────────────

DEFAULT_MODEL   = "deepseek/deepseek-chat"
DEFAULT_API_URL = "https://openrouter.ai/api/v1/chat/completions"


# ── 消息格式 ──────────────────────────────────────────────────────────────────

class Message(BaseModel):
    """单条对话消息，对应 OpenAI Chat 格式"""
    role:    str          # "system" | "user" | "assistant"
    content: str


# ── 主客户端 ──────────────────────────────────────────────────────────────────

class LLMClient:
    """
    LLM 调用的统一入口。

    两种模式：
      offline=True  → 不发网络请求，直接返回 offline_reply
      offline=False → 调用 OpenRouter（或兼容 API）
    """

    def __init__(
        self,
        offline: bool = False,
        api_key: Optional[str] = None,
        model:   Optional[str] = None,
        api_url: Optional[str] = None,
    ):
        self.offline = offline
        self.api_key = api_key or os.getenv("PROJECT_GUIDE_LLM_API_KEY", "")
        self.model   = model   or os.getenv("PROJECT_GUIDE_LLM_MODEL", DEFAULT_MODEL)
        self.api_url = api_url or os.getenv("PROJECT_GUIDE_LLM_URL",   DEFAULT_API_URL)

    # ── 核心方法 ───────────────────────────────────────────────────────────────

    async def chat(
        self,
        messages: List[Message],
        offline_reply: str = "",
        temperature: float = 0.2,
    ) -> str:
        """
        发送对话，返回模型的文本回复。

        参数：
          messages      – 完整对话历史（含 system prompt）
          offline_reply – 离线模式下直接返回的字符串
          temperature   – 采样温度，测试场景建议低温（更确定性）
        """
        if self.offline:
            return offline_reply

        payload = {
            "model":       self.model,
            "messages":    [m.model_dump() for m in messages],
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.api_url,
                json=payload,
                headers=headers,
            )

        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def chat_json(
        self,
        messages: List[Message],
        offline_reply: Dict[str, Any],
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """
        chat() 的便捷版本：期望模型返回 JSON，自动解析。

        offline_reply 直接传字典，不用自己 json.dumps。
        """
        if self.offline:
            return offline_reply

        raw = await self.chat(messages, temperature=temperature)

        # 模型有时会在 JSON 外面包一层 markdown 代码块，需要剥离
        raw = _strip_markdown_code_block(raw)

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM 返回了无法解析的 JSON：{e}\n原始内容：{raw!r}"
            ) from e


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _strip_markdown_code_block(text: str) -> str:
    """
    去除 LLM 常见的 markdown 代码块包装：

      ```json
      { ... }
      ```

    变成：

      { ... }
    """
    text = text.strip()
    if text.startswith("```"):
        # 去掉第一行（```json 或 ```）和最后一行（```）
        lines = text.splitlines()
        text  = "\n".join(lines[1:-1]).strip()
    return text