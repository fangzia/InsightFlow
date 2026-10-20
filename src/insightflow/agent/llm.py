"""轻量的 OpenAI 兼容 chat 客户端，对瞬时故障自动重试。"""

from __future__ import annotations

import time

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from insightflow.core.config import get_settings

_RETRYABLE = (APIConnectionError, APITimeoutError)
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """当 chat 端点在多次重试后仍然失败时抛出。"""


class LLMClient:
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise LLMError("INSIGHTFLOW_OPENAI_API_KEY is not set")
        self.model = settings.default_model
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=600.0,
            max_retries=0,  # 由我们自己做重试
        )

    def chat(self, system: str, user: str, *, temperature: float = 0.2, max_retries: int = 3) -> str:
        """单轮 chat 补全；瞬时错误指数退避后重试。"""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                )
                content = resp.choices[0].message.content
                if not content:
                    raise LLMError("LLM 返回了空内容")
                return content
            except APIStatusError as e:
                if e.status_code not in _RETRYABLE_STATUS:
                    raise LLMError(f"LLM API 错误 {e.status_code}: {e}") from e
                last_err = e
            except _RETRYABLE as e:
                last_err = e
            if attempt < max_retries:
                time.sleep(2**attempt * 5)
        raise LLMError(f"LLM 调用在 {max_retries + 1} 次尝试后仍失败") from last_err
