"""OpenAI-compatible provider (works with any OpenAI-format API)."""

from __future__ import annotations

import httpx

from .base import ChatMessage, ChatResponse, LLMProvider, ProviderConfig, Usage


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=config.timeout,
        )

    async def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        payload = {
            "model": kwargs.get("model", self.config.model),
            "messages": [m.to_dict() for m in messages],
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }

        with self._timed() as t:
            resp = await self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        usage_raw = data.get("usage", {})

        return ChatResponse(
            content=choice["message"]["content"],
            model=data.get("model", payload["model"]),
            usage=Usage(
                prompt_tokens=usage_raw.get("prompt_tokens", 0),
                completion_tokens=usage_raw.get("completion_tokens", 0),
                total_tokens=usage_raw.get("total_tokens", 0),
            ),
            latency_ms=t.elapsed_ms,
            raw=data,
        )

    async def close(self):
        await self._client.aclose()
