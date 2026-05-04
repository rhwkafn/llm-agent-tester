"""Anthropic Claude provider."""

from __future__ import annotations

import httpx

from .base import ChatMessage, ChatResponse, LLMProvider, ProviderConfig, Role, Usage


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            headers={
                "x-api-key": config.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            timeout=config.timeout,
        )

    async def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        # Anthropic separates system from conversation messages
        system_text = ""
        conversation = []
        for m in messages:
            if m.role == Role.SYSTEM:
                system_text += m.content + "\n"
            else:
                conversation.append(m.to_dict())

        payload: dict = {
            "model": kwargs.get("model", self.config.model),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "messages": conversation,
        }
        if system_text.strip():
            payload["system"] = system_text.strip()

        with self._timed() as t:
            resp = await self._client.post("/messages", json=payload)
            resp.raise_for_status()
            data = resp.json()

        content = data["content"][0]["text"]
        usage_raw = data.get("usage", {})

        return ChatResponse(
            content=content,
            model=data.get("model", payload["model"]),
            usage=Usage(
                prompt_tokens=usage_raw.get("input_tokens", 0),
                completion_tokens=usage_raw.get("output_tokens", 0),
                total_tokens=usage_raw.get("input_tokens", 0) + usage_raw.get("output_tokens", 0),
            ),
            latency_ms=t.elapsed_ms,
            raw=data,
        )

    async def close(self):
        await self._client.aclose()
