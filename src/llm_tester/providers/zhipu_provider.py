"""Zhipu AI (GLM) provider - OpenAI-compatible with JWT auth."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from base64 import b64encode

import httpx

from .base import ChatMessage, ChatResponse, LLMProvider, ProviderConfig, Usage


def _generate_jwt(api_key: str) -> str:
    """Generate a simple JWT for Zhipu API authentication."""
    try:
        import jwt
    except ImportError:
        # Minimal JWT implementation for Zhipu's HS256 scheme
        parts = api_key.split(".")
        if len(parts) != 2:
            raise ValueError("Invalid Zhipu API key format (expected id.secret)")

        key_id, secret = parts
        now = int(time.time())

        header = b64encode(json.dumps({"alg": "HS256", "sign_type": "SIGN", "typ": "JWT"}).encode()).rstrip(b"=")
        payload_data = {
            "api_key": key_id,
            "exp": now + 3600,
            "timestamp": now,
        }
        payload = b64encode(json.dumps(payload_data).encode()).rstrip(b"=")

        signing_input = header + b"." + payload
        signature = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
        sig_b64 = b64encode(signature).rstrip(b"=")

        return (signing_input + b"." + sig_b64).decode()

    # If PyJWT is available, use it
    parts = api_key.split(".")
    key_id, secret = parts
    now = int(time.time())
    return jwt.encode(
        {"api_key": key_id, "exp": now + 3600, "timestamp": now},
        secret,
        algorithm="HS256",
        headers={"sign_type": "SIGN"},
    )


class ZhipuProvider(LLMProvider):
    """Zhipu AI (GLM series) provider with automatic JWT token management."""

    name = "zhipu"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._token: str | None = None
        self._token_expires: float = 0
        self._client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            timeout=config.timeout,
        )

    def _ensure_token(self) -> str:
        now = time.time()
        if self._token is None or now >= self._token_expires:
            self._token = _generate_jwt(self.config.api_key)
            self._token_expires = now + 3500  # refresh before 1h expiry
        return self._token

    async def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        token = self._ensure_token()
        payload = {
            "model": kwargs.get("model", self.config.model),
            "messages": [m.to_dict() for m in messages],
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }

        with self._timed() as t:
            resp = await self._client.post(
                "/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )
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
