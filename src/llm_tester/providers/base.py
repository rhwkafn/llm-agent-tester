"""Base abstractions for LLM providers."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ChatMessage:
    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatResponse:
    content: str
    model: str
    usage: Usage
    latency_ms: float
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        if self.latency_ms <= 0:
            return 0.0
        return self.usage.completion_tokens / (self.latency_ms / 1000)


@dataclass
class ProviderConfig:
    api_key: str
    base_url: str
    model: str
    max_tokens: int = 2048
    temperature: float = 0.7
    timeout: float = 60.0


class LLMProvider(ABC):
    """Abstract base for all LLM providers."""

    name: str

    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    async def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        ...

    async def benchmark(self, messages: list[ChatMessage], runs: int = 3) -> list[ChatResponse]:
        """Run multiple calls and collect latency/throughput stats."""
        results = []
        for _ in range(runs):
            resp = await self.chat(messages)
            results.append(resp)
        return results

    def _timed(self) -> _Timer:
        return _Timer()


class _Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000
