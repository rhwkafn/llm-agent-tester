"""Configuration loading from environment / YAML."""

from __future__ import annotations

import os
from dataclasses import dataclass, field



@dataclass
class ProviderEntry:
    name: str
    type: str  # openai, zhipu, deepseek, anthropic
    api_key: str
    base_url: str
    model: str
    max_tokens: int = 2048
    temperature: float = 0.7


@dataclass
class Config:
    providers: list[ProviderEntry] = field(default_factory=list)

    def get_provider_configs(self) -> list[ProviderEntry]:
        return self.providers


def load_config(path: str | None = None) -> Config:
    """Load config from YAML file or fall back to environment variables."""
    if path and os.path.exists(path):
        return _load_yaml(path)
    return _load_from_env()


def _load_yaml(path: str) -> Config:
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    providers = []
    for p in data.get("providers", []):
        providers.append(ProviderEntry(**p))
    return Config(providers=providers)


def _load_from_env() -> Config:
    """Build config from standard environment variables."""
    providers = []

    if key := os.environ.get("OPENAI_API_KEY"):
        providers.append(ProviderEntry(
            name="openai", type="openai", api_key=key,
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        ))

    if key := os.environ.get("ZHIPU_API_KEY"):
        providers.append(ProviderEntry(
            name="zhipu", type="zhipu", api_key=key,
            base_url=os.environ.get("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
            model=os.environ.get("ZHIPU_MODEL", "glm-4-flash"),
        ))

    if key := os.environ.get("DEEPSEEK_API_KEY"):
        providers.append(ProviderEntry(
            name="deepseek", type="deepseek", api_key=key,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        ))

    if key := os.environ.get("ANTHROPIC_API_KEY"):
        providers.append(ProviderEntry(
            name="anthropic", type="anthropic", api_key=key,
            base_url=os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        ))

    return Config(providers=providers)
