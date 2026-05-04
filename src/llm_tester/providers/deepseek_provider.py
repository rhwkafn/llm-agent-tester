"""DeepSeek provider - OpenAI-compatible."""

from .openai_provider import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    name = "deepseek"
    # DeepSeek uses OpenAI-compatible format, base class handles everything.
