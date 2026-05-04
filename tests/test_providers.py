"""Unit tests for providers (no API calls)."""

from llm_tester.providers.base import ChatMessage, Role, Usage, ChatResponse


def test_chat_message_to_dict():
    msg = ChatMessage(role=Role.USER, content="hello")
    d = msg.to_dict()
    assert d == {"role": "user", "content": "hello"}


def test_chat_message_with_name():
    msg = ChatMessage(role=Role.ASSISTANT, content="hi", name="bot")
    d = msg.to_dict()
    assert d["name"] == "bot"


def test_usage_defaults():
    u = Usage()
    assert u.total_tokens == 0


def test_tokens_per_second():
    resp = ChatResponse(
        content="test",
        model="test-model",
        usage=Usage(prompt_tokens=10, completion_tokens=50, total_tokens=60),
        latency_ms=1000,
    )
    assert abs(resp.tokens_per_second - 50.0) < 0.1


def test_tokens_per_second_zero_latency():
    resp = ChatResponse(
        content="test",
        model="test-model",
        usage=Usage(),
        latency_ms=0,
    )
    assert resp.tokens_per_second == 0.0


def test_role_enum():
    assert Role.USER.value == "user"
    assert Role.SYSTEM.value == "system"
