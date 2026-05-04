"""Tests for benchmark runner with mock provider."""

import pytest
from llm_tester.providers.base import ChatMessage, Role, ProviderConfig, ChatResponse, Usage
from llm_tester.benchmarks.runner import BenchmarkRunner, TestCase


class MockProvider:
    name = "mock"
    config = ProviderConfig(api_key="test", base_url="http://test", model="mock-v1")

    async def chat(self, messages, **kwargs):
        return ChatResponse(
            content="The answer is 80 km/h",
            model="mock-v1",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            latency_ms=100,
        )


@pytest.mark.asyncio
async def test_run_case():
    runner = BenchmarkRunner([MockProvider()])
    case = TestCase(
        name="test_speed",
        messages=[ChatMessage(Role.USER, "What is 80?")],
        expected_contains="80",
    )
    results = await runner.run_case(case)
    assert len(results) == 1
    assert results[0].passed is True
    assert results[0].provider == "mock"


@pytest.mark.asyncio
async def test_run_case_fail():
    runner = BenchmarkRunner([MockProvider()])
    case = TestCase(
        name="test_fail",
        messages=[ChatMessage(Role.USER, "test")],
        expected_contains="999",
    )
    results = await runner.run_case(case)
    assert results[0].passed is False


@pytest.mark.asyncio
async def test_summary_table():
    runner = BenchmarkRunner([MockProvider()])
    case = TestCase(
        name="test",
        messages=[ChatMessage(Role.USER, "test")],
    )
    await runner.run_case(case)
    table = runner.summary_table()
    assert "mock" in table
    assert "mock-v1" in table
