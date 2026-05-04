"""Benchmark runner for comparing providers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ..providers.base import ChatMessage, LLMProvider


@dataclass
class TestCase:
    name: str
    messages: list[ChatMessage]
    expected_contains: str | None = None
    max_latency_ms: float | None = None


@dataclass
class BenchmarkResult:
    provider: str
    model: str
    test_name: str
    latency_ms: float
    tokens_per_second: float
    prompt_tokens: int
    completion_tokens: int
    passed: bool
    error: str | None = None
    response_preview: str = ""


class BenchmarkRunner:
    """Run test cases across multiple providers and collect comparison results."""

    def __init__(self, providers: list[LLMProvider]):
        self.providers = providers
        self.results: list[BenchmarkResult] = []

    async def run_case(self, case: TestCase) -> list[BenchmarkResult]:
        """Run a single test case against all providers."""
        tasks = [self._eval_single(p, case) for p in self.providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        case_results = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                case_results.append(BenchmarkResult(
                    provider=self.providers[i].name,
                    model=self.providers[i].config.model,
                    test_name=case.name,
                    latency_ms=0,
                    tokens_per_second=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    passed=False,
                    error=str(r),
                ))
            else:
                case_results.append(r)

        self.results.extend(case_results)
        return case_results

    async def run_suite(self, cases: list[TestCase]) -> list[BenchmarkResult]:
        """Run all test cases."""
        all_results = []
        for case in cases:
            results = await self.run_case(case)
            all_results.extend(results)
        return all_results

    async def _eval_single(self, provider: LLMProvider, case: TestCase) -> BenchmarkResult:
        resp = await provider.chat(case.messages)
        passed = True
        error = None

        if case.expected_contains and case.expected_contains.lower() not in resp.content.lower():
            passed = False
            error = f"Expected '{case.expected_contains}' in response"

        if case.max_latency_ms and resp.latency_ms > case.max_latency_ms:
            passed = False
            error = f"Latency {resp.latency_ms:.0f}ms > {case.max_latency_ms:.0f}ms"

        return BenchmarkResult(
            provider=provider.name,
            model=resp.model,
            test_name=case.name,
            latency_ms=resp.latency_ms,
            tokens_per_second=resp.tokens_per_second,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            passed=passed,
            error=error,
            response_preview=resp.content[:200],
        )

    def summary_table(self) -> str:
        """Return a formatted summary of all results."""
        if not self.results:
            return "No results yet."

        header = f"{'Provider':<15} {'Model':<25} {'Test':<20} {'Latency':>10} {'TPS':>8} {'Pass':>6}"
        sep = "-" * len(header)
        lines = [header, sep]

        for r in self.results:
            lines.append(
                f"{r.provider:<15} {r.model:<25} {r.test_name:<20} "
                f"{r.latency_ms:>8.0f}ms {r.tokens_per_second:>7.1f} {'PASS' if r.passed else 'FAIL':>6}"
            )

        return "\n".join(lines)
