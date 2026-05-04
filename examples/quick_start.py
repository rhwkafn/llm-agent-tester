"""
Quick Start - Multi-provider benchmark example.

Usage:
    export ZHIPU_API_KEY="your-key"
    export DEEPSEEK_API_KEY="your-key"
    python examples/quick_start.py
"""

import asyncio
from llm_tester.providers import ZhipuProvider, DeepSeekProvider
from llm_tester.providers.base import ChatMessage, ProviderConfig, Role
from llm_tester.benchmarks import BenchmarkRunner, TestCase
from llm_tester.utils.display import print_results_table


async def main():
    # Initialize providers from environment
    providers = []

    import os
    if key := os.environ.get("ZHIPU_API_KEY"):
        providers.append(ZhipuProvider(ProviderConfig(
            api_key=key,
            base_url="https://open.bigmodel.cn/api/paas/v4",
            model="glm-4-flash",
        )))

    if key := os.environ.get("DEEPSEEK_API_KEY"):
        providers.append(DeepSeekProvider(ProviderConfig(
            api_key=key,
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
        )))

    if not providers:
        print("Set ZHIPU_API_KEY and/or DEEPSEEK_API_KEY to run this example.")
        return

    # Define test cases
    test_cases = [
        TestCase(
            name="basic_reasoning",
            messages=[
                ChatMessage(Role.USER, "If a train travels 120km in 1.5 hours, what is its average speed? Answer with just the number."),
            ],
            expected_contains="80",
            max_latency_ms=5000,
        ),
        TestCase(
            name="code_generation",
            messages=[
                ChatMessage(Role.USER, "Write a Python function to check if a number is prime. Only output the function, no explanation."),
            ],
            expected_contains="def ",
            max_latency_ms=10000,
        ),
        TestCase(
            name="translation",
            messages=[
                ChatMessage(Role.USER, 'Translate to English: "大语言模型正在改变软件开发的方式"'),
            ],
            expected_contains="language model",
            max_latency_ms=8000,
        ),
    ]

    runner = BenchmarkRunner(providers)
    results = await runner.run_suite(test_cases)
    print_results_table(results, title="Multi-Provider Benchmark")


if __name__ == "__main__":
    asyncio.run(main())
