"""
Multi-Agent Demo - Orchestrator routing tasks to specialized agents.

Shows how a router agent delegates to sub-agents based on task type.
"""

import asyncio

from llm_tester.agents import MultiAgentOrchestrator
from llm_tester.agents.base import AgentSpec, Tool
from llm_tester.providers import ZhipuProvider
from llm_tester.providers.base import ProviderConfig


async def main():
    import os
    key = os.environ.get("ZHIPU_API_KEY")
    if not key:
        print("Set ZHIPU_API_KEY to run this example.")
        return

    provider = ZhipuProvider(ProviderConfig(
        api_key=key,
        base_url="https://open.bigmodel.cn/api/paas/v4",
        model="glm-4-flash",
    ))

    # Define specialized agents
    agents = [
        AgentSpec(
            name="coder",
            role_description="Expert at writing and explaining code. Handles programming questions.",
            system_prompt="You are a senior software engineer. Write clean, efficient code with brief explanations.",
        ),
        AgentSpec(
            name="analyst",
            role_description="Expert at data analysis, math, and logical reasoning.",
            system_prompt="You are a data analyst. Provide clear, step-by-step analysis with numbers.",
        ),
        AgentSpec(
            name="writer",
            role_description="Expert at writing, translation, and creative content.",
            system_prompt="You are a skilled writer. Produce polished, engaging content.",
        ),
    ]

    orchestrator = MultiAgentOrchestrator(provider, agents)

    tasks = [
        "Write a Python function to merge two sorted lists",
        "If I invest $10000 at 5% annual compound interest, how much will I have after 8 years?",
        "Translate '机器学习正在重塑各行各业' to English with a brief explanation",
    ]

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        result = await orchestrator.run(task)
        print(f"Routed to: {result['routed_to']}")
        print(f"Steps: {result['steps']}")
        print(f"Result: {result['result'][:300]}")


if __name__ == "__main__":
    asyncio.run(main())
