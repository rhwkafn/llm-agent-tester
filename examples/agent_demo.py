"""
Agent Demo - ReAct agent with tool use.

Demonstrates a ReAct agent that can reason and use tools.
"""

import asyncio
import math

from llm_tester.agents import ReActAgent
from llm_tester.agents.base import Tool
from llm_tester.providers import ZhipuProvider
from llm_tester.providers.base import ProviderConfig


# Define tools
def calculator(expression: str) -> str:
    """Evaluate a math expression safely."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: invalid characters in expression"
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def word_count(text: str) -> str:
    """Count words in text."""
    return str(len(text.split()))


tools = [
    Tool(
        name="calculator",
        description="Evaluate a math expression. Usage: calculator('2+2')",
        parameters={"type": "object", "properties": {"expression": {"type": "string"}}},
        func=calculator,
    ),
    Tool(
        name="word_count",
        description="Count words in a text. Usage: word_count('hello world')",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        func=word_count,
    ),
]


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

    agent = ReActAgent(provider, tools, name="math-agent")

    result = await agent.run("What is the square root of 144 multiplied by 3?")
    print(f"Answer: {result}")
    print(f"\nSteps taken: {len(agent.steps)}")
    for i, step in enumerate(agent.steps):
        print(f"  Step {i+1}: action={step.action or 'final'}, tokens={step.response.usage.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
