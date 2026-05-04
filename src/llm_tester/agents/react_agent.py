"""ReAct (Reasoning + Acting) agent implementation."""

from __future__ import annotations

import json
import re

from .base import Agent, AgentConfig, AgentStep, Tool
from ..providers.base import ChatMessage, LLMProvider, Role


REACT_SYSTEM_PROMPT = """You are a helpful assistant that follows the ReAct pattern:
1. Thought: reason about what to do next
2. Action: call a tool if needed, or provide Final Answer

When using a tool, respond in this exact format:
Thought: <your reasoning>
Action: tool_name("arg1", "arg2")

When you have enough information, respond with:
Thought: <your reasoning>
Final Answer: <your answer to the user>"""


class ReActAgent(Agent):
    """Agent using the ReAct (Reason + Act) paradigm."""

    def __init__(self, provider: LLMProvider, tools: list[Tool] | None = None,
                 name: str = "react-agent", max_iterations: int = 8):
        config = AgentConfig(
            name=name,
            system_prompt=REACT_SYSTEM_PROMPT,
            max_iterations=max_iterations,
            temperature=0.2,
        )
        super().__init__(provider, config, tools)

    async def run(self, user_input: str) -> str:
        self.history = [self._system_message()]
        self.steps = []
        self.history.append(ChatMessage(role=Role.USER, content=user_input))

        for _ in range(self.config.max_iterations):
            resp = await self.provider.chat(self.history)
            content = resp.content

            # Check for Final Answer
            final_match = re.search(r"Final Answer:\s*(.+)", content, re.DOTALL)
            if final_match:
                self.steps.append(AgentStep(
                    thought=content, action=None, observation="", response=resp,
                ))
                return final_match.group(1).strip()

            # Extract and execute action
            action_match = re.search(r"Action:\s*(\w+)\(([^)]*)\)", content)
            if action_match:
                tool_name = action_match.group(1)
                raw_args = action_match.group(2)

                # Try to parse args
                try:
                    args = [json.loads(a.strip()) for a in raw_args.split(",") if a.strip()]
                except json.JSONDecodeError:
                    args = [raw_args.strip('"').strip("'")]

                observation = self._execute_tool(tool_name, args)
                self.steps.append(AgentStep(
                    thought=content, action=tool_name, observation=observation, response=resp,
                ))
                self.history.append(ChatMessage(role=Role.ASSISTANT, content=content))
                self.history.append(ChatMessage(role=Role.USER, content=f"Observation: {observation}"))
                continue

            # No action or final answer — treat as final
            return content

        return "Max iterations reached."

    def _execute_tool(self, name: str, args: list) -> str:
        if name not in self.tools:
            return f"Error: tool '{name}' not found"
        try:
            result = self.tools[name].func(*args)
            return str(result)
        except Exception as e:
            return f"Error executing {name}: {e}"
