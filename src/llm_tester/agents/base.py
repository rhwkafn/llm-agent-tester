"""Base agent abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..providers.base import ChatMessage, ChatResponse, LLMProvider, Role


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class AgentConfig:
    name: str
    system_prompt: str = ""
    max_iterations: int = 10
    temperature: float = 0.3


@dataclass
class AgentStep:
    thought: str
    action: str | None
    observation: str
    response: ChatResponse


class Agent:
    """A simple tool-calling agent loop."""

    def __init__(self, provider: LLMProvider, config: AgentConfig, tools: list[Tool] | None = None):
        self.provider = provider
        self.config = config
        self.tools = {t.name: t for t in (tools or [])}
        self.history: list[ChatMessage] = []
        self.steps: list[AgentStep] = []

    def _system_message(self) -> ChatMessage:
        tool_desc = ""
        if self.tools:
            tool_desc = "\n\nAvailable tools:\n"
            for t in self.tools.values():
                tool_desc += f"- {t.name}: {t.description}\n"
        return ChatMessage(
            role=Role.SYSTEM,
            content=self.config.system_prompt + tool_desc,
        )

    async def run(self, user_input: str) -> str:
        self.history = [self._system_message()]
        self.steps = []
        self.history.append(ChatMessage(role=Role.USER, content=user_input))

        for _ in range(self.config.max_iterations):
            resp = await self.provider.chat(self.history)
            content = resp.content

            # Simple action detection: look for ACTION: tool_name(args)
            action_result = self._try_execute_action(content)
            if action_result is not None:
                self.steps.append(AgentStep(
                    thought=content,
                    action=action_result[0],
                    observation=action_result[1],
                    response=resp,
                ))
                self.history.append(ChatMessage(role=Role.ASSISTANT, content=content))
                self.history.append(ChatMessage(role=Role.USER, content=f"Observation: {action_result[1]}"))
                continue

            # No action => final answer
            self.steps.append(AgentStep(
                thought=content,
                action=None,
                observation="",
                response=resp,
            ))
            return content

        return "Max iterations reached without a final answer."

    def _try_execute_action(self, text: str) -> tuple[str, str] | None:
        """Extract and execute ACTION: tool_name(args) from model output."""
        import re
        match = re.search(r"ACTION:\s*(\w+)\(([^)]*)\)", text)
        if not match:
            return None
        tool_name = match.group(1)
        args_str = match.group(2).strip()
        if tool_name not in self.tools:
            return tool_name, f"Error: unknown tool '{tool_name}'"
        try:
            result = self.tools[tool_name].func(args_str)
            return tool_name, str(result)
        except Exception as e:
            return tool_name, f"Error: {e}"
