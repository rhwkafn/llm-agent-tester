"""Multi-agent orchestrator with role-based delegation."""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import Agent, AgentConfig, Tool
from ..providers.base import LLMProvider


@dataclass
class AgentSpec:
    name: str
    role_description: str
    tools: list[Tool] = field(default_factory=list)
    system_prompt: str = ""


class MultiAgentOrchestrator:
    """Routes tasks to specialized sub-agents and synthesizes results."""

    def __init__(self, provider: LLMProvider, agents: list[AgentSpec]):
        self.provider = provider
        self.agents: dict[str, Agent] = {}
        for spec in agents:
            cfg = AgentConfig(
                name=spec.name,
                system_prompt=spec.system_prompt or f"You are {spec.name}. {spec.role_description}",
            )
            self.agents[spec.name] = Agent(provider, cfg, spec.tools)

        # Router agent decides which sub-agent to use
        self._router = Agent(
            provider,
            AgentConfig(
                name="router",
                system_prompt=self._build_router_prompt(agents),
                max_iterations=1,
            ),
        )

    def _build_router_prompt(self, agents: list[AgentSpec]) -> str:
        lines = ["You are a task router. Given a user request, decide which agent should handle it."]
        lines.append("Respond with ONLY the agent name, nothing else.\n")
        lines.append("Available agents:")
        for a in agents:
            lines.append(f"- {a.name}: {a.role_description}")
        return "\n".join(lines)

    async def run(self, task: str) -> dict[str, str]:
        """Route the task to the best agent and return its result."""
        # Step 1: route
        route_resp = await self._router.run(task)
        agent_name = route_resp.strip().lower()

        # Step 2: execute with selected agent (fallback to first if unknown)
        agent = self.agents.get(agent_name)
        if agent is None:
            # Try fuzzy match
            for name, a in self.agents.items():
                if agent_name in name or name in agent_name:
                    agent = a
                    break
            if agent is None:
                agent = next(iter(self.agents.values()))

        result = await agent.run(task)

        return {
            "routed_to": agent.name,
            "result": result,
            "steps": len(agent.steps),
        }
