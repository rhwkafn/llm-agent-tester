# llm-agent-tester

A lightweight Python framework for **benchmarking LLM providers** and **testing multi-agent orchestration patterns**.

Born from real-world experience building production AI agent systems — I needed a systematic way to compare providers, validate agent behaviors, and catch regressions across model updates.

## Features

- **Multi-Provider Abstraction** — Unified interface for OpenAI, Zhipu (GLM), DeepSeek, Anthropic, and any OpenAI-compatible API
- **ReAct Agent** — Built-in Reasoning + Acting agent with pluggable tool support
- **Multi-Agent Orchestration** — Route tasks to specialized sub-agents via a router agent
- **Benchmark Runner** — Run test cases across providers with latency/throughput metrics
- **Evaluation Metrics** — Exact match, contains, length ratio, coherence scoring
- **Rich Terminal Output** — Beautiful tables and comparison views via `rich`

## Quick Start

```bash
pip install -e ".[dev]"

# Set your API keys
export ZHIPU_API_KEY="your-zhipu-key"
export DEEPSEEK_API_KEY="your-deepseek-key"

# Run benchmark
python examples/quick_start.py

# Run agent demo
python examples/agent_demo.py

# Run multi-agent demo
python examples/multi_agent.py
```

## Architecture

```
src/llm_tester/
├── providers/          # LLM provider implementations
│   ├── base.py         #   Abstract base + data types
│   ├── openai_provider.py    #   OpenAI / compatible
│   ├── zhipu_provider.py     #   Zhipu GLM (JWT auth)
│   ├── deepseek_provider.py  #   DeepSeek
│   └── anthropic_provider.py #   Anthropic Claude
├── agents/             # Agent implementations
│   ├── base.py         #   Tool-calling agent loop
│   ├── react_agent.py  #   ReAct paradigm
│   └── multi_agent.py  #   Orchestrator + routing
├── benchmarks/         # Testing & evaluation
│   ├── runner.py       #   Cross-provider benchmark runner
│   └── metrics.py      #   Output quality metrics
└── utils/              # Config & display
    ├── config.py       #   YAML / env config loading
    └── display.py      #   Rich terminal output
```

## Provider Comparison

Run the same test suite across multiple providers and compare:

```python
from llm_tester.providers import ZhipuProvider, DeepSeekProvider
from llm_tester.benchmarks import BenchmarkRunner, TestCase

runner = BenchmarkRunner([
    ZhipuProvider(zhipu_config),
    DeepSeekProvider(deepseek_config),
])

results = await runner.run_suite([
    TestCase(name="reasoning", messages=[...], expected_contains="42"),
    TestCase(name="code_gen", messages=[...], max_latency_ms=5000),
])

print(runner.summary_table())
```

## Agent Testing

Test agent behavior with tool use:

```python
from llm_tester.agents import ReActAgent
from llm_tester.agents.base import Tool

agent = ReActAgent(provider, tools=[
    Tool(name="search", description="Search the web", ...),
    Tool(name="calc", description="Calculate math", ...),
])

result = await agent.run("What is the GDP of Japan divided by its population?")
# Agent will reason, call tools, and synthesize an answer
```

## Multi-Agent Orchestration

Route complex tasks to domain-specific agents:

```python
from llm_tester.agents import MultiAgentOrchestrator, AgentSpec

orchestrator = MultiAgentOrchestrator(provider, [
    AgentSpec(name="coder", role_description="Handles code tasks"),
    AgentSpec(name="analyst", role_description="Handles data analysis"),
    AgentSpec(name="writer", role_description="Handles writing tasks"),
])

result = await orchestrator.run("Write a sorting algorithm")
# Automatically routes to "coder" agent
```

## Configuration

Use environment variables or a YAML config:

```yaml
# config.yaml
providers:
  - name: zhipu
    type: zhipu
    api_key: ${ZHIPU_API_KEY}
    base_url: https://open.bigmodel.cn/api/paas/v4
    model: glm-4-flash
  - name: deepseek
    type: deepseek
    api_key: ${DEEPSEEK_API_KEY}
    base_url: https://api.deepseek.com
    model: deepseek-chat
```

## Running Tests

```bash
pytest tests/ -v
```

## Why I Built This

In my work with LLM-powered agent systems, I constantly needed to:
- Compare how different models handle the same prompts (latency, quality, cost)
- Validate that agent tool-calling patterns work correctly across providers
- Benchmark before and after prompt engineering changes
- Test multi-agent routing logic in isolation

This framework codifies those patterns into a reusable toolkit.

## License

MIT
