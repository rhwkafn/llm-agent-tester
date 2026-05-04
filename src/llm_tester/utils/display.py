"""Rich terminal display helpers."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from ..benchmarks.runner import BenchmarkResult

console = Console()


def print_results_table(results: list[BenchmarkResult], title: str = "Benchmark Results"):
    table = Table(title=title, show_lines=True)
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Test", style="white")
    table.add_column("Latency", justify="right")
    table.add_column("TPS", justify="right", style="green")
    table.add_column("Tokens", justify="right")
    table.add_column("Status", justify="center")

    for r in results:
        status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        table.add_row(
            r.provider,
            r.model,
            r.test_name,
            f"{r.latency_ms:.0f}ms",
            f"{r.tokens_per_second:.1f}",
            f"{r.prompt_tokens}+{r.completion_tokens}",
            status,
        )

    console.print(table)


def print_comparison(results: list[BenchmarkResult]):
    """Print a grouped comparison by test case."""
    from collections import defaultdict
    grouped: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for r in results:
        grouped[r.test_name].append(r)

    for test_name, group in grouped.items():
        console.print(f"\n[bold]{test_name}[/bold]")
        table = Table(show_header=True)
        table.add_column("Provider", style="cyan")
        table.add_column("Latency", justify="right")
        table.add_column("TPS", justify="right", style="green")
        table.add_column("Response Preview", max_width=60)

        sorted_group = sorted(group, key=lambda x: x.latency_ms)
        for r in sorted_group:
            table.add_row(
                f"{r.provider} ({r.model})",
                f"{r.latency_ms:.0f}ms",
                f"{r.tokens_per_second:.1f}",
                r.response_preview[:80] + "..." if len(r.response_preview) > 80 else r.response_preview,
            )
        console.print(table)
