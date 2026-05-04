"""Evaluation metrics for LLM output quality."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


@dataclass
class MetricResult:
    name: str
    score: float  # 0.0 - 1.0
    details: str = ""


class MetricSuite:
    """A collection of evaluation metrics for LLM responses."""

    def __init__(self):
        self.metrics: list[tuple[str, Callable[[str, str], MetricResult]]] = []

    def add_exact_match(self):
        def _exact(response: str, expected: str) -> MetricResult:
            match = response.strip().lower() == expected.strip().lower()
            return MetricResult(name="exact_match", score=1.0 if match else 0.0)
        self.metrics.append(("exact_match", _exact))
        return self

    def add_contains(self):
        def _contains(response: str, expected: str) -> MetricResult:
            found = expected.lower() in response.lower()
            return MetricResult(name="contains", score=1.0 if found else 0.0)
        self.metrics.append(("contains", _contains))
        return self

    def add_length_ratio(self, ideal_length: int = 200):
        def _length(response: str, _expected: str) -> MetricResult:
            ratio = len(response) / ideal_length
            score = max(0, 1.0 - abs(1.0 - ratio) * 0.5)
            return MetricResult(
                name="length_ratio",
                score=score,
                details=f"length={len(response)}, ideal={ideal_length}",
            )
        self.metrics.append(("length_ratio", _length))
        return self

    def add_coherence(self):
        """Simple heuristic: penalize excessive repetition."""
        def _coherence(response: str, _expected: str) -> MetricResult:
            sentences = re.split(r'[.!?]\s+', response)
            if len(sentences) < 2:
                return MetricResult(name="coherence", score=0.5, details="too short to evaluate")
            unique = len(set(sentences))
            ratio = unique / len(sentences)
            return MetricResult(
                name="coherence",
                score=min(1.0, ratio * 1.2),
                details=f"{unique}/{len(sentences)} unique sentences",
            )
        self.metrics.append(("coherence", _coherence))
        return self

    def evaluate(self, response: str, expected: str = "") -> list[MetricResult]:
        return [fn(response, expected) for _, fn in self.metrics]

    def avg_score(self, response: str, expected: str = "") -> float:
        results = self.evaluate(response, expected)
        if not results:
            return 0.0
        return sum(r.score for r in results) / len(results)
