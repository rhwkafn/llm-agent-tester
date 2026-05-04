"""Tests for evaluation metrics."""

from llm_tester.benchmarks.metrics import MetricSuite


def test_exact_match():
    suite = MetricSuite().add_exact_match()
    results = suite.evaluate("hello world", "hello world")
    assert results[0].score == 1.0


def test_exact_match_fail():
    suite = MetricSuite().add_exact_match()
    results = suite.evaluate("hello", "world")
    assert results[0].score == 0.0


def test_contains():
    suite = MetricSuite().add_contains()
    results = suite.evaluate("The answer is 42", "42")
    assert results[0].score == 1.0


def test_contains_fail():
    suite = MetricSuite().add_contains()
    results = suite.evaluate("The answer is 42", "99")
    assert results[0].score == 0.0


def test_length_ratio():
    suite = MetricSuite().add_length_ratio(ideal_length=100)
    results = suite.evaluate("x" * 100)
    assert results[0].score == 1.0


def test_coherence():
    suite = MetricSuite().add_coherence()
    text = "The sky is blue. Grass is green. Water is clear."
    results = suite.evaluate(text)
    assert results[0].score > 0.5


def test_multiple_metrics():
    suite = (
        MetricSuite()
        .add_exact_match()
        .add_contains()
        .add_length_ratio()
        .add_coherence()
    )
    results = suite.evaluate("The answer is 42", "42")
    assert len(results) == 4
    assert suite.avg_score("The answer is 42", "42") > 0


def test_avg_score_empty():
    suite = MetricSuite()
    assert suite.avg_score("test") == 0.0
