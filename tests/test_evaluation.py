from __future__ import annotations

import pytest

from tinydevassist.evaluation import comparison_report, summarize_generations


def test_comparison_report_has_no_automatic_quality_metric():
    rows = [
        {
            "id": "git-005",
            "base": {
                "response": "base",
                "latency_seconds": 1.0,
                "input_tokens": 2,
                "generated_tokens": 3,
            },
            "adapter": {
                "response": "adapter",
                "latency_seconds": 2.0,
                "input_tokens": 2,
                "generated_tokens": 4,
            },
        }
    ]
    report = comparison_report(rows, model_id="model", revision="a" * 40, test_sha256="b" * 64)
    assert report["status"] == "measured_comparison"
    assert report["methodology"]["quality_metric"] is None
    assert report["summary"]["adapter"]["nonempty_responses"] == 1


def test_comparison_rejects_missing_generation_metrics():
    with pytest.raises(ValueError):
        comparison_report(
            [{"id": "a", "base": {"response": "x"}, "adapter": {"response": "y"}}],
            model_id="model",
            revision="a" * 40,
            test_sha256="b" * 64,
        )


def test_generation_summary_counts_blank_responses():
    rows = [
        {"adapter": {"response": "", "latency_seconds": 1.0, "generated_tokens": 0}},
        {"adapter": {"response": "ok", "latency_seconds": 3.0, "generated_tokens": 4}},
    ]
    assert summarize_generations(rows, "adapter") == {
        "samples": 2,
        "nonempty_responses": 1,
        "mean_latency_seconds": 2.0,
        "mean_generated_tokens": 2.0,
    }
