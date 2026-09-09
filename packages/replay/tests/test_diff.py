"""Structured diff between two runs."""

from __future__ import annotations

from pytracer_sdk import SpanStatus
from replay_helpers import span

from pytracer_replay import ReplayMode, ReplayResult, diff_results, diff_traces


def test_identical_traces() -> None:
    a = [span("x", tool="x", output=1, t=0), span("y", tool="y", output=2, t=1)]
    b = [span("x", tool="x", output=1, t=0), span("y", tool="y", output=2, t=1)]
    d = diff_traces(a, b)
    assert d["identical"] is True
    assert d["diverged_at"] is None


def test_output_divergence_is_located() -> None:
    a = [span("x", tool="x", output=1, t=0), span("y", tool="y", output=2, t=1)]
    b = [span("x", tool="x", output=1, t=0), span("y", tool="y", output=999, t=1)]
    d = diff_traces(a, b)
    assert d["identical"] is False
    assert d["diverged_at"] == 1
    assert d["steps"][1]["changed"] is True
    assert d["steps"][0]["changed"] is False


def test_different_lengths() -> None:
    a = [span("x", tool="x", t=0)]
    b = [span("x", tool="x", t=0), span("y", tool="y", t=1)]
    d = diff_traces(a, b)
    assert d["identical"] is False
    assert d["span_count"] == {"a": 1, "b": 2}
    assert d["steps"][1]["name_a"] is None
    assert d["steps"][1]["name_b"] == "y"


def test_diff_results_return_value_and_status() -> None:
    a = ReplayResult(return_value="hi", status=SpanStatus.OK, mode=ReplayMode.FULL)
    b = ReplayResult(
        return_value="bye", status=SpanStatus.ERROR, mode=ReplayMode.FULL, error_type="ValueError"
    )
    d = diff_results(a, b)
    assert d["return_value_changed"] is True
    assert d["status_changed"] is True
    assert d["status_a"] == "ok"
    assert d["status_b"] == "error"
