"""Hermetic tests for cross-branch reproduction orchestration."""

from __future__ import annotations

from typing import Any

import pytest
from pytracer_replay import Outcome
from pytracer_runner import classify_reproduction

from pytracer_cli.reproduce import all_fixed, format_verdicts, run_reproduction


class _Result:
    def __init__(self, results: Any, stderr: str = "") -> None:
        self.results = results
        self.stderr = stderr


class FakeRunner:
    def __init__(self, by_ref: dict[str, Any]) -> None:
        self._by_ref = by_ref
        self.calls: list[tuple[str, str, Any, Any]] = []

    def reproduce(
        self, ref: str, collection_path: str, *, frozen_trace: Any = None, trace_dest: Any = None
    ) -> _Result:
        self.calls.append((ref, collection_path, frozen_trace, trace_dest))
        return _Result(self._by_ref[ref])


def _run(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    return {"scenarios": scenarios}


def test_fixed_when_candidate_passes() -> None:
    runner = FakeRunner(
        {
            "main": _run([{"name": "s", "passed": False, "fingerprint": "fp1"}]),
            "feature": _run([{"name": "s", "passed": True, "fingerprint": None}]),
        }
    )
    _, _, verdicts = run_reproduction(
        runner, collection_path="c.yaml", baseline_ref="main", candidate_ref="feature",
        classify=classify_reproduction,
    )
    assert verdicts["s"] is Outcome.FIXED
    assert all_fixed(verdicts)


def test_diverged_when_new_failure() -> None:
    runner = FakeRunner(
        {
            "main": _run([{"name": "s", "passed": False, "fingerprint": "fp1"}]),
            "feature": _run([{"name": "s", "passed": False, "fingerprint": "fp2"}]),
        }
    )
    _, _, verdicts = run_reproduction(
        runner, collection_path="c.yaml", baseline_ref="main", candidate_ref="feature",
        classify=classify_reproduction,
    )
    assert verdicts["s"] is Outcome.DIVERGED
    assert not all_fixed(verdicts)


def test_frozen_trace_is_passed_through() -> None:
    same = _run([{"name": "s", "passed": True}])
    runner = FakeRunner({"main": same, "feature": same})
    run_reproduction(
        runner, collection_path="c.yaml", baseline_ref="main", candidate_ref="feature",
        classify=classify_reproduction, frozen_trace="base.jsonl", trace_dest="tests/t.jsonl",
    )
    assert runner.calls[0][2] == "base.jsonl"
    assert runner.calls[0][3] == "tests/t.jsonl"


def test_missing_results_raises() -> None:
    runner = FakeRunner({"main": None, "feature": _run([{"name": "s", "passed": True}])})
    with pytest.raises(RuntimeError):
        run_reproduction(
            runner, collection_path="c.yaml", baseline_ref="main", candidate_ref="feature",
            classify=classify_reproduction,
        )


def test_format_verdicts() -> None:
    table = format_verdicts({"a": Outcome.FIXED, "b": Outcome.DIVERGED})
    assert "fixed" in table
    assert "diverged" in table
    assert "SCENARIO" in table
