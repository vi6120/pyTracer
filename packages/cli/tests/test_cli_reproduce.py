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
        self.calls: list[tuple[str, str, Any, Any, Any, Any]] = []
        self.evaluate_flags: list[bool] = []
        self.install_flags: list[bool | None] = []

    def reproduce(
        self,
        ref: str,
        collection_path: str,
        *,
        frozen_trace: Any = None,
        trace_dest: Any = None,
        env_file: Any = None,
        env_dest: Any = ".env",
        evaluate_changed_prompts: bool = False,
        install: bool | None = None,
    ) -> _Result:
        self.calls.append((ref, collection_path, frozen_trace, trace_dest, env_file, env_dest))
        self.evaluate_flags.append(evaluate_changed_prompts)
        self.install_flags.append(install)
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


def test_env_file_is_passed_through() -> None:
    same = _run([{"name": "s", "passed": True}])
    runner = FakeRunner({"main": same, "feature": same})
    run_reproduction(
        runner, collection_path="c.yaml", baseline_ref="main", candidate_ref="feature",
        classify=classify_reproduction, env_file="prod.env", env_dest="config/.env",
    )
    assert runner.calls[0][4] == "prod.env"
    assert runner.calls[0][5] == "config/.env"


def test_evaluate_changed_prompts_is_passed_to_both_refs() -> None:
    same = _run([{"name": "s", "passed": True}])
    runner = FakeRunner({"main": same, "feature": same})
    run_reproduction(
        runner, collection_path="c.yaml", baseline_ref="main", candidate_ref="feature",
        classify=classify_reproduction, evaluate_changed_prompts=True,
    )
    assert runner.evaluate_flags == [True, True]


def test_install_choice_is_passed_to_both_refs() -> None:
    same = _run([{"name": "s", "passed": True}])
    for choice in (None, True, False):
        runner = FakeRunner({"main": same, "feature": same})
        run_reproduction(
            runner, collection_path="c.yaml", baseline_ref="main", candidate_ref="feature",
            classify=classify_reproduction, install=choice,
        )
        assert runner.install_flags == [choice, choice]


def test_missing_results_error_shows_full_stderr() -> None:
    # The real cause (here at the end of a long stderr) must be shown, not cut at 200.
    class _R:
        def __init__(self, results: Any, stderr: str = "") -> None:
            self.results = results
            self.stderr = stderr

    class Runner:
        def reproduce(self, ref: str, collection_path: str, **_: Any) -> Any:
            if ref == "main":
                return _R(None, "pip noise\n" + ("x" * 500) + "\nNo module named pytracer_cli")
            return _R(_run([{"name": "s", "passed": True}]))

    with pytest.raises(RuntimeError) as exc:
        run_reproduction(
            Runner(), collection_path="c.yaml", baseline_ref="main", candidate_ref="feature",
            classify=classify_reproduction,
        )
    assert "No module named pytracer_cli" in str(exc.value)


def test_format_verdicts() -> None:
    table = format_verdicts({"a": Outcome.FIXED, "b": Outcome.DIVERGED})
    assert "fixed" in table
    assert "diverged" in table
    assert "SCENARIO" in table


def test_format_verdicts_hints_the_inconclusive_outcomes() -> None:
    table = format_verdicts({"a": Outcome.CANNOT_REPRODUCE, "b": Outcome.CANNOT_JUDGE})
    assert "cannot_reproduce" in table and "--env-file" in table
    assert "cannot_judge" in table and "--evaluate-changed-prompts" in table


def test_format_verdicts_prints_each_reason_instead_of_the_generic_hint() -> None:
    reason = "The branch calls tool 'exchange' with arguments the recording does not have."
    table = format_verdicts({"s": Outcome.CANNOT_JUDGE}, {"s": reason})
    assert f"s: {reason}" in table
    assert "evaluate-changed-prompts" not in table  # the generic prompt hint is not added
