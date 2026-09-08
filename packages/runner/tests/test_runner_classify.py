"""Cross-branch classification from two `navik run --json` outputs."""

from __future__ import annotations

from typing import Any

from navik_replay import Outcome

from navik_runner import classify_reproduction


def _run(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    return {"collection": "c", "scenarios": scenarios}


def test_fixed_when_candidate_passes() -> None:
    baseline = _run([{"name": "s", "passed": False, "fingerprint": "fp1"}])
    candidate = _run([{"name": "s", "passed": True, "fingerprint": None}])
    assert classify_reproduction(baseline, candidate) == {"s": Outcome.FIXED}


def test_still_failing_when_same_fingerprint() -> None:
    baseline = _run([{"name": "s", "passed": False, "fingerprint": "fp1"}])
    candidate = _run([{"name": "s", "passed": False, "fingerprint": "fp1"}])
    assert classify_reproduction(baseline, candidate) == {"s": Outcome.STILL_FAILING}


def test_diverged_when_different_fingerprint() -> None:
    baseline = _run([{"name": "s", "passed": False, "fingerprint": "fp1"}])
    candidate = _run([{"name": "s", "passed": False, "fingerprint": "fp2"}])
    assert classify_reproduction(baseline, candidate) == {"s": Outcome.DIVERGED}


def test_diverged_when_no_matching_baseline() -> None:
    baseline = _run([])
    candidate = _run([{"name": "s", "passed": False, "fingerprint": "fp"}])
    assert classify_reproduction(baseline, candidate) == {"s": Outcome.DIVERGED}
