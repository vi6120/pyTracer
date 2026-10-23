"""Cross-branch classification from two `pytracer run --json` outputs."""

from __future__ import annotations

from typing import Any

from pytracer_replay import Outcome

from pytracer_runner import classify_reproduction


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


def test_cannot_judge_when_candidate_changed_a_model_input() -> None:
    # The baseline reproduced the failure, but the candidate edited a prompt, so a
    # frozen replay cannot judge the fix: not a diverge (X4).
    baseline = _run([{"name": "s", "passed": False, "fingerprint": "fp1"}])
    candidate = _run(
        [{"name": "s", "passed": False, "fingerprint": "fp2", "changed_models": ["call_model"]}]
    )
    assert classify_reproduction(baseline, candidate) == {"s": Outcome.CANNOT_JUDGE}


def test_cannot_reproduce_when_neither_ref_matched_its_recordings() -> None:
    # Both refs drifted (e.g. the run's env was missing), so the baseline failure
    # was never reproduced and there is nothing to compare (X2/X5).
    base = _run([{"name": "s", "passed": False, "changed_models": ["call_model"]}])
    cand = _run([{"name": "s", "passed": False, "changed_models": ["call_model"]}])
    assert classify_reproduction(base, cand) == {"s": Outcome.CANNOT_REPRODUCE}


def test_cannot_reproduce_when_both_miss_recordings() -> None:
    base = _run([{"name": "s", "passed": False, "error_type": "MissingMockError"}])
    cand = _run([{"name": "s", "passed": False, "error_type": "MissingMockError"}])
    assert classify_reproduction(base, cand) == {"s": Outcome.CANNOT_REPRODUCE}
