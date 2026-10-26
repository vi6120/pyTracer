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


def _assertion_failure(detail: str, *, name: str = "contains") -> dict[str, Any]:
    # A wrong answer: nothing raised (no fingerprint), an assertion failed.
    return {
        "name": "s", "passed": False, "fingerprint": None, "error_type": None,
        "assertions": [
            {"name": "no_errors", "passed": True, "detail": ""},
            {"name": name, "passed": False, "detail": detail},
        ],
    }


def test_still_failing_when_the_same_assertion_fails_the_same_way() -> None:
    baseline = _run([_assertion_failure("'Reviewed 3 sources' not found")])
    candidate = _run([_assertion_failure("'Reviewed 3 sources' not found")])
    assert classify_reproduction(baseline, candidate) == {"s": Outcome.STILL_FAILING}


def test_diverged_when_a_different_assertion_fails_or_fails_differently() -> None:
    baseline = _run([_assertion_failure("'Reviewed 3 sources' not found")])
    other_detail = _run([_assertion_failure("expected 'a', got 'b'")])
    other_assertion = _run([_assertion_failure("'Reviewed 3 sources' not found", name="exact_match")])
    assert classify_reproduction(baseline, other_detail) == {"s": Outcome.DIVERGED}
    assert classify_reproduction(baseline, other_assertion) == {"s": Outcome.DIVERGED}


def test_diverged_when_the_candidate_now_raises_instead() -> None:
    baseline = _run([_assertion_failure("'Reviewed 3 sources' not found")])
    candidate = _run([{"name": "s", "passed": False, "fingerprint": "fp1", "error_type": "KeyError"}])
    assert classify_reproduction(baseline, candidate) == {"s": Outcome.DIVERGED}
