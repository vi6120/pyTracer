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


# --- Runs that left their recordings (partner study, gap 3) ----------------------

def _scenario(*, passed: bool, fp: str | None = None, error: str | None = None,
              unrecorded: list[str] | None = None, evaluated: bool = False) -> dict[str, Any]:
    return {
        "name": "s", "passed": passed, "fingerprint": fp, "error_type": error,
        "unrecorded_calls": unrecorded or [], "evaluated_changed_prompts": evaluated,
        "assertions": [],
    }


BASE = _run([_scenario(passed=False, fp="bug")])  # reproduced the captured failure


def test_a_pass_that_left_its_recording_is_not_fixed() -> None:
    # Fallback mode: a changed tool call ran live and the run "passed". Its outcome
    # came from live calls, so it cannot be reported fixed (it was, before).
    live_pass = _run([_scenario(passed=True, unrecorded=["tool 'get_order'"])])
    assert classify_reproduction(BASE, live_pass) == {"s": Outcome.CANNOT_JUDGE}


def test_the_baselines_failure_showing_again_is_still_failing_despite_drift() -> None:
    # Partial mode: both refs drifted, but the candidate fails with exactly the
    # baseline's failure, so the bug did reproduce (it was cannot_reproduce).
    drift = ["llm 'user_llm'"]
    base = _run([_scenario(passed=False, fp="bug", unrecorded=drift)])
    cand = _run([_scenario(passed=False, fp="bug", unrecorded=drift)])
    assert classify_reproduction(base, cand) == {"s": Outcome.STILL_FAILING}


def test_a_baseline_that_left_its_recording_cannot_reproduce_even_if_the_candidate_passes() -> None:
    base = _run([_scenario(passed=False, fp="other", unrecorded=["llm 'user_llm'"])])
    cand = _run([_scenario(passed=True)])
    assert classify_reproduction(base, cand) == {"s": Outcome.CANNOT_REPRODUCE}


def test_two_failures_on_a_missing_recording_are_not_the_same_bug() -> None:
    miss = _scenario(passed=False, fp="miss", error="MissingMockError", unrecorded=["llm 'a'"])
    assert classify_reproduction(_run([miss]), _run([dict(miss)])) == {"s": Outcome.CANNOT_REPRODUCE}


def test_a_model_change_evaluated_live_on_request_is_judged() -> None:
    changed = ["llm 'agent'"]
    fixed = _run([_scenario(passed=True, unrecorded=changed, evaluated=True)])
    other = _run([_scenario(passed=False, fp="new", unrecorded=changed, evaluated=True)])
    same = _run([_scenario(passed=False, fp="bug", unrecorded=changed, evaluated=True)])
    assert classify_reproduction(BASE, fixed) == {"s": Outcome.FIXED}
    assert classify_reproduction(BASE, other) == {"s": Outcome.DIVERGED}
    assert classify_reproduction(BASE, same) == {"s": Outcome.STILL_FAILING}


def test_a_tool_miss_is_not_excused_by_evaluating_model_changes() -> None:
    cand = _run([_scenario(passed=True, unrecorded=["tool 'exchange'", "llm 'agent'"], evaluated=True)])
    assert classify_reproduction(BASE, cand) == {"s": Outcome.CANNOT_JUDGE}
