"""Three-way cross-branch classifier: fixed / still failing / diverged."""

from __future__ import annotations

from replay_helpers import failing_result, passing_result

from navik_replay import Outcome, classify


def test_fixed_when_candidate_passes() -> None:
    baseline = failing_result(["fetch", "parse"], "KeyError")
    candidate = passing_result(["fetch", "parse"])
    assert classify(baseline, candidate) is Outcome.FIXED


def test_still_failing_when_same_fingerprint() -> None:
    baseline = failing_result(["fetch", "parse"], "KeyError")
    candidate = failing_result(["fetch", "parse"], "KeyError")
    assert classify(baseline, candidate) is Outcome.STILL_FAILING


def test_diverged_when_different_failure() -> None:
    baseline = failing_result(["fetch", "parse"], "KeyError")
    candidate = failing_result(["fetch", "parse"], "ValueError")
    assert classify(baseline, candidate) is Outcome.DIVERGED

    candidate2 = failing_result(["fetch", "validate", "parse"], "KeyError")
    assert classify(baseline, candidate2) is Outcome.DIVERGED
