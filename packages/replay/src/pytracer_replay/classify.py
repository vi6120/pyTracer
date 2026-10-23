"""Three-way cross-branch outcome classifier.

Given a known-failing baseline run and a replay of the same scenario on a
candidate branch (model version and recorded tools held frozen, only the agent
code varying), classify the candidate as fixed, still failing, or diverged.
A plain pass/fail flag cannot tell a team whether their branch introduced a new
problem - this can.
"""

from __future__ import annotations

from enum import Enum

from .engine import ReplayResult
from .fingerprint import fingerprint


class Outcome(str, Enum):
    """Result of comparing a candidate branch's replay to a failing baseline."""

    FIXED = "fixed"  # candidate no longer fails
    STILL_FAILING = "still_failing"  # candidate fails the same way (same fingerprint)
    DIVERGED = "diverged"  # candidate fails, but differently than the baseline
    # A model input changed on the candidate (a prompt edit), which a frozen replay
    # cannot judge: the recording no longer applies, so this is not a real verdict.
    CANNOT_JUDGE = "cannot_judge"
    # Neither ref matched its recordings (e.g. the run's env was not supplied, so
    # every model call drifted): the baseline failure was never reproduced, so there
    # is nothing to compare.
    CANNOT_REPRODUCE = "cannot_reproduce"


def classify(baseline: ReplayResult, candidate: ReplayResult) -> Outcome:
    """Classify a candidate replay against a known-failing baseline."""
    if not candidate.failed:
        return Outcome.FIXED
    baseline_fp = fingerprint(baseline)
    candidate_fp = fingerprint(candidate)
    if baseline_fp is not None and baseline_fp == candidate_fp:
        return Outcome.STILL_FAILING
    return Outcome.DIVERGED
