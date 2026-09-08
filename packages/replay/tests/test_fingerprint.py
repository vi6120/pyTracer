"""Failure fingerprint: branch-independent, discriminates distinct failures."""

from __future__ import annotations

from replay_helpers import failing_result, passing_result, span

from navik_replay import failure_fingerprint, fingerprint


def test_no_failure_has_no_fingerprint() -> None:
    assert fingerprint(passing_result(["a", "b"])) is None
    assert failure_fingerprint([span("a"), span("b")]) is None


def test_same_failure_across_branches_matches() -> None:
    # Same call sequence + error type, but different ids/timestamps (i.e. branches).
    a = failing_result(["fetch", "parse"], "KeyError")
    b = failing_result(["fetch", "parse"], "KeyError")
    fp_a = fingerprint(a)
    fp_b = fingerprint(b)
    assert fp_a is not None
    assert fp_a == fp_b  # deduped to one issue


def test_different_error_type_differs() -> None:
    a = fingerprint(failing_result(["fetch", "parse"], "KeyError"))
    b = fingerprint(failing_result(["fetch", "parse"], "ValueError"))
    assert a != b


def test_different_call_sequence_differs() -> None:
    a = fingerprint(failing_result(["fetch", "parse"], "KeyError"))
    b = fingerprint(failing_result(["fetch", "validate", "parse"], "KeyError"))
    assert a != b


def test_agent_level_error_without_span_still_fingerprints() -> None:
    fp = failure_fingerprint([span("a"), span("b")], agent_error_type="TimeoutError")
    assert fp is not None
