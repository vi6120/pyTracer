"""Failure fingerprint: branch-independent, discriminates distinct failures."""

from __future__ import annotations

from pytracer_sdk import SpanKind, SpanStatus
from replay_helpers import failing_result, passing_result, span

from pytracer_replay import failure_fingerprint, fingerprint


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


def test_same_type_but_different_message_differs() -> None:
    # A different bug of the same exception type must not read as "still failing".
    a = fingerprint(failing_result(["fetch", "parse"], "ValueError", "bad field: name"))
    b = fingerprint(failing_result(["fetch", "parse"], "ValueError", "bad field: age"))
    assert a != b


def test_message_volatile_values_are_ignored() -> None:
    # The same failure across runs (volatile numbers/ids differ) still matches.
    a = fingerprint(failing_result(["x"], "ValueError", "timeout after 1234 ms id=0xABCDEF01"))
    b = fingerprint(failing_result(["x"], "ValueError", "timeout after 5678 ms id=0x12345678"))
    assert a == b


def test_innermost_error_decides_the_fingerprint() -> None:
    # Two runs whose deepest failure differs (different op / type) must differ,
    # even though an outer agent span also errors in both.
    a = failure_fingerprint(
        [
            span("agent", kind=SpanKind.AGENT, status=SpanStatus.ERROR, error_type="RuntimeError", t=0),
            span("search", tool="search", status=SpanStatus.ERROR, error_type="KeyError", t=1),
        ]
    )
    b = failure_fingerprint(
        [
            span("agent", kind=SpanKind.AGENT, status=SpanStatus.ERROR, error_type="RuntimeError", t=0),
            span("fetch", tool="fetch", status=SpanStatus.ERROR, error_type="KeyError", t=1),
        ]
    )
    assert a is not None and b is not None
    assert a != b  # keyed on the innermost op (search vs fetch), not the shared outer agent
