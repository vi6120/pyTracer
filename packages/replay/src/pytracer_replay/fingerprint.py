"""Failure fingerprinting.

A fingerprint identifies a *failure*, not a run: it is derived from the sequence
of op/tool calls leading to the error plus the error type, and deliberately
excludes ids, timestamps, branch, and commit. So the same underlying bug hit on
two different branches produces the same fingerprint (and is deduplicated),
while a genuinely different failure produces a different one.
"""

from __future__ import annotations

import hashlib

from pytracer_sdk.schema import Span, SpanStatus

from .engine import ReplayResult
from .hashing import canonical_json


def _short_hash(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def failure_fingerprint(
    spans: list[Span], *, agent_error_type: str | None = None
) -> str | None:
    """Fingerprint the first failure in ``spans``.

    Returns ``None`` when there is no failure. When the agent raised outside any
    span, pass ``agent_error_type`` so the failure is still fingerprinted.
    """
    ordered = sorted(spans, key=lambda s: s.start_time_ns)
    errored = [s for s in ordered if s.status is SpanStatus.ERROR]

    if errored:
        first = errored[0]
        sequence: list[str] = []
        for span in ordered:
            sequence.append(span.tool_name or span.name)
            if span.span_id == first.span_id:
                break
        return _short_hash(
            {"seq": sequence, "error_type": first.error_type, "at": first.tool_name or first.name}
        )

    if agent_error_type is not None:
        sequence = [s.tool_name or s.name for s in ordered]
        return _short_hash({"seq": sequence, "error_type": agent_error_type, "at": "<agent>"})

    return None


def fingerprint(result: ReplayResult) -> str | None:
    """Fingerprint a replay result's failure, or ``None`` if it did not fail."""
    if not result.failed:
        return None
    return failure_fingerprint(result.spans, agent_error_type=result.error_type)
