"""Failure fingerprinting.

A fingerprint identifies a *failure*, not a run: it is derived from the sequence
of op/tool calls leading to the error plus the error type, and deliberately
excludes ids, timestamps, branch, and commit. So the same underlying bug hit on
two different branches produces the same fingerprint (and is deduplicated),
while a genuinely different failure produces a different one.
"""

from __future__ import annotations

import hashlib
import re

from pytracer_sdk.schema import Span, SpanStatus

from .engine import ReplayResult
from .hashing import canonical_json


def _short_hash(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]


_HEX = re.compile(r"\b(?:0x)?[0-9a-fA-F]{6,}\b")
_NUM = re.compile(r"\d+")


def _normalize_message(message: str | None) -> str:
    """Mask volatile values (hex ids, numbers) in an error message.

    So the same failure fingerprints the same across runs, while a genuinely
    different message (a different bug of the same type) still differs.
    """
    if not message:
        return ""
    masked = _HEX.sub("<hex>", message)
    masked = _NUM.sub("#", masked)
    return masked[:200]


def failure_fingerprint(
    spans: list[Span],
    *,
    agent_error_type: str | None = None,
    agent_error_message: str | None = None,
) -> str | None:
    """Fingerprint the failure in ``spans``.

    Keys on the **innermost** errored span (the origin, deepest and latest to
    start), the path to it, its error type, and a normalized message, so the same
    failure matches across branches while a different failure, even of the same
    type at the same op, does not. Returns ``None`` when there is no failure; when
    the agent raised outside any span, pass ``agent_error_type`` (and message).
    """
    ordered = sorted(spans, key=lambda s: s.start_time_ns)
    errored = [s for s in ordered if s.status is SpanStatus.ERROR]

    if errored:
        innermost = max(errored, key=lambda s: s.start_time_ns)
        sequence = [
            s.tool_name or s.name for s in ordered if s.start_time_ns <= innermost.start_time_ns
        ]
        return _short_hash(
            {
                "seq": sequence,
                "error_type": innermost.error_type,
                "at": innermost.tool_name or innermost.name,
                "message": _normalize_message(innermost.error_message),
            }
        )

    if agent_error_type is not None:
        sequence = [s.tool_name or s.name for s in ordered]
        return _short_hash(
            {
                "seq": sequence,
                "error_type": agent_error_type,
                "at": "<agent>",
                "message": _normalize_message(agent_error_message),
            }
        )

    return None


def fingerprint(result: ReplayResult) -> str | None:
    """Fingerprint a replay result's failure, or ``None`` if it did not fail."""
    if not result.failed:
        return None
    return failure_fingerprint(
        result.spans,
        agent_error_type=result.error_type,
        agent_error_message=result.error_message,
    )
