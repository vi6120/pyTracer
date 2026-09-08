"""Trace and span identifier generation, OpenTelemetry-compatible.

OpenTelemetry uses a 128-bit trace id (32 hex chars) and a 64-bit span id
(16 hex chars), both lowercase hex. We generate the same shapes so spans
interoperate with existing OTel tooling and the OTLP wire format.
"""

from __future__ import annotations

import os

TRACE_ID_BYTES = 16  # 128-bit → 32 hex chars
SPAN_ID_BYTES = 8  # 64-bit → 16 hex chars


def new_trace_id() -> str:
    """Return a random 128-bit trace id as 32 lowercase hex characters."""
    return os.urandom(TRACE_ID_BYTES).hex()


def new_span_id() -> str:
    """Return a random 64-bit span id as 16 lowercase hex characters."""
    return os.urandom(SPAN_ID_BYTES).hex()


def is_valid_trace_id(value: str) -> bool:
    """True if ``value`` is a valid, non-zero 32-char hex trace id."""
    return _is_hex(value, 2 * TRACE_ID_BYTES)


def is_valid_span_id(value: str) -> bool:
    """True if ``value`` is a valid, non-zero 16-char hex span id."""
    return _is_hex(value, 2 * SPAN_ID_BYTES)


def _is_hex(value: str, length: int) -> bool:
    if len(value) != length:
        return False
    try:
        # All-zero ids are "invalid"/absent per the OTel spec.
        return int(value, 16) != 0
    except ValueError:
        return False
