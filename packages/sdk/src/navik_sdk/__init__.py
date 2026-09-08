"""Navik SDK — capture agent runs as OpenTelemetry-compliant spans.

Quick start::

    import navik_sdk as navik

    @navik.op(kind=navik.SpanKind.LLM)
    def call_model(prompt: str) -> str:
        ...

By default spans go to an in-process :class:`InMemoryTransport`. Call
:func:`configure` to point the SDK at the ingestion gateway (Layer 2) and to
stamp every span with the current service name, branch, and commit.
"""

from __future__ import annotations

from typing import Any

from .buffer import SpanBuffer
from .ids import new_span_id, new_trace_id
from .redaction import DEFAULT_RULES, RedactionRule, Redactor
from .schema import Resource, Span, SpanKind, SpanStatus
from .tracer import ActiveSpan, SpanContext, Tracer
from .transport import HTTPTransport, InMemoryTransport, Transport, TransportError

__all__ = [
    "DEFAULT_RULES",
    "ActiveSpan",
    "HTTPTransport",
    "InMemoryTransport",
    "RedactionRule",
    "Redactor",
    "Resource",
    "Span",
    "SpanBuffer",
    "SpanContext",
    "SpanKind",
    "SpanStatus",
    "Tracer",
    "Transport",
    "TransportError",
    "configure",
    "flush",
    "get_tracer",
    "new_span_id",
    "new_trace_id",
    "op",
    "shutdown",
    "start_span",
]

__version__ = "0.0.1"

_default_tracer: Tracer | None = None


def configure(
    transport: Transport | None = None,
    *,
    service_name: str | None = None,
    branch: str | None = None,
    commit: str | None = None,
    environment: str | None = None,
    redactor: Redactor | None = None,
) -> Tracer:
    """Create (or replace) the module-level default tracer and return it."""
    global _default_tracer
    resource = Resource(
        service_name=service_name or "unknown-agent",
        branch=branch,
        commit=commit,
        environment=environment,
    )
    _default_tracer = Tracer(
        transport or InMemoryTransport(),
        resource=resource,
        redactor=redactor,
    )
    return _default_tracer


def get_tracer() -> Tracer:
    """Return the default tracer, creating an in-memory one on first use."""
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = Tracer(InMemoryTransport())
    return _default_tracer


def op(func: Any = None, **kwargs: Any) -> Any:
    """Module-level ``@op`` delegating to the default tracer."""
    return get_tracer().op(func, **kwargs)


def start_span(name: str, **kwargs: Any) -> ActiveSpan:
    """Open a span on the default tracer (use as a context manager)."""
    return get_tracer().start_span(name, **kwargs)


def flush(timeout: float | None = None) -> None:
    """Flush the default tracer's buffer."""
    get_tracer().flush(timeout)


def shutdown(timeout: float = 5.0) -> None:
    """Drain and stop the default tracer's background worker."""
    get_tracer().shutdown(timeout)
