"""SpanRecorder - build spans from external framework events.

The ``@op`` decorator captures spans by nesting on the Python call stack, which
works when Navik controls the code flow. Framework integrations do not: agent
frameworks (LangGraph, CrewAI, AutoGen, the OpenAI Agents SDK) report work as
start/end callbacks that carry their own run ids and parent run ids, often from
different threads. This recorder bridges that model. An adapter calls
:meth:`start` when the framework begins an LLM call, tool call, or agent step,
and :meth:`end` when it finishes; the recorder maps the framework's run ids to
Navik trace/span ids, links parents explicitly, redacts, and buffers the span.

Every framework adapter is a thin translation of that framework's events into
these two calls.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from .ids import new_span_id, new_trace_id
from .schema import Span, SpanKind, SpanStatus
from .tracer import Tracer


class SpanRecorder:
    """Records spans from framework events keyed by the framework's run ids."""

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer
        self._lock = threading.Lock()
        self._open: dict[str, Span] = {}

    def start(
        self,
        run_id: str,
        *,
        name: str,
        kind: SpanKind = SpanKind.UNKNOWN,
        parent_run_id: str | None = None,
        agent_name: str | None = None,
        tool_name: str | None = None,
        input: Any | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Begin a span for a framework run.

        Trace and parent linkage come from ``parent_run_id``: if its span is
        open, this span joins that trace as a child; otherwise it starts a new
        trace as a root. A duplicate ``run_id`` overwrites the earlier open span.
        """
        redactor = self._tracer.redactor
        with self._lock:
            parent = self._open.get(parent_run_id) if parent_run_id is not None else None
            trace_id = parent.trace_id if parent is not None else new_trace_id()
            parent_span_id = parent.span_id if parent is not None else None
            span = Span(
                trace_id=trace_id,
                span_id=new_span_id(),
                parent_span_id=parent_span_id,
                name=name,
                kind=kind,
                agent_name=agent_name,
                tool_name=tool_name,
                input=redactor.redact(input) if input is not None else None,
                start_time_ns=time.time_ns(),
                attributes=dict(attributes or {}),
                resource=self._tracer.resource,
            )
            self._open[run_id] = span

    def end(
        self,
        run_id: str,
        *,
        output: Any | None = None,
        token_count: int | None = None,
        cost_usd: float | None = None,
        error: BaseException | str | None = None,
    ) -> None:
        """Finish the span for ``run_id`` and hand it to the buffer.

        An unknown ``run_id`` (never started, or already ended) is ignored, so a
        stray end callback can never crash the host agent.
        """
        with self._lock:
            span = self._open.pop(run_id, None)
        if span is None:
            return

        redactor = self._tracer.redactor
        span.end_time_ns = time.time_ns()
        if output is not None:
            span.output = redactor.redact(output)
        if token_count is not None:
            span.token_count = token_count
        if cost_usd is not None:
            span.cost_usd = cost_usd
        if error is not None:
            span.status = SpanStatus.ERROR
            span.error_type = type(error).__name__ if isinstance(error, BaseException) else "Error"
            span.error_message = redactor.redact_text(str(error))
        elif span.status is SpanStatus.UNSET:
            span.status = SpanStatus.OK

        # Re-validate to derive latency_ms from the timestamps, then buffer.
        self._tracer.buffer.record(Span.model_validate(span.model_dump()))

    def open_count(self) -> int:
        """Number of spans started but not yet ended (useful for diagnostics)."""
        with self._lock:
            return len(self._open)
