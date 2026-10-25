"""SpanRecorder - build spans from external framework events.

The ``@op`` decorator captures spans by nesting on the Python call stack, which
works when pyTracer controls the code flow. Framework integrations do not: agent
frameworks (LangGraph, CrewAI, AutoGen, the OpenAI Agents SDK) report work as
start/end callbacks that carry their own run ids and parent run ids, often from
different threads. This recorder bridges that model. An adapter calls
:meth:`start` when the framework begins an LLM call, tool call, or agent step,
and :meth:`end` when it finishes; the recorder maps the framework's run ids to
pyTracer trace/span ids, links parents explicitly, redacts, and buffers the span.

Every framework adapter is a thin translation of that framework's events into
these two calls.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from .ids import new_span_id, new_trace_id
from .schema import Span, SpanKind, SpanStatus
from .tracer import SpanContext, Tracer, _current


class SpanRecorder:
    """Records spans from framework events keyed by the framework's run ids."""

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer
        self._lock = threading.Lock()
        self._open: dict[str, Span] = {}
        # Framework runs recorded by a seam span instead (see adopt): run id -> the
        # seam span, so the run's children still nest under it.
        self._adopted: dict[str, SpanContext] = {}

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
        open, this span joins that trace as a child. Otherwise, if a pyTracer span
        is active in this context (an ``@op`` or ``pytracer.start_span`` around the
        call, e.g. one framework handing off to another), the span nests under it,
        so one request is one trace across frameworks. Failing both, it starts a new
        trace as a root. A duplicate ``run_id`` overwrites the earlier open span.
        """
        redactor = self._tracer.redactor
        with self._lock:
            parent = self._parent(parent_run_id)
            if parent is not None:
                trace_id = parent.trace_id
                parent_span_id: str | None = parent.span_id
            else:
                ambient = _current.get()
                trace_id = ambient.trace_id if ambient is not None else new_trace_id()
                parent_span_id = ambient.span_id if ambient is not None else None
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

    def _parent(self, run_id: str | None) -> Span | SpanContext | None:
        if run_id is None:
            return None
        return self._open.get(run_id) or self._adopted.get(run_id)

    def adopt(self, run_id: str, kind: SpanKind | None = None) -> bool:
        """Let ``run_id`` be the framework call a seam is recording right now.

        When a framework call runs through a seam (``pytracer_sdk.seam``), the seam
        span records it, so the adapter must not open a second span for the same
        call. Call this from the adapter's start hook: it returns True (and the
        adapter skips its own span) when a seam call is active in this context, and
        maps ``run_id`` to the seam span so the run's children nest under it. The
        matching :meth:`end` is then a no-op, since the seam closes its own span.
        With ``kind``, only a seam call of that kind is adopted (a tool callback
        never stands in for a model call's seam, or the reverse).
        """
        from .seam import active_seam  # late: seam imports the tracer module

        seam = active_seam()
        if seam is None or not seam.span.span_id or (kind is not None and seam.kind is not kind):
            return False
        with self._lock:
            self._adopted[run_id] = seam.span
        return True

    def end(
        self,
        run_id: str,
        *,
        input: Any | None = None,
        output: Any | None = None,
        token_count: int | None = None,
        cost_usd: float | None = None,
        error: BaseException | str | None = None,
    ) -> None:
        """Finish the span for ``run_id`` and hand it to the buffer.

        ``input`` may be supplied here (not only at :meth:`start`) for frameworks
        that only populate a call's input by the time it completes. An unknown
        ``run_id`` (never started, or already ended) is ignored, so a stray end
        callback can never crash the host agent.
        """
        with self._lock:
            span = self._open.pop(run_id, None)
            if span is None:
                self._adopted.pop(run_id, None)  # a seam span closes itself
        if span is None:
            return

        redactor = self._tracer.redactor
        span.end_time_ns = time.time_ns()
        if input is not None:
            span.input = redactor.redact(input)
        if output is not None:
            span.output = redactor.redact(output)
        if token_count is not None:
            span.token_count = token_count
        if cost_usd is not None:
            span.cost_usd = cost_usd
        if error is not None:
            span.status = SpanStatus.ERROR
            span.error_type = type(error).__name__ if isinstance(error, BaseException) else "Error"
            if isinstance(error, BaseException):
                span.error_module = type(error).__module__
            span.error_message = redactor.redact_text(str(error))
        elif span.status is SpanStatus.UNSET:
            span.status = SpanStatus.OK

        # latency_ms was derived when end_time_ns was set; buffer the span as is, so
        # an object output (a pydantic model, a dataclass) keeps its type.
        self._tracer.buffer.record(span)

    def context_of(self, run_id: str | None) -> SpanContext | None:
        """The span recording ``run_id`` (open, or adopted by a seam), so a seam
        call made inside that framework run can nest under it."""
        with self._lock:
            parent = self._parent(run_id)
        if parent is None:
            return None
        return SpanContext(parent.trace_id, parent.span_id)

    def open_count(self) -> int:
        """Number of spans started but not yet ended (useful for diagnostics)."""
        with self._lock:
            return len(self._open)

    def is_open(self, run_id: str) -> bool:
        """True when ``run_id`` names a started span that has not ended, so a
        caller can tell whether a reported parent id can actually be nested under."""
        with self._lock:
            return run_id in self._open or run_id in self._adopted
