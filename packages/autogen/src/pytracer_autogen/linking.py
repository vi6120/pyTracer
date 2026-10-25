"""Grouping an AutoGen run's linked OpenTelemetry traces into one pyTracer trace."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from pytracer_sdk.tracer import _current

from .genai import _MAX_TRACKED, _has_genai

if TYPE_CHECKING:
    from .exporter import PyTracerSpanExporter

@dataclass(frozen=True)
class _Run:
    """The pyTracer trace a group of linked OpenTelemetry traces is recorded under."""

    trace_id: str
    parent_span_id: str | None  # the enclosing pyTracer span, if the run had one


@dataclass(frozen=True)
class _Placement:
    trace_id: str
    parent_span_id: str | None


_K = TypeVar("_K")



class _RunLinker:
    """Groups the OpenTelemetry traces of one AutoGen run into one pyTracer trace.

    AutoGen starts a new OpenTelemetry trace for every message it delivers, and
    connects it to the trace that sent the message with a span *link*, not a
    parent. So a team run arrives as many traces. Watching spans as they start,
    this follows those links: a trace whose root links to a known trace joins
    that trace's run. A run that starts inside an open pyTracer span (e.g.
    ``pytracer.start_span(...)``) is recorded under it, like the other adapters.

    Only GenAI spans are recorded, so each recorded span is re-parented to its
    nearest recorded ancestor (within its own trace); a span with none sits at
    the top of the run, under the enclosing pyTracer span when there is one.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: OrderedDict[int, _Run] = OrderedDict()  # OTel trace id -> run
        # OTel span id -> (the span, its parent's span id within the same trace,
        # and for a trace's root the span its link points to, i.e. the span that
        # sent the message this trace delivers). The span is kept so whether it is
        # recorded (has GenAI attributes) is read when a descendant is placed, since
        # attributes are often set after a span starts.
        self._spans: OrderedDict[int, tuple[OTelSpan, int | None, int | None]] = OrderedDict()
        # Runs whose model client logged its calls, so no call is derived from
        # messages there as well; and message ids already turned into a call.
        self._logged_runs: OrderedDict[str, None] = OrderedDict()
        self._derived: OrderedDict[tuple[str, str], None] = OrderedDict()

    def on_start(self, span: OTelSpan) -> None:
        context = span.get_span_context()
        if context is None:
            return
        trace_id, span_id = context.trace_id, context.span_id
        parent = span.parent
        parent_id = parent.span_id if parent is not None and parent.trace_id == trace_id else None
        cause_id = None
        if parent is None:
            cause_id = next((link.context.span_id for link in span.links or ()), None)
        with self._lock:
            if trace_id not in self._runs:
                self._remember(self._runs, trace_id, self._run_for(span))
            self._remember(self._spans, span_id, (span, parent_id, cause_id))

    def _run_for(self, span: OTelSpan) -> _Run:
        """The run a newly seen trace belongs to: a linked trace's, else a new one."""
        for link in span.links or ():
            linked = self._runs.get(link.context.trace_id)
            if linked is not None:
                return linked
        ambient = _current.get()
        if ambient is not None:
            return _Run(ambient.trace_id, ambient.span_id)
        context = span.get_span_context()
        return _Run(f"{context.trace_id:032x}", None)

    def run_trace_id(self, span: ReadableSpan) -> str | None:
        """The pyTracer trace id of the run ``span`` belongs to, if known."""
        context = span.get_span_context()
        if context is None:
            return None
        with self._lock:
            run = self._runs.get(context.trace_id)
        return run.trace_id if run is not None else None

    def place(self, span: ReadableSpan) -> _Placement | None:
        """Where to record a finished span, or ``None`` if its start was not seen."""
        context = span.get_span_context()
        if context is None:
            return None
        with self._lock:
            run = self._runs.get(context.trace_id)
            entry = self._spans.get(context.span_id)
            if run is None or entry is None:
                return None
            ancestor = self._nearest_recorded(entry[1], follow_links=False)
        return self._placement(run, ancestor)

    def place_child(self, trace_id: int, span_id: int, *, follow_links: bool) -> _Placement | None:
        """Where to record a new span created under the OTel span ``span_id``.

        The new span nests under that span if it is recorded, else under its
        nearest recorded ancestor. With ``follow_links``, the walk continues from a
        trace's root to the span that sent its message, so a span derived from a
        delivered message lands under the agent turn that produced it.
        """
        with self._lock:
            run = self._runs.get(trace_id)
            if run is None or span_id not in self._spans:
                return None
            ancestor = self._nearest_recorded(span_id, follow_links=follow_links)
        return self._placement(run, ancestor)

    def _nearest_recorded(self, start: int | None, *, follow_links: bool) -> int | None:
        """The first recorded span from ``start`` upward (``start`` included)."""
        current = start
        seen: set[int] = set()
        while current is not None and current not in seen:
            seen.add(current)
            found = self._spans.get(current)
            if found is None:
                return None  # forgotten or never seen: treat as the top
            if _has_genai(dict(found[0].attributes or {})):
                return current
            _, parent_id, cause_id = found
            current = parent_id if parent_id is not None else (cause_id if follow_links else None)
        return None

    @staticmethod
    def _placement(run: _Run, ancestor: int | None) -> _Placement:
        if ancestor is not None:
            return _Placement(run.trace_id, f"{ancestor:016x}")
        return _Placement(run.trace_id, run.parent_span_id)

    def note_logged_calls(self, run_trace_id: str) -> None:
        with self._lock:
            self._remember(self._logged_runs, run_trace_id, None)

    def claim_derived(self, run_trace_id: str, message_id: str) -> bool:
        """True the first time a message is seen in a run whose client logs nothing."""
        with self._lock:
            if run_trace_id in self._logged_runs or (run_trace_id, message_id) in self._derived:
                return False
            self._remember(self._derived, (run_trace_id, message_id), None)
            return True

    @staticmethod
    def _remember(store: OrderedDict[_K, Any], key: _K, value: Any) -> None:
        store[key] = value
        store.move_to_end(key)
        while len(store) > _MAX_TRACKED:
            store.popitem(last=False)


class _LinkingSpanProcessor(SimpleSpanProcessor):
    """Exports each span when it ends, and feeds span starts to the run linker."""

    def __init__(self, exporter: PyTracerSpanExporter, linker: _RunLinker) -> None:
        super().__init__(exporter)
        self._linker = linker
        self._pytracer_exporter = exporter

    def on_start(self, span: OTelSpan, parent_context: Context | None = None) -> None:
        self._linker.on_start(span)
        super().on_start(span, parent_context)

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        # SimpleSpanProcessor does not pass a flush on to its exporter; this one
        # must, so tool spans still waiting for their result are recorded.
        return self._pytracer_exporter.force_flush(timeout_millis)
