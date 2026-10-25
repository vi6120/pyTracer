"""OpenTelemetry span exporter that records pyTracer spans.

AutoGen (0.4+) instruments agents, tool calls, and model calls with OpenTelemetry
using the GenAI semantic conventions. Rather than hooking framework internals,
this adapter plugs a pyTracer exporter into an OpenTelemetry ``TracerProvider`` that
AutoGen's runtime uses, and maps each finished OTel span to a pyTracer span. Because
it keys off the standard GenAI attributes, it also works for any other
GenAI-instrumented library, not just AutoGen.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode
from pytracer_sdk import Span, SpanKind, SpanStatus, Tracer, get_tracer
from pytracer_sdk.tracer import _current

# GenAI semantic-convention attribute keys.
_OP = "gen_ai.operation.name"
_TOOL_NAME = "gen_ai.tool.name"
_AGENT_NAME = "gen_ai.agent.name"
_ERROR_TYPE = "error.type"
_LLM_OPS = frozenset({"chat", "text_completion", "generate_content", "embeddings"})
_AGENT_OPS = frozenset({"create_agent", "invoke_agent"})
# A model call may arrive with only the request/response model set (no operation
# name), so these also mark a span as an LLM call.
_MODEL_ATTRS = ("gen_ai.request.model", "gen_ai.response.model")

# Where prompt/response content turns up across GenAI convention versions, in
# attributes and (newer convention) span events.
_INPUT_ATTRS = ("gen_ai.prompt", "gen_ai.input", "gen_ai.input.messages", "gen_ai.request.messages")
_OUTPUT_ATTRS = ("gen_ai.completion", "gen_ai.output", "gen_ai.output.messages", "gen_ai.response")
_OUTPUT_EVENT_SUFFIXES = (".assistant.message", ".choice")
_INPUT_EVENT_SUFFIXES = (".user.message", ".system.message", ".tool.message")


def _has_genai(attrs: dict[str, Any]) -> bool:
    """True if a span carries any GenAI attribute (so it is an agent/model/tool call).

    AutoGen's runtime emits many internal messaging spans with no GenAI attributes
    and generated, id-like names; those are noise, so only GenAI-relevant spans are
    recorded.
    """
    return any(str(k).startswith("gen_ai.") for k in attrs)


def _classify(attrs: dict[str, Any]) -> tuple[SpanKind, str | None, str | None]:
    """Infer (kind, agent_name, tool_name) from GenAI attributes."""
    operation = attrs.get(_OP)
    tool = attrs.get(_TOOL_NAME)
    agent = attrs.get(_AGENT_NAME)
    if operation == "execute_tool" or tool is not None:
        return SpanKind.TOOL, None, str(tool) if tool is not None else None
    if operation in _LLM_OPS or any(attrs.get(k) for k in _MODEL_ATTRS):
        return SpanKind.LLM, None, None
    if operation in _AGENT_OPS or agent is not None:
        return SpanKind.AGENT, str(agent) if agent is not None else None, None
    return SpanKind.UNKNOWN, None, None


def _event_content(event: Any) -> Any:
    """The content payload of a GenAI span event, if any."""
    attrs = dict(getattr(event, "attributes", None) or {})
    return attrs.get("content") or attrs.get("gen_ai.event.content") or (attrs or None)


def _io_from_events(span: ReadableSpan) -> tuple[Any, Any]:
    """Collect model input/output recorded as span events (newer GenAI convention)."""
    inputs: list[Any] = []
    outputs: list[Any] = []
    for event in getattr(span, "events", None) or []:
        name = str(getattr(event, "name", ""))
        content = _event_content(event)
        if content is None:
            continue
        if name.endswith(_OUTPUT_EVENT_SUFFIXES):
            outputs.append(content)
        elif name.endswith(_INPUT_EVENT_SUFFIXES) or name.endswith(".message"):
            inputs.append(content)
    return (inputs or None), (outputs or None)


def _first(attrs: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = attrs.get(key)
        if value is not None:
            return value
    return None


def _tokens(attrs: dict[str, Any]) -> int | None:
    total = attrs.get("gen_ai.usage.total_tokens")
    if total is not None:
        return int(total)
    inp = attrs.get("gen_ai.usage.input_tokens")
    out = attrs.get("gen_ai.usage.output_tokens")
    if inp is not None or out is not None:
        return int(inp or 0) + int(out or 0)
    return None


@dataclass(frozen=True)
class _Run:
    """The pyTracer trace a group of linked OpenTelemetry traces is recorded under."""

    trace_id: str
    parent_span_id: str | None  # the enclosing pyTracer span, if the run had one


@dataclass(frozen=True)
class _Placement:
    trace_id: str
    parent_span_id: str | None


# Bound on remembered spans and traces, so a long-lived process does not grow
# without limit; far more than any one run holds.
_MAX_TRACKED = 20_000


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
        # OTel span id -> (the span, its parent's span id within the same trace).
        # The span is kept so whether it is recorded (has GenAI attributes) is
        # read when a descendant is placed, since attributes are often set after
        # a span starts.
        self._spans: OrderedDict[int, tuple[OTelSpan, int | None]] = OrderedDict()

    def on_start(self, span: OTelSpan) -> None:
        context = span.get_span_context()
        if context is None:
            return
        trace_id, span_id = context.trace_id, context.span_id
        parent = span.parent
        parent_id = parent.span_id if parent is not None and parent.trace_id == trace_id else None
        with self._lock:
            if trace_id not in self._runs:
                self._remember(self._runs, trace_id, self._run_for(span))
            self._remember(self._spans, span_id, (span, parent_id))

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
            # Walk up to the nearest ancestor that is itself recorded.
            ancestor = entry[1]
            while ancestor is not None:
                found = self._spans.get(ancestor)
                if found is None:
                    ancestor = None  # forgotten or never seen: treat as the top
                elif _has_genai(dict(found[0].attributes or {})):
                    break
                else:
                    ancestor = found[1]
        if ancestor is not None:
            return _Placement(run.trace_id, f"{ancestor:016x}")
        return _Placement(run.trace_id, run.parent_span_id)

    @staticmethod
    def _remember(store: OrderedDict[int, Any], key: int, value: Any) -> None:
        store[key] = value
        store.move_to_end(key)
        while len(store) > _MAX_TRACKED:
            store.popitem(last=False)


class _LinkingSpanProcessor(SimpleSpanProcessor):
    """Exports each span when it ends, and feeds span starts to the run linker."""

    def __init__(self, exporter: PyTracerSpanExporter, linker: _RunLinker) -> None:
        super().__init__(exporter)
        self._linker = linker

    def on_start(self, span: OTelSpan, parent_context: Context | None = None) -> None:
        self._linker.on_start(span)
        super().on_start(span, parent_context)


class PyTracerSpanExporter(SpanExporter):
    """Maps finished OpenTelemetry spans to pyTracer spans."""

    def __init__(self, tracer: Tracer | None = None, *, linker: _RunLinker | None = None) -> None:
        # Resolve the tracer lazily at export time, not here, so an exporter built
        # before pytracer.configure(...) still records into the configured tracer.
        self._explicit_tracer = tracer
        # With a linker (see pytracer_tracer_provider), an AutoGen run's linked
        # traces are recorded as one trace; without one, OTel's ids are kept.
        self._linker = linker

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        tracer = self._explicit_tracer or get_tracer()
        redactor = tracer.redactor
        for span in spans:
            context = span.get_span_context()
            if context is None:
                continue
            raw_attrs = dict(span.attributes or {})
            # Skip the runtime's internal messaging spans (no GenAI attributes), so
            # a run is the agent/model/tool calls, not dozens of unknown id-named spans.
            if not _has_genai(raw_attrs):
                continue
            safe_attrs = redactor.redact(raw_attrs)
            kind, agent_name, tool_name = _classify(raw_attrs)
            event_input, event_output = _io_from_events(span)
            span_input = _first(safe_attrs, _INPUT_ATTRS)
            span_output = _first(safe_attrs, _OUTPUT_ATTRS)
            if span_input is None and event_input is not None:
                span_input = redactor.redact(event_input)
            if span_output is None and event_output is not None:
                span_output = redactor.redact(event_output)
            error_type = raw_attrs.get(_ERROR_TYPE)
            status = (
                SpanStatus.ERROR
                if span.status.status_code is StatusCode.ERROR
                else SpanStatus.OK
            )
            placement = self._linker.place(span) if self._linker is not None else None
            if placement is not None:
                trace_id, parent_span_id = placement.trace_id, placement.parent_span_id
            else:
                trace_id = f"{context.trace_id:032x}"
                parent_span_id = f"{span.parent.span_id:016x}" if span.parent else None
            pytracer_span = Span(
                trace_id=trace_id,
                span_id=f"{context.span_id:016x}",
                parent_span_id=parent_span_id,
                name=span.name,
                kind=kind,
                agent_name=agent_name,
                tool_name=tool_name,
                input=span_input,
                output=span_output,
                start_time_ns=int(span.start_time or 0),
                end_time_ns=int(span.end_time) if span.end_time is not None else None,
                token_count=_tokens(raw_attrs),
                status=status,
                error_type=str(error_type) if error_type is not None else None,
                attributes={"framework": "autogen", **safe_attrs},
                resource=tracer.resource,
            )
            tracer.buffer.record(Span.model_validate(pytracer_span.model_dump()))
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


def pytracer_tracer_provider(tracer: Tracer | None = None) -> TracerProvider:
    """Build an OpenTelemetry TracerProvider that records into pyTracer.

    Register it globally with :func:`install` (recommended) so AutoGen's GenAI
    model and tool spans are captured. Passing it only to the runtime
    (``SingleThreadedAgentRuntime(tracer_provider=...)``) sees just the runtime's
    internal messaging spans, which this exporter drops as noise.

    The provider records each AutoGen run as one trace (following the links
    AutoGen puts between the per-message traces), nested under the pyTracer span
    that is open when the run starts, if any.
    """
    linker = _RunLinker()
    provider = TracerProvider()
    provider.add_span_processor(
        _LinkingSpanProcessor(PyTracerSpanExporter(tracer, linker=linker), linker)
    )
    return provider


def install(tracer: Tracer | None = None) -> TracerProvider:
    """Register a pyTracer TracerProvider as the global OpenTelemetry provider.

    AutoGen emits its GenAI model and tool spans through the *global* provider, so
    register globally (before building the runtime) to capture them. Returns the
    provider. Call once per process, before any span is emitted.
    """
    from opentelemetry import trace as otel_trace

    provider = pytracer_tracer_provider(tracer)
    otel_trace.set_tracer_provider(provider)
    return provider
