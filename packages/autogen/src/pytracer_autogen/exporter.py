"""OpenTelemetry span exporter that records pyTracer spans.

AutoGen (0.4+) instruments agents, tool calls, and model calls with OpenTelemetry
using the GenAI semantic conventions. Rather than hooking framework internals,
this adapter plugs a pyTracer exporter into an OpenTelemetry ``TracerProvider`` that
AutoGen's runtime uses, and maps each finished OTel span to a pyTracer span. Because
it keys off the standard GenAI attributes, it also works for any other
GenAI-instrumented library, not just AutoGen.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode
from pytracer_sdk import Span, SpanKind, SpanStatus, Tracer, get_tracer, new_span_id, new_trace_id
from pytracer_sdk.seam import active_seam
from pytracer_sdk.tracer import SpanContext, _current

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


_K = TypeVar("_K")

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


# AutoGen's tool span carries the call id but not the result; the result travels
# in a later runtime message (a ToolCallExecutionEvent) with the same call id.
_TOOL_CALL_ID = "gen_ai.tool.call.id"
_RESULT_EVENT = "ToolCallExecutionEvent"
_MESSAGE_ATTR = "message"
# How long a tool span waits for its result message before it is recorded without
# one (e.g. a tool run outside AgentChat, where no such message is published).
_RESULT_WAIT_SECONDS = 2.0


def _tool_results(message: Any) -> list[tuple[str, Any, bool]]:
    """``(call_id, content, is_error)`` for every tool result in a runtime message."""
    if not isinstance(message, str) or _RESULT_EVENT not in message:
        return []
    try:
        data = json.loads(message)
    except ValueError:
        return []
    found: list[tuple[str, Any, bool]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == _RESULT_EVENT and isinstance(node.get("content"), list):
                for result in node["content"]:
                    if isinstance(result, dict) and result.get("call_id") is not None:
                        found.append(
                            (str(result["call_id"]), result.get("content"), bool(result.get("is_error")))
                        )
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data)
    return found


class _ToolResultJoiner:
    """Records each AutoGen tool span once its result message has been seen.

    A tool span ends before the message carrying its result is delivered, so the
    span is held (per run and call id) until that message is exported, then
    recorded with the result as its output. A span whose result never arrives is
    recorded without it after ``_RESULT_WAIT_SECONDS``, or on flush.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[tuple[str, str], tuple[Span, Tracer, threading.Timer]] = {}
        # Results seen before their tool span ended (bounded; duplicates are common,
        # since AutoGen forwards the same result to several recipients).
        self._early: OrderedDict[tuple[str, str], tuple[Any, bool]] = OrderedDict()

    def hold(self, key: tuple[str, str], span: Span, tracer: Tracer) -> None:
        with self._lock:
            early = self._early.pop(key, None)
            if early is None:
                timer = threading.Timer(_RESULT_WAIT_SECONDS, self._expire, args=(key,))
                timer.daemon = True
                self._pending[key] = (span, tracer, timer)
                timer.start()
                return
        _record(tracer, _with_result(span, tracer, *early))

    def resolve(self, key: tuple[str, str], content: Any, is_error: bool) -> None:
        with self._lock:
            entry = self._pending.pop(key, None)
            if entry is None:
                self._early[key] = (content, is_error)
                self._early.move_to_end(key)
                while len(self._early) > _MAX_TRACKED:
                    self._early.popitem(last=False)
                return
        span, tracer, timer = entry
        timer.cancel()
        _record(tracer, _with_result(span, tracer, content, is_error))

    def _expire(self, key: tuple[str, str]) -> None:
        with self._lock:
            entry = self._pending.pop(key, None)
        if entry is not None:
            _record(entry[1], entry[0])

    def flush(self) -> None:
        """Record every held span now, with or without its result."""
        with self._lock:
            entries = list(self._pending.values())
            self._pending.clear()
        for span, tracer, timer in entries:
            timer.cancel()
            _record(tracer, span)


def _with_result(span: Span, tracer: Tracer, content: Any, is_error: bool) -> Span:
    update: dict[str, Any] = {"output": tracer.redactor.redact(content)}
    if is_error:
        update["status"] = SpanStatus.ERROR
        update["error_message"] = tracer.redactor.redact_text(str(content))
    return span.model_copy(update=update)


def _record(tracer: Tracer, span: Span) -> None:
    tracer.buffer.record(Span.model_validate(span.model_dump()))


# AutoGen's model clients (autogen_ext: OpenAI, Azure, Anthropic, Ollama, ...) log
# each call as an event on this logger; AutoGen itself emits no model-call span.
_EVENT_LOGGER = "autogen_core.events"


def _model_output(response: Any) -> Any:
    """The reply in a logged model response: the first choice's message for the
    OpenAI shape, else the response's content, else the whole response."""
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            return choices[0].get("message", choices[0])
        if "content" in response:
            return response["content"]
    return response


def _usage_messages(message: Any) -> list[dict[str, Any]]:
    """Messages in a runtime message that carry model usage (a model reply)."""
    if not isinstance(message, str) or '"models_usage"' not in message:
        return []
    try:
        data = json.loads(message)
    except ValueError:
        return []
    found: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("models_usage"), dict) and node.get("id") is not None:
                found.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data)
    return found


class _ModelCallLogHandler(logging.Handler):
    """Turns the model-call events AutoGen's clients log into pyTracer LLM spans."""

    def __init__(self, exporter: PyTracerSpanExporter) -> None:
        super().__init__(logging.INFO)
        self._exporter = exporter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._exporter.record_model_event(record.msg)
        except Exception:  # noqa: BLE001 - capture must never break the agent
            _log.debug("pytracer could not record an AutoGen model event", exc_info=True)


_log = logging.getLogger("pytracer_autogen")


def _attach_model_call_handler(exporter: PyTracerSpanExporter) -> None:
    """Route AutoGen's model-call events to ``exporter`` (the latest provider wins).

    The events are logged at INFO. When the event logger would drop them (its
    effective level is above INFO, the default), its level is lowered and its
    propagation turned off, so other handlers see exactly what they saw before.
    """
    logger = logging.getLogger(_EVENT_LOGGER)
    for handler in list(logger.handlers):
        if isinstance(handler, _ModelCallLogHandler):
            logger.removeHandler(handler)
    logger.addHandler(_ModelCallLogHandler(exporter))
    if logger.getEffectiveLevel() > logging.INFO:
        logger.setLevel(logging.INFO)
        logger.propagate = False


class _GenAIContentStore:
    """Model input and output that GenAI instrumentation sends as log records.

    Libraries such as ``opentelemetry-instrumentation-openai-v2`` leave the model
    span empty and emit the prompt (``gen_ai.<role>.message``) and the reply
    (``gen_ai.choice``) as OpenTelemetry log records carrying the span's id. They
    are kept here by span id until that span is exported.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._content: OrderedDict[int, tuple[list[Any], list[Any]]] = OrderedDict()

    def add(self, span_id: int, event_name: str, body: Any) -> None:
        if event_name == "gen_ai.choice":
            item = body.get("message", body) if isinstance(body, dict) else body
            slot = 1
        elif event_name.startswith("gen_ai.") and event_name.endswith(".message"):
            role = event_name[len("gen_ai.") : -len(".message")]
            item = {"role": role, **body} if isinstance(body, dict) else {"role": role, "content": body}
            slot = 0
        else:
            return
        with self._lock:
            entry = self._content.setdefault(span_id, ([], []))
            entry[slot].append(item)
            self._content.move_to_end(span_id)
            while len(self._content) > _MAX_TRACKED:
                self._content.popitem(last=False)

    def take(self, span_id: int) -> tuple[list[Any], list[Any]] | None:
        with self._lock:
            return self._content.pop(span_id, None)


_GENAI_CONTENT = _GenAIContentStore()


def _genai_log_processor() -> Any:
    """A LogRecordProcessor feeding ``_GENAI_CONTENT`` (built lazily: the logs SDK
    is optional and its module is still underscored in OpenTelemetry)."""
    from opentelemetry.sdk._logs import LogRecordProcessor

    class _GenAIContentProcessor(LogRecordProcessor):
        def on_emit(self, log_data: Any) -> None:
            record = getattr(log_data, "log_record", log_data)
            span_id = getattr(record, "span_id", None)
            attributes = dict(getattr(record, "attributes", None) or {})
            name = getattr(record, "event_name", None) or attributes.get("event.name")
            if span_id and isinstance(name, str):
                _GENAI_CONTENT.add(span_id, name, getattr(record, "body", None))

        emit = on_emit  # the SDK's method name before on_emit

        def shutdown(self) -> None:
            pass

        def force_flush(self, timeout_millis: int = 30_000) -> bool:
            return True

    return _GenAIContentProcessor()


_genai_logs_attached = False


def _attach_genai_log_records() -> None:
    """Receive GenAI content log records: add a processor to the global
    LoggerProvider, or register one if none is set. Done once per process."""
    global _genai_logs_attached
    if _genai_logs_attached:
        return
    try:
        from opentelemetry import _logs
        from opentelemetry.sdk._logs import LoggerProvider
    except ImportError:  # the logs SDK is not available
        return
    provider = _logs.get_logger_provider()
    if not isinstance(provider, LoggerProvider):
        provider = LoggerProvider()
        _logs.set_logger_provider(provider)
    provider.add_log_record_processor(_genai_log_processor())
    _genai_logs_attached = True


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


class PyTracerSpanExporter(SpanExporter):
    """Maps finished OpenTelemetry spans to pyTracer spans."""

    def __init__(self, tracer: Tracer | None = None, *, linker: _RunLinker | None = None) -> None:
        # Resolve the tracer lazily at export time, not here, so an exporter built
        # before pytracer.configure(...) still records into the configured tracer.
        self._explicit_tracer = tracer
        # With a linker (see pytracer_tracer_provider), an AutoGen run's linked
        # traces are recorded as one trace; without one, OTel's ids are kept.
        self._linker = linker
        # With a linker, tool spans are joined to the result AutoGen sends in a
        # later message, so a tool's output is recorded.
        self._results = _ToolResultJoiner() if linker is not None else None
        # Streamed model calls: start time and request, keyed by the OTel span the
        # call runs under and the agent, until the stream's end event arrives.
        self._stream_starts: dict[tuple[int, str | None], tuple[int, Any]] = {}
        # With replay seams installed (install(replayable=True)), model and tool
        # calls are recorded by the seams; AutoGen's own spans and model events for
        # the same calls are skipped instead of recorded twice.
        self.seams = False

    def _recorded_by_seam(self, kind: SpanKind) -> bool:
        if not self.seams or kind not in (SpanKind.LLM, SpanKind.TOOL):
            return False
        seam = active_seam()
        return seam is not None and seam.kind is kind

    def seam_parent(self) -> SpanContext | None:
        """The pyTracer span a seam call made in the current context nests under:
        the agent turn (or other recorded span) AutoGen is running it in."""
        from opentelemetry import trace as otel_trace

        context = otel_trace.get_current_span().get_span_context()
        if self._linker is None or not context.is_valid:
            return None
        placement = self._linker.place_child(context.trace_id, context.span_id, follow_links=False)
        if placement is None or placement.parent_span_id is None:
            return None
        return SpanContext(placement.trace_id, placement.parent_span_id)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        tracer = self._explicit_tracer or get_tracer()
        redactor = tracer.redactor
        for span in spans:
            context = span.get_span_context()
            if context is None:
                continue
            raw_attrs = dict(span.attributes or {})
            # A runtime message can carry tool results; hand them to the held tool spans.
            if self._results is not None and self._linker is not None:
                for call_id, content, is_error in _tool_results(raw_attrs.get(_MESSAGE_ATTR)):
                    run = self._linker.run_trace_id(span)
                    if run is not None:
                        self._results.resolve((run, call_id), content, is_error)
            # A model reply in a runtime message stands in for the model call when
            # the client logs no call events (e.g. a custom client).
            if self._linker is not None and not self.seams:
                for reply in _usage_messages(raw_attrs.get(_MESSAGE_ATTR)):
                    self._record_derived_call(span, reply, tracer)
            # Skip the runtime's internal messaging spans (no GenAI attributes), so
            # a run is the agent/model/tool calls, not dozens of unknown id-named spans.
            if not _has_genai(raw_attrs):
                continue
            kind, agent_name, tool_name = _classify(raw_attrs)
            if self._recorded_by_seam(kind):
                continue  # the seam recording this call records it (and makes it replayable)
            safe_attrs = redactor.redact(raw_attrs)
            event_input, event_output = _io_from_events(span)
            span_input = _first(safe_attrs, _INPUT_ATTRS)
            span_output = _first(safe_attrs, _OUTPUT_ATTRS)
            if span_input is None and event_input is not None:
                span_input = redactor.redact(event_input)
            if span_output is None and event_output is not None:
                span_output = redactor.redact(event_output)
            logged = _GENAI_CONTENT.take(context.span_id)
            if logged is not None:
                logged_input, logged_output = logged
                if span_input is None and logged_input:
                    span_input = redactor.redact(logged_input)
                if span_output is None and logged_output:
                    span_output = redactor.redact(
                        logged_output[0] if len(logged_output) == 1 else logged_output
                    )
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
            tool_call_id = raw_attrs.get(_TOOL_CALL_ID)
            if (
                self._results is not None
                and kind is SpanKind.TOOL
                and span_output is None
                and tool_call_id is not None
            ):
                # The result arrives in a later message; record the span then.
                self._results.hold((trace_id, str(tool_call_id)), pytracer_span, tracer)
                continue
            _record(tracer, pytracer_span)
        return SpanExportResult.SUCCESS

    def record_model_event(self, event: Any) -> None:
        """Record a model call from an AutoGen ``LLMCallEvent`` / stream events."""
        from opentelemetry import trace as otel_trace

        kwargs = getattr(event, "kwargs", None)
        if not isinstance(kwargs, dict):
            return
        kind = kwargs.get("type")
        if kind not in ("LLMCall", "LLMStreamStart", "LLMStreamEnd"):
            return
        if self._recorded_by_seam(SpanKind.LLM):
            return  # the model call's seam records it
        now = time.time_ns()
        agent_id = kwargs.get("agent_id")
        context = otel_trace.get_current_span().get_span_context()
        key = (context.span_id if context.is_valid else 0, agent_id)
        if kind == "LLMStreamStart":
            self._stream_starts[key] = (now, kwargs.get("messages"))
            return
        start, messages = now, kwargs.get("messages")
        if kind == "LLMStreamEnd":
            start, messages = self._stream_starts.pop(key, (now, None))
        response = kwargs.get("response")
        tracer = self._explicit_tracer or get_tracer()
        redactor = tracer.redactor
        placement = (
            self._linker.place_child(context.trace_id, context.span_id, follow_links=False)
            if self._linker is not None and context.is_valid
            else None
        )
        if placement is not None:
            trace_id, parent_span_id = placement.trace_id, placement.parent_span_id
            if self._linker is not None:
                self._linker.note_logged_calls(trace_id)
        elif context.is_valid:
            trace_id, parent_span_id = f"{context.trace_id:032x}", f"{context.span_id:016x}"
        else:
            trace_id, parent_span_id = new_trace_id(), None
        model = response.get("model") if isinstance(response, dict) else None
        tokens = int(kwargs.get("prompt_tokens") or 0) + int(kwargs.get("completion_tokens") or 0)
        attributes: dict[str, Any] = {"framework": "autogen"}
        if kind == "LLMCall":
            # The event is logged when the call returns, so its start is not known.
            attributes["pytracer.start_unknown"] = True
        if model:
            attributes["gen_ai.response.model"] = model
        if agent_id:
            attributes["gen_ai.agent.id"] = agent_id
        _record(
            tracer,
            Span(
                trace_id=trace_id,
                span_id=new_span_id(),
                parent_span_id=parent_span_id,
                name=f"chat {model}" if model else "chat",
                kind=SpanKind.LLM,
                input=redactor.redact(messages) if messages is not None else None,
                output=redactor.redact(_model_output(response)),
                start_time_ns=start,
                end_time_ns=now,
                token_count=tokens or None,
                status=SpanStatus.OK,
                attributes=redactor.redact(attributes),
                resource=tracer.resource,
            ),
        )

    def _record_derived_call(self, span: ReadableSpan, reply: dict[str, Any], tracer: Tracer) -> None:
        """Record a model call from a model reply AutoGen delivered in a message.

        Used only in runs whose client logs no call events. It has the reply and
        its token counts, but not the request or the call's timing, and is marked
        ``pytracer.derived``.
        """
        context = span.get_span_context()
        if context is None or self._linker is None:
            return
        run = self._linker.run_trace_id(span)
        if run is None or not self._linker.claim_derived(run, str(reply["id"])):
            return
        placement = self._linker.place_child(context.trace_id, context.span_id, follow_links=True)
        if placement is None:
            return
        usage = reply["models_usage"]
        tokens = int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
        at = int(span.start_time or time.time_ns())
        source = reply.get("source")
        _record(
            tracer,
            Span(
                trace_id=placement.trace_id,
                span_id=new_span_id(),
                parent_span_id=placement.parent_span_id,
                name="chat",
                kind=SpanKind.LLM,
                agent_name=str(source) if source is not None else None,
                output=tracer.redactor.redact(reply.get("content")),
                start_time_ns=at,
                end_time_ns=at,
                token_count=tokens or None,
                status=SpanStatus.OK,
                attributes={"framework": "autogen", "pytracer.derived": True},
                resource=tracer.resource,
            ),
        )

    def shutdown(self) -> None:
        self.force_flush()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        if self._results is not None:
            self._results.flush()
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
    return _build_provider(tracer)[0]


def _build_provider(tracer: Tracer | None) -> tuple[TracerProvider, PyTracerSpanExporter]:
    linker = _RunLinker()
    exporter = PyTracerSpanExporter(tracer, linker=linker)
    provider = TracerProvider()
    provider.add_span_processor(_LinkingSpanProcessor(exporter, linker))
    _attach_model_call_handler(exporter)
    return provider, exporter


_installed_seams: Any = None


def install(tracer: Tracer | None = None, *, replayable: bool = False) -> TracerProvider:
    """Register a pyTracer TracerProvider as the global OpenTelemetry provider.

    AutoGen emits its GenAI model and tool spans through the *global* provider, so
    register globally (before building the runtime) to capture them. It also
    receives the model content that GenAI instrumentation sends as log records
    (adding a processor to the global LoggerProvider, or registering one), so an
    instrumented model call keeps its prompt and reply. Returns the provider. Call
    once per process, before any span is emitted.

    With ``replayable=True``, AutoGen's model calls and tool calls are also routed
    through pyTracer (see :mod:`pytracer_autogen.replay`), so a captured run can be
    replayed without adding ``@op`` to anything.
    """
    from opentelemetry import trace as otel_trace

    global _installed_seams
    provider, exporter = _build_provider(tracer)
    otel_trace.set_tracer_provider(provider)
    # Model content from GenAI instrumentation arrives as log records (P7).
    _attach_genai_log_records()
    uninstall_seams()
    if replayable:
        from .replay import Seams

        exporter.seams = True
        _installed_seams = Seams(exporter.seam_parent)
        _installed_seams.install()
    return provider


def install_seams(exporter: PyTracerSpanExporter) -> None:
    """Install the replay seams for ``exporter`` (``install(replayable=True)``
    does this; exposed for a provider registered by other means)."""
    global _installed_seams
    from .replay import Seams

    uninstall_seams()
    exporter.seams = True
    _installed_seams = Seams(exporter.seam_parent)
    _installed_seams.install()


def uninstall_seams() -> None:
    """Remove the replay seams installed by ``install(replayable=True)``."""
    global _installed_seams
    if _installed_seams is not None:
        _installed_seams.uninstall()
        _installed_seams = None
