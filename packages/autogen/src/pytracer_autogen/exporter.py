"""OpenTelemetry span exporter that records pyTracer spans.

AutoGen instruments agents, tool calls, and model calls with OpenTelemetry using
the GenAI semantic conventions (fully from 0.7.2; older releases are captured
through seams, see :mod:`pytracer_autogen.versions`). Rather than hooking framework internals,
this adapter plugs a pyTracer exporter into an OpenTelemetry ``TracerProvider`` that
AutoGen's runtime uses, and maps each finished OTel span to a pyTracer span. Because
it keys off the standard GenAI attributes, it also works for any other
GenAI-instrumented library, not just AutoGen.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from weakref import WeakKeyDictionary
from collections.abc import Sequence
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode
from pytracer_sdk import Span, SpanKind, SpanStatus, Tracer, get_tracer, new_span_id, new_trace_id
from pytracer_sdk.seam import active_seam
from pytracer_sdk.tracer import SpanContext

from .genai import (
    _ERROR_TYPE,
    _INPUT_ATTRS,
    _MAX_TRACKED,
    _OUTPUT_ATTRS,
    _classify,
    _first,
    _has_genai,
    _io_from_events,
    _record,
    _tokens,
)
from .genai_logs import _GENAI_CONTENT, _attach_genai_log_records
from .linking import _LinkingSpanProcessor, _RunLinker
from .model_calls import _attach_model_call_handler, _model_output, _usage_messages
from .tool_results import _MESSAGE_ATTR, _TOOL_CALL_ID, _tool_results, _ToolResultJoiner
from .versions import captures_by_seams, seams_agent_turns


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
        self._stream_starts: OrderedDict[tuple[int, str | None], tuple[int, Any]] = OrderedDict()
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

        seam = active_seam()
        if seam is not None and seam.kind is SpanKind.AGENT and seam.span is not None:
            return seam.span  # the agent turn recorded by its seam (older AutoGen)
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
        if self._results is not None:
            self._results.sweep()  # tool spans whose result never came
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
            while len(self._stream_starts) > _MAX_TRACKED:
                self._stream_starts.popitem(last=False)  # a stream that never ended
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
    _EXPORTERS[provider] = exporter
    _attach_model_call_handler(exporter)
    return provider, exporter


# Each pyTracer provider's exporter, so the exporter can be found from the provider.
_EXPORTERS: WeakKeyDictionary[TracerProvider, PyTracerSpanExporter] = WeakKeyDictionary()


def exporter_for(provider: Any) -> PyTracerSpanExporter | None:
    """The exporter behind a provider built by :func:`pytracer_tracer_provider`
    (or :func:`install`), e.g. to pass to :func:`install_seams`; else ``None``."""
    try:
        return _EXPORTERS.get(provider)
    except TypeError:  # not weak-referenceable, so not one of ours
        return None


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
    replayed without adding ``@op`` to anything. On AutoGen releases before 0.7.2,
    which do not report a tool's result (or, before 0.6.2, any agent or tool span),
    the calls are always routed this way, since that is how they are captured (see
    :mod:`pytracer_autogen.versions`).
    """
    from opentelemetry import trace as otel_trace

    provider, exporter = _build_provider(tracer)
    otel_trace.set_tracer_provider(provider)
    # Model content from GenAI instrumentation arrives as log records (P7).
    _attach_genai_log_records()
    if replayable or captures_by_seams():
        install_seams(exporter)
    else:
        uninstall_seams()
    return provider


def install_seams(exporter: PyTracerSpanExporter) -> None:
    """Install the replay seams for ``exporter`` (``install(replayable=True)``
    does this; exposed for a provider registered by other means)."""
    global _installed_seams
    from .replay import Seams

    uninstall_seams()
    exporter.seams = True
    seams = Seams(exporter.seam_parent, agent_turns=seams_agent_turns())
    seams.install()
    _installed_seams = (seams, exporter)


def uninstall_seams() -> None:
    """Remove the replay seams installed by ``install(replayable=True)``, and let
    the exporter record AutoGen's own spans and model events again."""
    global _installed_seams
    if _installed_seams is not None:
        seams, exporter = _installed_seams
        seams.uninstall()
        exporter.seams = False
        _installed_seams = None
