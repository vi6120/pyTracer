"""OpenTelemetry span exporter that records pyTracer spans.

AutoGen (0.4+) instruments agents, tool calls, and model calls with OpenTelemetry
using the GenAI semantic conventions. Rather than hooking framework internals,
this adapter plugs a pyTracer exporter into an OpenTelemetry ``TracerProvider`` that
AutoGen's runtime uses, and maps each finished OTel span to a pyTracer span. Because
it keys off the standard GenAI attributes, it also works for any other
GenAI-instrumented library, not just AutoGen.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode
from pytracer_sdk import Span, SpanKind, SpanStatus, Tracer, get_tracer

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


class PyTracerSpanExporter(SpanExporter):
    """Maps finished OpenTelemetry spans to pyTracer spans."""

    def __init__(self, tracer: Tracer | None = None) -> None:
        self._tracer = tracer or get_tracer()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        redactor = self._tracer.redactor
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
            pytracer_span = Span(
                trace_id=f"{context.trace_id:032x}",
                span_id=f"{context.span_id:016x}",
                parent_span_id=f"{span.parent.span_id:016x}" if span.parent else None,
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
                resource=self._tracer.resource,
            )
            self._tracer.buffer.record(Span.model_validate(pytracer_span.model_dump()))
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


def pytracer_tracer_provider(tracer: Tracer | None = None) -> TracerProvider:
    """Build an OpenTelemetry TracerProvider that records into pyTracer.

    Pass the result to AutoGen's runtime (``SingleThreadedAgentRuntime(
    tracer_provider=...)``) or register it globally so AutoGen's spans are
    captured as pyTracer spans.
    """
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(PyTracerSpanExporter(tracer)))
    return provider
