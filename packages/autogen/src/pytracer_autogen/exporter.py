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


def _classify(attrs: dict[str, Any]) -> tuple[SpanKind, str | None, str | None]:
    """Infer (kind, agent_name, tool_name) from GenAI attributes."""
    operation = attrs.get(_OP)
    tool = attrs.get(_TOOL_NAME)
    agent = attrs.get(_AGENT_NAME)
    if operation == "execute_tool" or tool is not None:
        return SpanKind.TOOL, None, str(tool) if tool is not None else None
    if operation in _LLM_OPS:
        return SpanKind.LLM, None, None
    if operation in _AGENT_OPS or agent is not None:
        return SpanKind.AGENT, str(agent) if agent is not None else None, None
    return SpanKind.UNKNOWN, None, None


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
            safe_attrs = redactor.redact(raw_attrs)
            kind, agent_name, tool_name = _classify(raw_attrs)
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
                input=safe_attrs.get("gen_ai.prompt") or safe_attrs.get("gen_ai.input"),
                output=safe_attrs.get("gen_ai.completion") or safe_attrs.get("gen_ai.output"),
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
