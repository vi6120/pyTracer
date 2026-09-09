"""OpenAI Agents SDK tracing processor that records pyTracer spans.

The Agents SDK emits its own traces and spans (agent runs, model generations,
function/tool calls, handoffs) and lets you register a ``TracingProcessor`` to
observe them. This processor translates those into
:class:`~pytracer_sdk.SpanRecorder` start/end calls, so an agent is instrumented
just by registering the processor, with no ``@op`` decorators in the agent code.

Span-data input/output/usage are only populated by the time a span ends, so the
span is started at ``on_span_start`` (to anchor parent linkage) and its payload
is filled at ``on_span_end``.
"""

from __future__ import annotations

from typing import Any

from agents.tracing import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    ResponseSpanData,
    Span,
    TracingProcessor,
)
from agents.tracing import Trace as AgentsTrace
from pytracer_sdk import SpanKind, SpanRecorder, Tracer, get_tracer

_MAX_DEPTH = 6


def _jsonify(value: Any, depth: int = 0) -> Any:
    """Coerce SDK objects (pydantic responses, etc.) to JSON-friendly primitives."""
    if depth > _MAX_DEPTH:
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonify(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v, depth + 1) for v in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _jsonify(dump(mode="json"), depth + 1)
        except Exception:  # noqa: BLE001 - fall back to a string form
            return str(value)
    return str(value)


def _tokens(usage: Any) -> int | None:
    if not isinstance(usage, dict):
        return None
    if usage.get("total_tokens") is not None:
        return int(usage["total_tokens"])
    inp = usage.get("input_tokens") or usage.get("prompt_tokens")
    out = usage.get("output_tokens") or usage.get("completion_tokens")
    if inp is not None or out is not None:
        return int(inp or 0) + int(out or 0)
    return None


def _classify(data: Any) -> tuple[str, SpanKind, str | None, str | None]:
    """Return (name, kind, agent_name, tool_name) for a span's data."""
    if isinstance(data, AgentSpanData):
        return data.name, SpanKind.AGENT, data.name, None
    if isinstance(data, FunctionSpanData):
        return data.name, SpanKind.TOOL, None, data.name
    if isinstance(data, GenerationSpanData):
        return (getattr(data, "model", None) or "generation"), SpanKind.LLM, None, None
    if isinstance(data, ResponseSpanData):
        return "response", SpanKind.LLM, None, None
    if isinstance(data, HandoffSpanData):
        return f"{data.from_agent} -> {data.to_agent}", SpanKind.HANDOFF, None, None
    if isinstance(data, (GuardrailSpanData, CustomSpanData)):
        return data.name, SpanKind.UNKNOWN, None, None
    return str(getattr(data, "type", "span")), SpanKind.UNKNOWN, None, None


def _output(data: Any) -> Any:
    if isinstance(data, ResponseSpanData):
        return _jsonify(getattr(data, "response", None))
    return _jsonify(getattr(data, "output", None))


def _usage_tokens(data: Any) -> int | None:
    usage = getattr(data, "usage", None)
    if usage is None:
        response = getattr(data, "response", None)
        response_usage = getattr(response, "usage", None) if response is not None else None
        dump = getattr(response_usage, "model_dump", None)
        if callable(dump):
            usage = dump(mode="json")
    return _tokens(usage)


def _error(span: Any) -> str | None:
    err = getattr(span, "error", None)
    if not err:
        return None
    if isinstance(err, dict):
        return str(err.get("message") or err)
    return str(getattr(err, "message", None) or err)


class PyTracerTracingProcessor(TracingProcessor):
    """Records pyTracer spans from OpenAI Agents SDK trace/span callbacks."""

    def __init__(self, tracer: Tracer | None = None) -> None:
        self._recorder = SpanRecorder(tracer or get_tracer())

    def on_trace_start(self, trace: AgentsTrace) -> None:
        self._recorder.start(
            trace.trace_id,
            name=getattr(trace, "name", None) or "agent_run",
            kind=SpanKind.AGENT,
            attributes={"framework": "openai-agents"},
        )

    def on_trace_end(self, trace: AgentsTrace) -> None:
        self._recorder.end(trace.trace_id)

    def on_span_start(self, span: Span[Any]) -> None:
        name, kind, agent_name, tool_name = _classify(span.span_data)
        self._recorder.start(
            span.span_id,
            name=str(name),
            kind=kind,
            agent_name=agent_name,
            tool_name=tool_name,
            parent_run_id=span.parent_id or span.trace_id,
            attributes={"framework": "openai-agents"},
        )

    def on_span_end(self, span: Span[Any]) -> None:
        data = span.span_data
        self._recorder.end(
            span.span_id,
            input=_jsonify(getattr(data, "input", None)),
            output=_output(data),
            token_count=_usage_tokens(data),
            error=_error(span),
        )

    def force_flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass
