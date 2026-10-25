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
    set_trace_processors,
)
from agents.tracing import Trace as AgentsTrace
from pytracer_sdk import SpanKind, SpanRecorder, Tracer, get_tracer
from pytracer_sdk.seam import active_seam
from pytracer_sdk.tracer import SpanContext

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


# Span data types that carry a real agent/model/tool/handoff. Anything else (the
# SDK's structural "task"/"turn" spans) is folded into its parent so the tree has
# no "unknown" spans and the real work still nests correctly.
_KNOWN_SPAN_DATA = (
    AgentSpanData,
    FunctionSpanData,
    GenerationSpanData,
    ResponseSpanData,
    HandoffSpanData,
    GuardrailSpanData,
    CustomSpanData,
)


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

    def __init__(self, tracer: Tracer | None = None, *, seams: bool = False) -> None:
        self._recorder = SpanRecorder(tracer or get_tracer())
        # Folded structural spans: id -> the parent its children should attach to.
        self._folded: dict[str, str] = {}
        # With seams installed, model and tool calls are recorded by the seams (so
        # they replay); the SDK's own spans for them are folded, not recorded twice.
        self._seams = seams

    def _recorded_by_seam(self, data: Any) -> bool:
        if not self._seams:
            return False
        if isinstance(data, FunctionSpanData):
            return True  # opened just before the tool's seam call, which records it
        if isinstance(data, (GenerationSpanData, ResponseSpanData)):
            seam = active_seam()  # opened inside the model's seam call
            return seam is not None and seam.kind is SpanKind.LLM
        return False

    def seam_parent(self) -> SpanContext | None:
        """The pyTracer span a seam call made in the current SDK context nests under."""
        from .replay import current_sdk_run_id

        run = current_sdk_run_id()
        while run is not None and run in self._folded:
            run = self._folded[run]
        return self._recorder.context_of(run)

    def _effective_parent(self, span: Span[Any]) -> str:
        """The parent to nest under, skipping any folded (structural) ancestors."""
        parent = span.parent_id or span.trace_id
        while parent in self._folded:
            parent = self._folded[parent]
        return parent

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
        parent = self._effective_parent(span)
        if not isinstance(span.span_data, _KNOWN_SPAN_DATA) or self._recorded_by_seam(span.span_data):
            # A structural span (e.g. "task"/"turn"): fold it so its children nest
            # under its parent instead of an "unknown" span.
            self._folded[span.span_id] = parent
            return
        name, kind, agent_name, tool_name = _classify(span.span_data)
        self._recorder.start(
            span.span_id,
            name=str(name),
            kind=kind,
            agent_name=agent_name,
            tool_name=tool_name,
            parent_run_id=parent,
            attributes={"framework": "openai-agents"},
        )

    def on_span_end(self, span: Span[Any]) -> None:
        if self._folded.pop(span.span_id, None) is not None:
            return
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


_installed_seams: Any = None


def install(tracer: Tracer | None = None, *, replayable: bool = False) -> PyTracerTracingProcessor:
    """Register pyTracer as the sole trace processor and return it.

    Uses ``set_trace_processors`` (replace), not ``add_trace_processor`` (append).
    The Agents SDK ships a default processor that exports traces to OpenAI when
    ``OPENAI_API_KEY`` is set, so appending would quietly upload every run to
    OpenAI as well; replacing sends traces only to pyTracer. Call this after
    ``pytracer.configure(...)``.

    With ``replayable=True``, the SDK's model calls and function tools are also
    routed through pyTracer (see :mod:`pytracer_openai_agents.replay`), so a
    captured run can be replayed without adding ``@op`` to anything.
    """
    global _installed_seams
    from .replay import Seams

    if _installed_seams is not None:
        _installed_seams.uninstall()
        _installed_seams = None
    processor = PyTracerTracingProcessor(tracer, seams=replayable)
    set_trace_processors([processor])
    if replayable:
        _installed_seams = Seams(processor.seam_parent)
        _installed_seams.install()
    return processor


def uninstall_seams() -> None:
    """Remove the replay seams installed by ``install(replayable=True)``."""
    global _installed_seams
    if _installed_seams is not None:
        _installed_seams.uninstall()
        _installed_seams = None
