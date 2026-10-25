"""GenAI semantic-convention helpers shared by the AutoGen adapter's modules."""

from __future__ import annotations

from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from pytracer_sdk import Span, SpanKind, Tracer

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


# Bound on remembered spans and traces, so a long-lived process does not grow
# without limit; far more than any one run holds.
_MAX_TRACKED = 20_000


def _record(tracer: Tracer, span: Span) -> None:
    """Hand a finished span to the tracer's buffer, as built (types kept)."""
    tracer.buffer.record(span)
