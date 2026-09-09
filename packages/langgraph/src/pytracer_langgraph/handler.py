"""LangChain / LangGraph callback handler that records pyTracer spans.

LangChain reports work through callbacks that carry a ``run_id`` and
``parent_run_id`` (UUIDs) for every LLM call, tool call, and chain/graph node.
This handler translates those callbacks into :class:`~pytracer_sdk.SpanRecorder`
start/end calls, so an agent is instrumented just by passing the handler in the
run config, with no ``@op`` decorators in the agent code.

Mapping:
- ``on_chat_model_start`` / ``on_llm_start`` -> LLM spans
- ``on_tool_start``                          -> tool spans
- ``on_chain_start``                         -> agent spans for LangGraph nodes,
                                                unknown-kind spans for other chains
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from pytracer_sdk import SpanKind, SpanRecorder, Tracer, get_tracer

_MAX_DEPTH = 6


def _rid(run_id: UUID | None) -> str | None:
    return str(run_id) if run_id is not None else None


def _name(serialized: dict[str, Any] | None, default: str) -> str:
    if not serialized:
        return default
    name = serialized.get("name")
    if isinstance(name, str) and name:
        return name
    ident = serialized.get("id")
    if isinstance(ident, list) and ident:
        return str(ident[-1])
    return default


def _jsonify(value: Any, depth: int = 0) -> Any:
    """Coerce LangChain objects (messages, etc.) to JSON-friendly primitives.

    Spans are serialized for storage and transport, so framework objects must be
    reduced to strings, numbers, lists, and dicts before they reach a span.
    """
    if depth > _MAX_DEPTH:
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonify(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v, depth + 1) for v in value]
    content = getattr(value, "content", None)  # BaseMessage-like
    if content is not None and not callable(content):
        return {"type": type(value).__name__, "content": _jsonify(content, depth + 1)}
    return str(value)


def _extract_llm_output(response: LLMResult) -> tuple[Any, int | None]:
    """Pull the generated text and total token count out of an LLMResult."""
    texts: list[str] = []
    tokens: int | None = None
    for batch in response.generations:
        for gen in batch:
            texts.append(getattr(gen, "text", "") or "")
            message = getattr(gen, "message", None)
            usage = getattr(message, "usage_metadata", None) if message is not None else None
            if isinstance(usage, dict) and usage.get("total_tokens") is not None:
                tokens = (tokens or 0) + int(usage["total_tokens"])
    if tokens is None and response.llm_output:
        block = response.llm_output.get("token_usage") or response.llm_output.get("usage") or {}
        if isinstance(block, dict) and block.get("total_tokens") is not None:
            tokens = int(block["total_tokens"])
    output: Any = texts[0] if len(texts) == 1 else texts
    return output, tokens


class PyTracerCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that records spans into pyTracer."""

    def __init__(self, tracer: Tracer | None = None) -> None:
        self._recorder = SpanRecorder(tracer or get_tracer())

    # --- LLM / chat model -> LLM spans --------------------------------------
    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._recorder.start(
            str(run_id),
            name=_name(serialized, "chat_model"),
            kind=SpanKind.LLM,
            parent_run_id=_rid(parent_run_id),
            input=_jsonify(messages),
            attributes={"framework": "langgraph"},
        )

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._recorder.start(
            str(run_id),
            name=_name(serialized, "llm"),
            kind=SpanKind.LLM,
            parent_run_id=_rid(parent_run_id),
            input=list(prompts),
            attributes={"framework": "langgraph"},
        )

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        output, tokens = _extract_llm_output(response)
        self._recorder.end(str(run_id), output=output, token_count=tokens)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._recorder.end(str(run_id), error=error)

    # --- tools -> tool spans ------------------------------------------------
    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = _name(serialized, "tool")
        self._recorder.start(
            str(run_id),
            name=name,
            kind=SpanKind.TOOL,
            tool_name=name,
            parent_run_id=_rid(parent_run_id),
            input=_jsonify(inputs if inputs is not None else input_str),
            attributes={"framework": "langgraph"},
        )

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._recorder.end(str(run_id), output=_jsonify(output))

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._recorder.end(str(run_id), error=error)

    # --- chains / graph nodes -> agent or unknown spans ---------------------
    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        node = (metadata or {}).get("langgraph_node")
        name = node or kwargs.get("name") or _name(serialized, "chain")
        self._recorder.start(
            str(run_id),
            name=str(name),
            kind=SpanKind.AGENT if node else SpanKind.UNKNOWN,
            agent_name=str(node) if node else None,
            parent_run_id=_rid(parent_run_id),
            input=_jsonify(inputs),
            attributes={"framework": "langgraph"},
        )

    def on_chain_end(self, outputs: dict[str, Any], *, run_id: UUID, **kwargs: Any) -> None:
        self._recorder.end(str(run_id), output=_jsonify(outputs))

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._recorder.end(str(run_id), error=error)
