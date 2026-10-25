"""LangChain / LangGraph callback handler that records pyTracer spans.

LangChain reports work through callbacks that carry a ``run_id`` and
``parent_run_id`` (UUIDs) for every LLM call, tool call, and chain/graph node.
This handler translates those callbacks into :class:`~pytracer_sdk.SpanRecorder`
start/end calls, so an agent is instrumented just by passing the handler in the
run config, with no ``@op`` decorators in the agent code.

Mapping:
- ``on_chat_model_start`` / ``on_llm_start`` -> LLM spans, named after the model
  (read from ``invocation_params`` / the LangSmith ``ls_model_name`` metadata) with
  the provider recorded, so a recording is tied to a model version
- ``on_tool_start``                          -> tool spans
- ``on_chain_start``                         -> an AGENT span for each LangGraph node
  run and for the graph root; a HANDOFF span for a conditional-edge routing run
  (which executes under its own function name while carrying the source node's
  ``langgraph_node`` metadata, so it is not counted as another run of the node);
  an UNKNOWN-kind span for any other chain

Scope: the handler only sees work that flows through LangChain callbacks. Model
calls made on a raw provider client, and tools that are plain Python functions
(not LangChain tools), do not emit callbacks and are not captured; instrument those
with ``@op`` if you need them. Replay also needs ``@op``: the handler records but
adds no interception seam, so an adapter-only capture replays live.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from pytracer_sdk import SpanKind, SpanRecorder, Tracer, get_tracer
from pytracer_sdk.tracer import SpanContext

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


# Class names LangGraph gives the compiled graph run, so its root chain is recorded
# as the run's agent rather than an UNKNOWN "LangGraph" span the AGENT column misses.
_GRAPH_ROOT_NAMES = frozenset({"LangGraph", "Pregel", "CompiledStateGraph", "CompiledGraph"})


def _model_and_provider(
    metadata: dict[str, Any] | None, invocation_params: dict[str, Any] | None
) -> tuple[str | None, str | None]:
    """Pull the model name and provider from a chat/LLM start callback.

    LangChain puts the resolved model in ``invocation_params`` (``model`` or
    ``model_name`` / a deployment name) and the LangSmith standard metadata
    (``ls_model_name`` / ``ls_provider``). Either is enough to tie a recording to a
    model version; without them the span would only know the client class name.
    """
    meta = metadata or {}
    inv = invocation_params or {}
    model = (
        meta.get("ls_model_name")
        or inv.get("model")
        or inv.get("model_name")
        or inv.get("deployment_name")
        or inv.get("azure_deployment")
    )
    provider = meta.get("ls_provider")
    return (str(model) if model else None), (str(provider) if provider else None)


def _is_graph_root(
    serialized: dict[str, Any] | None, run_name: str | None, parent_run_id: UUID | None
) -> bool:
    """True when this chain is the compiled graph's own top-level run."""
    if parent_run_id is not None:
        return False
    candidates = {run_name}
    if serialized:
        candidates.add(serialized.get("name"))
        ident = serialized.get("id")
        if isinstance(ident, list) and ident:
            candidates.add(str(ident[-1]))
    return any(c in _GRAPH_ROOT_NAMES for c in candidates if c)


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

    def context_of(self, run_id: str | None) -> SpanContext | None:
        """The pyTracer span recording a LangChain run, for nesting a seam call."""
        return self._recorder.context_of(run_id)

    # --- LLM / chat model -> LLM spans --------------------------------------
    def _llm_attributes(
        self, metadata: dict[str, Any] | None, invocation_params: dict[str, Any] | None
    ) -> tuple[str | None, dict[str, Any]]:
        """Build the LLM span's model name and attributes (framework/model/provider)."""
        model, provider = _model_and_provider(metadata, invocation_params)
        attributes: dict[str, Any] = {"framework": "langgraph"}
        if model:
            attributes["model"] = model
        if provider:
            attributes["provider"] = provider
        return model, attributes

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if self._recorder.adopt(str(run_id), SpanKind.LLM):
            return  # a replay seam records this call
        model, attributes = self._llm_attributes(metadata, kwargs.get("invocation_params"))
        self._recorder.start(
            str(run_id),
            name=model or _name(serialized, "chat_model"),
            kind=SpanKind.LLM,
            parent_run_id=_rid(parent_run_id),
            input=_jsonify(messages),
            attributes=attributes,
        )

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if self._recorder.adopt(str(run_id), SpanKind.LLM):
            return  # a replay seam records this call
        model, attributes = self._llm_attributes(metadata, kwargs.get("invocation_params"))
        self._recorder.start(
            str(run_id),
            name=model or _name(serialized, "llm"),
            kind=SpanKind.LLM,
            parent_run_id=_rid(parent_run_id),
            input=list(prompts),
            attributes=attributes,
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
        if self._recorder.adopt(str(run_id), SpanKind.TOOL):
            return  # a replay seam records this call
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
        own = kwargs.get("name")
        own_name = str(own) if isinstance(own, str) and own else None
        if node:
            # A node run executes under the node's name; a conditional-edge routing
            # function runs under its own function name while still carrying the
            # source node's langgraph_node metadata. Treat the latter as a handoff,
            # so it is not recorded as another run of the node (which inflated the
            # node's count and left a 0 ms agent span whose output was a node name).
            if own_name is not None and own_name != str(node):
                kind, name, agent_name = SpanKind.HANDOFF, own_name, None
            else:
                kind, name, agent_name = SpanKind.AGENT, str(node), str(node)
        elif _is_graph_root(serialized, own_name or _name(serialized, ""), parent_run_id):
            root = own_name or _name(serialized, "LangGraph")
            kind, name, agent_name = SpanKind.AGENT, root, root
        else:
            kind, name, agent_name = SpanKind.UNKNOWN, own_name or _name(serialized, "chain"), None
        self._recorder.start(
            str(run_id),
            name=name,
            kind=kind,
            agent_name=agent_name,
            parent_run_id=_rid(parent_run_id),
            input=_jsonify(inputs),
            attributes={"framework": "langgraph"},
        )

    def on_chain_end(self, outputs: dict[str, Any], *, run_id: UUID, **kwargs: Any) -> None:
        self._recorder.end(str(run_id), output=_jsonify(outputs))

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._recorder.end(str(run_id), error=error)
