"""Hermetic tests: the handler maps LangChain callbacks to spans correctly.

These call the handler directly with synthesized callback arguments, so they run
without building a real graph (they need only langchain-core, a runtime dep).
"""

from __future__ import annotations

import json
import uuid

from langchain_core.messages import HumanMessage
from langchain_core.outputs import Generation, LLMResult
from pytracer_sdk import Resource, SpanStatus, Tracer
from pytracer_sdk import SpanKind as K
from pytracer_sdk.transport import InMemoryTransport

from pytracer_langgraph import PyTracerCallbackHandler


def _setup() -> tuple[PyTracerCallbackHandler, InMemoryTransport, Tracer]:
    sink = InMemoryTransport()
    tracer = Tracer(sink, resource=Resource(service_name="t", branch="main", commit="c0"))
    return PyTracerCallbackHandler(tracer), sink, tracer


def test_llm_span_mapping_with_tokens() -> None:
    handler, sink, tracer = _setup()
    rid = uuid.uuid4()
    handler.on_chat_model_start({"name": "gpt"}, [[]], run_id=rid)
    handler.on_llm_end(
        LLMResult(generations=[[Generation(text="hello")]], llm_output={"token_usage": {"total_tokens": 9}}),
        run_id=rid,
    )
    tracer.flush()
    (span,) = sink.spans
    assert span.kind is K.LLM
    assert span.name == "gpt"
    assert span.output == "hello"
    assert span.token_count == 9


def test_tool_span_and_parent_linkage() -> None:
    handler, sink, tracer = _setup()
    parent, child = uuid.uuid4(), uuid.uuid4()
    handler.on_chain_start({"name": "node"}, {}, run_id=parent, metadata={"langgraph_node": "node"})
    handler.on_tool_start({"name": "search"}, "cats", run_id=child, parent_run_id=parent, inputs={"query": "cats"})
    handler.on_tool_end("results", run_id=child)
    handler.on_chain_end({}, run_id=parent)
    tracer.flush()
    spans = {s.name: s for s in sink.spans}
    assert spans["node"].kind is K.AGENT
    assert spans["search"].kind is K.TOOL
    assert spans["search"].tool_name == "search"
    assert spans["search"].parent_span_id == spans["node"].span_id
    assert spans["search"].trace_id == spans["node"].trace_id


def test_non_node_chain_is_unknown_kind() -> None:
    handler, sink, tracer = _setup()
    rid = uuid.uuid4()
    handler.on_chain_start({"name": "RunnableSequence"}, {}, run_id=rid)
    handler.on_chain_end({}, run_id=rid)
    tracer.flush()
    (span,) = sink.spans
    assert span.kind is K.UNKNOWN


def test_error_maps_to_failed_span() -> None:
    handler, sink, tracer = _setup()
    rid = uuid.uuid4()
    handler.on_tool_start({"name": "t"}, "x", run_id=rid)
    handler.on_tool_error(ValueError("boom"), run_id=rid)
    tracer.flush()
    (span,) = sink.spans
    assert span.status is SpanStatus.ERROR
    assert span.error_type == "ValueError"


def test_messages_are_jsonified_and_redacted() -> None:
    handler, sink, tracer = _setup()
    rid = uuid.uuid4()
    secret = "sk-abc123DEF456ghi789JKL012mno"
    handler.on_chat_model_start({}, [[HumanMessage(content=f"my key {secret}")]], run_id=rid)
    handler.on_llm_end(LLMResult(generations=[[Generation(text="ok")]]), run_id=rid)
    tracer.flush()
    (span,) = sink.spans
    assert secret not in str(span.input)
    json.dumps(span.input)  # must be JSON-serializable for storage/transport
