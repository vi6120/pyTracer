"""Integration test against a real LangGraph graph.

Builds a two-node graph (a fake chat model plus a tool) and confirms that a
single ``invoke`` with the handler captures the model call, the tool call, and
the graph nodes as one connected trace, with no instrumentation in the agent
code itself. Skipped when langgraph is not installed.
"""

from __future__ import annotations

import importlib.util
from typing import TypedDict

import navik_sdk as navik
import pytest
from navik_sdk import SpanKind
from navik_sdk.transport import InMemoryTransport

from navik_langgraph import NavikCallbackHandler

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("langgraph") is None, reason="langgraph not installed"
)


class _State(TypedDict):
    messages: list


def test_real_graph_run_is_captured_as_one_trace() -> None:
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import tool
    from langgraph.graph import END, START, StateGraph

    sink = InMemoryTransport()
    tracer = navik.Tracer(sink, resource=navik.Resource(service_name="lg", branch="main", commit="c0"))

    @tool
    def search(query: str) -> str:
        """search the web"""
        return f"results for {query}"

    model = GenericFakeChatModel(messages=iter([AIMessage(content="final")]))

    def call_model(state: _State) -> dict:
        return {"messages": state["messages"] + [model.invoke(state["messages"])]}

    def call_tool(state: _State) -> dict:
        return {"messages": state["messages"] + [AIMessage(content=search.invoke({"query": "cats"}))]}

    graph = StateGraph(_State)
    graph.add_node("model", call_model)
    graph.add_node("tool", call_tool)
    graph.add_edge(START, "model")
    graph.add_edge("model", "tool")
    graph.add_edge("tool", END)
    app = graph.compile()

    app.invoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"callbacks": [NavikCallbackHandler(tracer)]},
    )
    tracer.flush()

    spans = sink.spans
    # One connected trace for the whole run.
    assert len({s.trace_id for s in spans}) == 1
    kinds = {s.kind for s in spans}
    assert SpanKind.LLM in kinds  # the model call was captured
    assert SpanKind.TOOL in kinds  # the tool call was captured
    assert SpanKind.AGENT in kinds  # graph nodes captured as agent spans
    tool_span = next(s for s in spans if s.kind is SpanKind.TOOL)
    assert tool_span.tool_name == "search"
    assert "results for cats" in str(tool_span.output)
    # Every non-root span links to a parent within the trace.
    roots = [s for s in spans if s.parent_span_id is None]
    assert len(roots) == 1
