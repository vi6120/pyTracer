"""Hermetic tests: the listener maps CrewAI events to spans correctly.

These call the listener's handler methods directly with lightweight fake events
(``register=False`` skips attaching to the global bus), so they exercise the
mapping without running a crew.
"""

from __future__ import annotations

from typing import Any

from navik_sdk import Resource, SpanStatus, Tracer
from navik_sdk import SpanKind as K
from navik_sdk.transport import InMemoryTransport

from navik_crewai import NavikEventListener


class _E:
    """A minimal stand-in for a CrewAI event (duck-typed attributes)."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def _setup() -> tuple[NavikEventListener, InMemoryTransport, Tracer]:
    sink = InMemoryTransport()
    tracer = Tracer(sink, resource=Resource(service_name="cw", branch="main", commit="c0"))
    return NavikEventListener(tracer, register=False), sink, tracer


def test_tool_under_agent_linkage() -> None:
    lis, sink, tracer = _setup()
    lis.on_agent_start(None, _E(event_id="a1", agent_role="planner"))
    lis.on_tool_start(None, _E(event_id="t1", parent_event_id="a1", tool_name="search", tool_args={"q": "cats"}))
    lis.on_tool_end(None, _E(started_event_id="t1", output="results"))
    lis.on_agent_end(None, _E(started_event_id="a1", output="done"))
    tracer.flush()
    spans = {s.name: s for s in sink.spans}
    assert spans["planner"].kind is K.AGENT
    assert spans["search"].kind is K.TOOL and spans["search"].tool_name == "search"
    assert spans["search"].parent_span_id == spans["planner"].span_id
    assert spans["search"].output == "results"


def test_llm_tokens() -> None:
    lis, sink, tracer = _setup()
    lis.on_llm_start(None, _E(event_id="l1", model="gpt-4o", messages=[{"role": "user", "content": "hi"}]))
    lis.on_llm_end(None, _E(started_event_id="l1", response="hello", usage={"total_tokens": 11}))
    tracer.flush()
    (span,) = [s for s in sink.spans if s.kind is K.LLM]
    assert span.name == "gpt-4o"
    assert span.token_count == 11


def test_tool_failure_marks_error() -> None:
    lis, sink, tracer = _setup()
    lis.on_tool_start(None, _E(event_id="t1", tool_name="search", tool_args={}))
    lis.on_tool_end(None, _E(started_event_id="t1", failure="tool blew up"))
    tracer.flush()
    (span,) = sink.spans
    assert span.status is SpanStatus.ERROR


def test_end_without_started_id_is_ignored() -> None:
    lis, sink, tracer = _setup()
    lis.on_tool_start(None, _E(event_id="t1", tool_name="search", tool_args={}))
    lis.on_tool_end(None, _E(started_event_id=None, output="x"))  # no correlation id
    tracer.flush()
    assert sink.spans == []  # the still-open span is never buffered


def test_tool_args_are_redacted() -> None:
    lis, sink, tracer = _setup()
    secret = "sk-abc123DEF456ghi789JKL012mno"
    lis.on_tool_start(None, _E(event_id="t1", tool_name="call", tool_args={"key": secret}))
    lis.on_tool_end(None, _E(started_event_id="t1", output="done"))
    tracer.flush()
    (span,) = sink.spans
    assert secret not in str(span.input)
