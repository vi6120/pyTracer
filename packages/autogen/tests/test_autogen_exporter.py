"""Exporter tests against the real OpenTelemetry SDK.

These emit GenAI-convention OTel spans through the pyTracer exporter (no AutoGen
needed, only opentelemetry-sdk, a runtime dependency) and assert the mapping.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest
import pytracer_sdk as pytracer
from opentelemetry.trace import Status, StatusCode
from pytracer_sdk import SpanKind as K
from pytracer_sdk import SpanStatus
from pytracer_sdk.transport import InMemoryTransport

from pytracer_autogen import install, pytracer_tracer_provider


def _setup() -> tuple[object, InMemoryTransport, pytracer.Tracer]:
    sink = InMemoryTransport()
    tracer = pytracer.Tracer(sink, resource=pytracer.Resource(service_name="ag", branch="main", commit="c0"))
    return pytracer_tracer_provider(tracer).get_tracer("test"), sink, tracer


def test_kinds_and_parent_linkage() -> None:
    ot, sink, tracer = _setup()
    with ot.start_as_current_span("invoke_agent planner") as agent:  # type: ignore[attr-defined]
        agent.set_attribute("gen_ai.operation.name", "invoke_agent")
        agent.set_attribute("gen_ai.agent.name", "planner")
        with ot.start_as_current_span("execute_tool search") as tool:  # type: ignore[attr-defined]
            tool.set_attribute("gen_ai.operation.name", "execute_tool")
            tool.set_attribute("gen_ai.tool.name", "search")
    tracer.flush()
    agent_span = next(s for s in sink.spans if s.kind is K.AGENT)
    tool_span = next(s for s in sink.spans if s.kind is K.TOOL)
    assert agent_span.agent_name == "planner"
    assert tool_span.tool_name == "search"
    assert tool_span.parent_span_id == agent_span.span_id
    assert tool_span.trace_id == agent_span.trace_id


def test_llm_kind_and_tokens() -> None:
    ot, sink, tracer = _setup()
    with ot.start_as_current_span("chat gpt-4o") as span:  # type: ignore[attr-defined]
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.usage.total_tokens", 42)
    tracer.flush()
    (llm,) = [s for s in sink.spans if s.kind is K.LLM]
    assert llm.token_count == 42


def test_error_status_is_mapped() -> None:
    ot, sink, tracer = _setup()
    with ot.start_as_current_span("chat") as span:  # type: ignore[attr-defined]
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("error.type", "ValueError")
        span.set_status(Status(StatusCode.ERROR))
    tracer.flush()
    (span,) = sink.spans
    assert span.status is SpanStatus.ERROR
    assert span.error_type == "ValueError"


def test_runtime_messaging_span_without_genai_attrs_is_skipped() -> None:
    # AutoGen's internal messaging spans carry no GenAI attributes; they are noise
    # and must not be recorded (A5: 44 "unknown" id-named spans per run).
    ot, sink, tracer = _setup()
    with ot.start_as_current_span("some.internal.op"):  # type: ignore[attr-defined]
        pass
    tracer.flush()
    assert sink.spans == []


def test_model_span_recognized_from_request_model_attr() -> None:
    # A model call that arrives with only gen_ai.request.model (no operation name)
    # is still classified as an LLM span, not dropped as unknown.
    ot, sink, tracer = _setup()
    with ot.start_as_current_span("ChatCompletion") as span:  # type: ignore[attr-defined]
        span.set_attribute("gen_ai.request.model", "gpt-5-mini")
    tracer.flush()
    (llm,) = [s for s in sink.spans if s.kind is K.LLM]
    assert llm.attributes["gen_ai.request.model"] == "gpt-5-mini"


def test_model_output_read_from_span_events() -> None:
    # Newer GenAI instrumentation records prompt/response as span events, not
    # attributes; the exporter must still capture them (A5: empty outputs).
    ot, sink, tracer = _setup()
    with ot.start_as_current_span("chat gpt-4o") as span:  # type: ignore[attr-defined]
        span.set_attribute("gen_ai.operation.name", "chat")
        span.add_event("gen_ai.user.message", {"content": "how many words?"})
        span.add_event("gen_ai.choice", {"content": "34876"})
    tracer.flush()
    (llm,) = [s for s in sink.spans if s.kind is K.LLM]
    assert "how many words?" in str(llm.input)
    assert "34876" in str(llm.output)


def test_attributes_are_redacted() -> None:
    ot, sink, tracer = _setup()
    secret = "sk-abc123DEF456ghi789JKL012mno"
    with ot.start_as_current_span("chat") as span:  # type: ignore[attr-defined]
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.input", f"my key {secret}")
    tracer.flush()
    (span,) = sink.spans
    assert secret not in str(span.input)
    assert secret not in str(span.attributes)


def test_exporter_built_before_configure_uses_the_configured_tracer() -> None:
    # P8: the tracer is resolved at export time, so an exporter built before
    # pytracer.configure(...) still records into the configured transport.
    ot = pytracer_tracer_provider().get_tracer("test")  # built with no explicit tracer
    sink = InMemoryTransport()
    previous = pytracer.get_tracer()
    pytracer.configure(sink, service_name="ag", branch="main", commit="c0")
    try:
        with ot.start_as_current_span("chat gpt-4o") as span:  # type: ignore[attr-defined]
            span.set_attribute("gen_ai.operation.name", "chat")
        pytracer.get_tracer().flush()
    finally:
        pytracer.set_tracer(previous)
    assert any(s.kind is K.LLM for s in sink.spans)


def test_install_registers_the_provider_globally(monkeypatch: pytest.MonkeyPatch) -> None:
    # P1/P2: install() registers pyTracer on the global OTel provider, where
    # AutoGen's GenAI spans go (passing a provider only to the runtime misses them).
    from opentelemetry import trace as otel_trace

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        otel_trace, "set_tracer_provider", lambda provider: captured.__setitem__("provider", provider)
    )
    provider = install()
    assert captured["provider"] is provider


# --- P2: one trace per run ---------------------------------------------------
# AutoGen starts a new OTel trace for each message and links it to the sender's
# trace (a span link, not a parent). These build that shape with plain OTel spans.


def _genai(span: object, operation: str, **attrs: str) -> None:
    span.set_attribute("gen_ai.operation.name", operation)  # type: ignore[attr-defined]
    for key, value in attrs.items():
        span.set_attribute(f"gen_ai.{key}", value)  # type: ignore[attr-defined]


def _message_hop(ot: object, sender_span: object) -> object:
    """A runtime 'publish' root in a new trace, linked to the sender's span."""
    from opentelemetry.trace import Link

    return ot.start_as_current_span(  # type: ignore[attr-defined]
        "autogen publish", context=None, links=[Link(sender_span.get_span_context())]  # type: ignore[attr-defined]
    )


def test_linked_message_traces_are_one_trace() -> None:
    from opentelemetry import context as otel_context

    ot, sink, tracer = _setup()
    token = otel_context.attach(otel_context.Context())  # start from no active span
    try:
        with ot.start_as_current_span("invoke_agent writer") as writer:  # type: ignore[attr-defined]
            _genai(writer, "invoke_agent", **{"agent.name": "writer"})
            with ot.start_as_current_span("autogen create group_topic") as create:  # type: ignore[attr-defined]
                pass
        # The next message is delivered in a fresh trace, linked to `create`.
        with _message_hop(ot, create):  # type: ignore[attr-defined]
            with ot.start_as_current_span("autogen process critic"):  # type: ignore[attr-defined]
                with ot.start_as_current_span("invoke_agent critic") as critic:  # type: ignore[attr-defined]
                    _genai(critic, "invoke_agent", **{"agent.name": "critic"})
    finally:
        otel_context.detach(token)
    tracer.flush()
    spans = {s.agent_name: s for s in sink.spans}
    assert spans["writer"].trace_id == spans["critic"].trace_id  # one trace for the run
    assert spans["critic"].parent_span_id is None  # top of the run, not a dropped noise span


def test_unlinked_runs_stay_separate_traces() -> None:
    from opentelemetry import context as otel_context

    ot, sink, tracer = _setup()
    for name in ("first", "second"):
        token = otel_context.attach(otel_context.Context())
        try:
            with ot.start_as_current_span(f"invoke_agent {name}") as span:  # type: ignore[attr-defined]
                _genai(span, "invoke_agent", **{"agent.name": name})
        finally:
            otel_context.detach(token)
    tracer.flush()
    assert len({s.trace_id for s in sink.spans}) == 2


def test_tool_nests_under_its_agent_across_dropped_messaging_spans() -> None:
    ot, sink, tracer = _setup()
    with ot.start_as_current_span("invoke_agent planner") as agent:  # type: ignore[attr-defined]
        _genai(agent, "invoke_agent", **{"agent.name": "planner"})
        with ot.start_as_current_span("autogen create output_topic"):  # type: ignore[attr-defined]
            with ot.start_as_current_span("execute_tool search") as tool:  # type: ignore[attr-defined]
                _genai(tool, "execute_tool", **{"tool.name": "search"})
    tracer.flush()
    agent_span = next(s for s in sink.spans if s.kind is K.AGENT)
    tool_span = next(s for s in sink.spans if s.kind is K.TOOL)
    assert tool_span.parent_span_id == agent_span.span_id  # not the dropped noise span


def test_run_nests_under_an_enclosing_pytracer_span() -> None:
    ot, sink, tracer = _setup()
    with tracer.start_span("request") as request:
        with ot.start_as_current_span("invoke_agent writer") as writer:  # type: ignore[attr-defined]
            _genai(writer, "invoke_agent", **{"agent.name": "writer"})
    tracer.flush()
    request_span = next(s for s in sink.spans if s.name == "request")
    writer_span = next(s for s in sink.spans if s.agent_name == "writer")
    assert writer_span.trace_id == request_span.trace_id
    assert writer_span.parent_span_id == request.span.span_id


# --- tool results (AutoGen sends them in a later runtime message) ---------------


def _tool_call(ot: object, call_id: str) -> None:
    with ot.start_as_current_span("execute_tool search") as tool:  # type: ignore[attr-defined]
        _genai(tool, "execute_tool", **{"tool.name": "search", "tool.call.id": call_id})


def _result_message(ot: object, call_id: str, content: str, *, is_error: bool = False) -> None:
    import json

    message = {
        "message": {
            "type": "ToolCallExecutionEvent",
            "source": "planner",
            "content": [{"content": content, "name": "search", "call_id": call_id, "is_error": is_error}],
        }
    }
    with ot.start_as_current_span("autogen process manager") as span:  # type: ignore[attr-defined]
        span.set_attribute("message", json.dumps(message))


def test_tool_output_is_joined_from_the_result_message() -> None:
    ot, sink, tracer = _setup()
    with ot.start_as_current_span("invoke_agent planner") as agent:  # type: ignore[attr-defined]
        _genai(agent, "invoke_agent", **{"agent.name": "planner"})
        _tool_call(ot, "c1")
        _result_message(ot, "c1", "3 results")
    tracer.flush()
    (tool,) = [s for s in sink.spans if s.kind is K.TOOL]
    assert tool.output == "3 results"
    assert tool.status is SpanStatus.OK


def test_tool_error_result_marks_the_span_failed() -> None:
    ot, sink, tracer = _setup()
    with ot.start_as_current_span("invoke_agent planner") as agent:  # type: ignore[attr-defined]
        _genai(agent, "invoke_agent", **{"agent.name": "planner"})
        _tool_call(ot, "c2")
        _result_message(ot, "c2", "Error: index offline", is_error=True)
    tracer.flush()
    (tool,) = [s for s in sink.spans if s.kind is K.TOOL]
    assert tool.status is SpanStatus.ERROR
    assert tool.error_message == "Error: index offline"


def test_tool_span_without_a_result_is_still_recorded_on_flush() -> None:
    sink = InMemoryTransport()
    tracer = pytracer.Tracer(sink, resource=pytracer.Resource(service_name="ag", branch="main", commit="c0"))
    provider = pytracer_tracer_provider(tracer)
    ot = provider.get_tracer("test")
    _tool_call(ot, "never-answered")
    provider.force_flush()
    tracer.flush()
    (tool,) = [s for s in sink.spans if s.kind is K.TOOL]
    assert tool.output is None


# --- model calls from AutoGen's model-call log events ---------------------------


def _log_event(event: object) -> None:
    import logging

    logging.getLogger("autogen_core.events").info(event)


def test_logged_model_call_becomes_an_llm_span_under_its_agent() -> None:
    pytest.importorskip("autogen_core", reason="autogen-core not installed")
    from autogen_core.logging import LLMCallEvent

    ot, sink, tracer = _setup()
    with ot.start_as_current_span("invoke_agent planner") as agent:  # type: ignore[attr-defined]
        _genai(agent, "invoke_agent", **{"agent.name": "planner"})
        _log_event(
            LLMCallEvent(
                messages=[{"role": "user", "content": "plan a trip"}],
                response={"model": "gpt-4o", "choices": [{"message": {"role": "assistant", "content": "ok"}}]},
                prompt_tokens=7,
                completion_tokens=3,
            )
        )
    tracer.flush()
    agent_span = next(s for s in sink.spans if s.kind is K.AGENT)
    (llm,) = [s for s in sink.spans if s.kind is K.LLM]
    assert llm.parent_span_id == agent_span.span_id
    assert llm.trace_id == agent_span.trace_id
    assert llm.name == "chat gpt-4o"
    assert llm.input == [{"role": "user", "content": "plan a trip"}]
    assert llm.output == {"role": "assistant", "content": "ok"}
    assert llm.token_count == 10
    assert llm.attributes["pytracer.start_unknown"] is True


def test_streamed_model_call_spans_start_to_end() -> None:
    pytest.importorskip("autogen_core", reason="autogen-core not installed")
    from autogen_core.logging import LLMStreamEndEvent, LLMStreamStartEvent

    ot, sink, tracer = _setup()
    with ot.start_as_current_span("invoke_agent planner") as agent:  # type: ignore[attr-defined]
        _genai(agent, "invoke_agent", **{"agent.name": "planner"})
        _log_event(LLMStreamStartEvent(messages=[{"role": "user", "content": "hi"}]))
        _log_event(
            LLMStreamEndEvent(
                response={"model": "gpt-4o", "choices": [{"message": {"content": "hello"}}]},
                prompt_tokens=1,
                completion_tokens=1,
            )
        )
    tracer.flush()
    (llm,) = [s for s in sink.spans if s.kind is K.LLM]
    assert llm.input == [{"role": "user", "content": "hi"}]
    assert llm.output == {"content": "hello"}
    assert llm.end_time_ns is not None and llm.end_time_ns >= llm.start_time_ns
    assert "pytracer.start_unknown" not in llm.attributes


def test_event_logger_is_enabled_without_changing_other_handlers() -> None:
    import logging

    logger = logging.getLogger("autogen_core.events")
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    _setup()
    assert logger.getEffectiveLevel() <= logging.INFO
    assert logger.propagate is False  # events were dropped before; nobody else sees them now either

    logger.setLevel(logging.DEBUG)  # the user already listens: leave their setup alone
    logger.propagate = True
    _setup()
    assert logger.level == logging.DEBUG
    assert logger.propagate is True


def test_a_malformed_model_event_never_raises() -> None:
    _setup()
    _log_event("not an autogen event")
    _log_event(type("E", (), {"kwargs": {"type": "LLMCall", "response": None}})())


# --- P7: model content sent as GenAI log records -------------------------------


def _event_name(name: str) -> dict[str, Any]:
    """The log record's event name in the form this OpenTelemetry supports: the
    ``event_name`` field (1.35 and later) or the ``event.name`` attribute (before)."""
    import inspect

    from opentelemetry._logs import LogRecord

    if "event_name" in inspect.signature(LogRecord.__init__).parameters:
        return {"event_name": name}
    return {"attributes": {"event.name": name}}


def test_model_content_from_genai_log_records_fills_the_llm_span() -> None:
    # opentelemetry-instrumentation-openai-v2 leaves the chat span empty and sends
    # the prompt and reply as log records carrying the span's id.
    from opentelemetry._logs import LogRecord
    from opentelemetry.sdk._logs import LoggerProvider

    from pytracer_autogen import genai_log_processor

    logs = LoggerProvider()
    logs.add_log_record_processor(genai_log_processor())
    logger = logs.get_logger("gen_ai.test")
    ot, sink, tracer = _setup()
    with ot.start_as_current_span("chat gpt-4o-mini") as span:  # type: ignore[attr-defined]
        span.set_attribute("gen_ai.operation.name", "chat")
        span_id = span.get_span_context().span_id
        for name, body in (
            ("gen_ai.system.message", {"content": "be brief"}),
            ("gen_ai.user.message", {"content": "weather?"}),
            ("gen_ai.choice", {"index": 0, "message": {"role": "assistant", "content": "sunny"}}),
        ):
            logger.emit(LogRecord(span_id=span_id, body=body, **_event_name(name)))
    tracer.flush()
    (llm,) = [s for s in sink.spans if s.kind is K.LLM]
    assert llm.input == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "weather?"},
    ]
    assert llm.output == {"role": "assistant", "content": "sunny"}


def test_uninstall_seams_lets_the_exporter_record_again() -> None:
    from pytracer_autogen import exporter_for, install_seams, uninstall_seams

    pytest.importorskip("autogen_core", reason="autogen-core not installed")
    exporter = exporter_for(pytracer_tracer_provider())
    assert exporter is not None
    install_seams(exporter)
    assert exporter.seams is True
    uninstall_seams()
    assert exporter.seams is False  # derived model calls and AutoGen's own spans resume


def test_a_tool_span_whose_result_never_comes_is_swept_on_a_later_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No thread per held span: an expired one is recorded by the next export.
    import pytracer_autogen.tool_results as tool_results

    monkeypatch.setattr(tool_results, "_RESULT_WAIT_SECONDS", 0.0)
    ot, sink, tracer = _setup()
    _tool_call(ot, "never-answered")
    with ot.start_as_current_span("invoke_agent planner") as agent:  # type: ignore[attr-defined]
        _genai(agent, "invoke_agent", **{"agent.name": "planner"})  # any later export
    tracer.flush()
    (tool,) = [s for s in sink.spans if s.kind is K.TOOL]
    assert tool.output is None


def test_a_held_tool_span_is_recorded_when_the_process_goes_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No later export comes (an idle process): the background sweep records the
    # span once its wait is over, without the result.
    import time

    import pytracer_autogen.tool_results as tool_results

    monkeypatch.setattr(tool_results, "_RESULT_WAIT_SECONDS", 0.3)
    ot, sink, tracer = _setup()
    _tool_call(ot, "never-answered")
    deadline = time.monotonic() + 0.3 + 0.5
    tools: list[pytracer.Span] = []
    while time.monotonic() < deadline and not tools:
        time.sleep(0.02)
        tracer.flush()
        tools = [s for s in sink.spans if s.kind is K.TOOL]
    assert len(tools) == 1 and tools[0].output is None


def test_one_sweeper_thread_runs_only_while_tool_spans_are_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pytracer_autogen.tool_results as tool_results
    from pytracer_autogen import exporter_for

    monkeypatch.setattr(tool_results, "_RESULT_WAIT_SECONDS", 30.0)
    tracer = pytracer.Tracer(InMemoryTransport(), resource=pytracer.Resource(service_name="ag"))
    provider = pytracer_tracer_provider(tracer)
    ot = provider.get_tracer("test")
    exporter = exporter_for(provider)
    assert exporter is not None and exporter._results is not None
    joiner = exporter._results
    assert joiner._sweeper is None  # no thread before any span is held
    for call_id in ("a", "b", "c"):
        _tool_call(ot, call_id)
    sweeper = joiner._sweeper
    assert sweeper is not None and sweeper.is_alive()
    names = [t.name for t in threading.enumerate() if t is sweeper]
    assert names == ["pytracer-autogen-tool-results"]  # one for all held spans
    joiner.flush()  # flush records them and the thread stops
    sweeper.join(timeout=2.0)
    assert not sweeper.is_alive() and joiner._sweeper is None


def test_exporter_for_finds_only_pytracer_providers() -> None:
    from opentelemetry.sdk.trace import TracerProvider

    from pytracer_autogen import PyTracerSpanExporter, exporter_for

    assert isinstance(exporter_for(pytracer_tracer_provider()), PyTracerSpanExporter)
    assert exporter_for(TracerProvider()) is None
    assert exporter_for(object()) is None
