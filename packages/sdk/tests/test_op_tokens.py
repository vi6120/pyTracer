"""Token counts on @op model calls: read from the usage the result reports."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import pytracer_sdk as pytracer
from pytracer_sdk import SpanKind, Tracer, usage_tokens
from pytracer_sdk.transport import InMemoryTransport


@pytest.fixture
def sink() -> Any:
    transport = InMemoryTransport()
    tracer = Tracer(transport)
    pytracer.set_tracer(tracer)
    yield transport, tracer
    pytracer.set_tracer(None)


def _spans(sink: Any) -> list[Any]:
    transport, tracer = sink
    tracer.flush()
    return transport.spans


OPENAI = SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=900, completion_tokens=100, total_tokens=1000))
ANTHROPIC = SimpleNamespace(content=[], usage=SimpleNamespace(input_tokens=700, output_tokens=50))
LANGCHAIN = SimpleNamespace(content="hi", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})


@pytest.mark.parametrize(
    "result, expected",
    [
        (OPENAI, 1000),
        (ANTHROPIC, 750),
        (LANGCHAIN, 15),
        ({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 4}}, 7),
        ([{"usage": {"total_tokens": 2}}, {"usage": None}, {"usage": {"total_tokens": 5}}], 7),
        ("plain text", None),
        ({"usage": {"total_tokens": True}}, None),
    ],
)
def test_usage_tokens_reads_the_common_shapes(result: Any, expected: int | None) -> None:
    assert usage_tokens(result) == expected


def test_a_model_op_records_the_tokens_its_result_reports(sink: Any) -> None:
    @pytracer.op(kind=SpanKind.LLM, name="model")
    def model(prompt: str) -> Any:
        return OPENAI

    model("hi")
    (span,) = _spans(sink)
    assert span.token_count == 1000


def test_an_explicit_reader_wins(sink: Any) -> None:
    @pytracer.op(kind=SpanKind.LLM, name="model", tokens=lambda out: 42)
    def model(prompt: str) -> Any:
        return OPENAI

    model("hi")
    (span,) = _spans(sink)
    assert span.token_count == 42


def test_a_tool_is_not_read_for_tokens(sink: Any) -> None:
    @pytracer.op(kind=SpanKind.TOOL, tool_name="lookup")
    def lookup() -> Any:
        return {"usage": {"total_tokens": 99}}

    lookup()
    (span,) = _spans(sink)
    assert span.token_count is None


def test_a_model_returning_text_has_no_count(sink: Any) -> None:
    @pytracer.op(kind=SpanKind.LLM, name="model")
    def model(prompt: str) -> str:
        return "an answer"

    model("hi")
    (span,) = _spans(sink)
    assert span.token_count is None


def test_a_streamed_model_call_sums_its_chunks(sink: Any) -> None:
    @pytracer.op(kind=SpanKind.LLM, name="model")
    def model(prompt: str) -> Any:
        yield {"delta": "a"}
        yield {"delta": "b", "usage": {"prompt_tokens": 8, "completion_tokens": 2}}

    assert [c["delta"] for c in model("hi")] == ["a", "b"]
    (span,) = _spans(sink)
    assert span.token_count == 10


def test_a_tracer_op_takes_a_reader_too(sink: Any) -> None:
    _, tracer = sink

    @tracer.op(kind=SpanKind.LLM, name="model", tokens=lambda out: 7)
    def model(prompt: str) -> str:
        return "x"

    model("hi")
    (span,) = _spans(sink)
    assert span.token_count == 7
