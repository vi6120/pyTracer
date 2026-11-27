"""Retry steps: model calls inside pytracer.step are one chain of attempts."""

from __future__ import annotations

import asyncio
from typing import Any

import pydantic
import pytest

import pytracer_sdk as pytracer
from pytracer_sdk import SpanKind, Tracer
from pytracer_sdk.schema import Span
from pytracer_sdk.transport import InMemoryTransport


class Order(pydantic.BaseModel):
    order_id: str = pydantic.Field(pattern=r"^#W\d{7}$")
    total: float


REPLIES = [
    '{"order_id": "W2378156", "total": "12,30"}',  # both fields invalid
    '{"order_id": "#W2378156", "total": "12,30"}',  # total invalid
    '{"order_id": "#W2378156", "total": 12.3}',  # valid
]


@pytest.fixture
def sink() -> Any:
    transport = InMemoryTransport()
    tracer = Tracer(transport)
    pytracer.set_tracer(tracer)
    yield transport, tracer
    pytracer.set_tracer(None)


def _model(replies: list[str]) -> Any:
    @pytracer.op(kind=SpanKind.LLM, name="extractor")
    def model(messages: list[dict[str, str]]) -> str:
        return replies[sum(m["role"] == "user" for m in messages) - 1]

    return model


def _extract(model: Any, tries: int = 3) -> Order | None:
    messages = [{"role": "user", "content": "Extract the order."}]
    with pytracer.step("extract_order") as step:
        for _ in range(tries):
            raw = model(messages)
            try:
                return Order.model_validate_json(raw)
            except pydantic.ValidationError as exc:
                step.reject(exc)
                messages = [*messages, {"role": "assistant", "content": raw},
                            {"role": "user", "content": f"Fix: {exc.errors()[0]['msg']}"}]
    return None


def _spans(sink: Any) -> tuple[list[Span], Span]:
    transport, tracer = sink
    tracer.flush()
    calls = [s for s in transport.spans if s.kind is SpanKind.LLM]
    step = next(s for s in transport.spans if "pytracer.step.attempts" in s.attributes)
    return calls, step


def test_a_validation_retry_loop_is_one_chain_of_attempts(sink: Any) -> None:
    assert _extract(_model(REPLIES)) == Order(order_id="#W2378156", total=12.3)
    calls, step = _spans(sink)
    assert [c.attributes["pytracer.attempt"] for c in calls] == [1, 2, 3]
    assert "pytracer.retry_of" not in calls[0].attributes
    assert calls[1].attributes["pytracer.retry_of"] == calls[0].span_id
    assert calls[2].attributes["pytracer.retry_of"] == calls[1].span_id
    assert all(c.attributes["pytracer.step"] == "extract_order" for c in calls)
    assert all(c.parent_span_id == step.span_id for c in calls)
    assert step.attributes["pytracer.step.attempts"] == 3
    assert step.attributes["pytracer.step.accepted"] == 3
    assert [(r["attempt"], r["error_type"]) for r in step.attributes["pytracer.step.rejections"]] == [
        (1, "ValidationError"), (2, "ValidationError")]


def test_a_step_that_never_validates_accepts_nothing(sink: Any) -> None:
    assert _extract(_model([REPLIES[0]] * 3)) is None
    _, step = _spans(sink)
    assert step.attributes["pytracer.step.attempts"] == 3
    assert step.attributes["pytracer.step.accepted"] is None
    assert len(step.attributes["pytracer.step.rejections"]) == 3


def test_a_step_that_raises_accepts_nothing(sink: Any) -> None:
    model = _model(REPLIES)
    with pytest.raises(RuntimeError), pytracer.step("extract_order"):
        model([{"role": "user", "content": "x"}])
        raise RuntimeError("gave up")
    _, step = _spans(sink)
    assert step.status.value == "error"
    assert step.attributes["pytracer.step.accepted"] is None


def test_a_rejection_message_is_redacted(sink: Any) -> None:
    with pytracer.step("extract_order") as step:
        _model(REPLIES)([{"role": "user", "content": "x"}])
        step.reject("bad value: yusuf.rossi7110@example.com")
    _, step_span = _spans(sink)
    error = step_span.attributes["pytracer.step.rejections"][0]["error"]
    assert "example.com" not in error and "[redacted:email]" in error


def test_a_model_call_nested_in_a_tool_is_not_an_attempt(sink: Any) -> None:
    model = _model(REPLIES)

    @pytracer.op(kind=SpanKind.TOOL, tool_name="lookup")
    def lookup() -> str:
        return model([{"role": "user", "content": "x"}])

    with pytracer.step("extract_order"):
        lookup()
        model([{"role": "user", "content": "x"}])
    transport, tracer = sink
    tracer.flush()
    nested, direct = [s for s in transport.spans if s.kind is SpanKind.LLM]
    assert "pytracer.step" not in nested.attributes
    assert direct.attributes["pytracer.attempt"] == 1


def test_steps_work_in_async_code(sink: Any) -> None:
    @pytracer.op(kind=SpanKind.LLM, name="extractor")
    async def model(n: int) -> str:
        return REPLIES[n]

    async def run() -> Order:
        with pytracer.step("extract_order") as step:
            for n in range(3):
                raw = await model(n)
                try:
                    return Order.model_validate_json(raw)
                except pydantic.ValidationError as exc:
                    step.reject(exc)
        raise AssertionError("never validated")

    asyncio.run(run())
    calls, step = _spans(sink)
    assert [c.attributes["pytracer.attempt"] for c in calls] == [1, 2, 3]
    assert step.attributes["pytracer.step.accepted"] == 3


def test_outside_a_step_only_identical_calls_are_numbered(sink: Any) -> None:
    model = _model(REPLIES)
    model([{"role": "user", "content": "x"}])
    model([{"role": "user", "content": "x"}, {"role": "user", "content": "y"}])
    transport, tracer = sink
    tracer.flush()
    assert all("pytracer.attempt" not in s.attributes for s in transport.spans)
