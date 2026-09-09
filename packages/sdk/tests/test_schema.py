"""Span schema: field serialization, validation, latency, and OTLP mapping."""

from __future__ import annotations

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pytracer_sdk import new_span_id, new_trace_id
from pytracer_sdk.ids import is_valid_span_id, is_valid_trace_id
from pytracer_sdk.schema import Span, SpanKind, SpanStatus


def _span(**overrides: object) -> Span:
    base: dict[str, object] = {
        "trace_id": new_trace_id(),
        "span_id": new_span_id(),
        "name": "call_model",
        "kind": SpanKind.LLM,
        "start_time_ns": 1_000_000_000,
        "end_time_ns": 1_050_000_000,
    }
    base.update(overrides)
    return Span(**base)  # type: ignore[arg-type]


def test_ids_have_otel_shape() -> None:
    assert is_valid_trace_id(new_trace_id())
    assert is_valid_span_id(new_span_id())
    assert not is_valid_trace_id("00" * 16)  # all-zero is invalid
    assert not is_valid_span_id("xyz")


def test_every_field_round_trips_through_json() -> None:
    span = _span(
        agent_name="planner",
        tool_name="search",
        input={"q": "hello"},
        output=["a", "b"],
        token_count=42,
        cost_usd=0.0013,
        attributes={"model": "opus"},
    )
    restored = Span.from_json(span.to_json())
    assert restored == span
    # And it is valid JSON with the expected keys.
    payload = json.loads(span.to_json())
    for key in ("trace_id", "span_id", "name", "kind", "start_time_ns"):
        assert key in payload


def test_latency_is_derived_from_timestamps() -> None:
    span = _span(start_time_ns=1_000_000_000, end_time_ns=1_050_000_000)
    assert span.latency_ms == pytest.approx(50.0)


def test_end_before_start_is_rejected() -> None:
    with pytest.raises(ValueError):
        _span(start_time_ns=2_000, end_time_ns=1_000)


def test_invalid_ids_are_rejected() -> None:
    with pytest.raises(ValueError):
        _span(trace_id="not-hex")
    with pytest.raises(ValueError):
        _span(span_id="deadbeef")  # too short


def test_unknown_fields_are_forbidden() -> None:
    with pytest.raises(ValueError):
        Span(
            trace_id=new_trace_id(),
            span_id=new_span_id(),
            name="x",
            start_time_ns=1,
            surprise="boom",  # type: ignore[call-arg]
        )


def test_otlp_mapping_shape() -> None:
    span = _span(status=SpanStatus.OK, token_count=10, agent_name="planner")
    otlp = span.to_otlp()
    assert otlp["traceId"] == span.trace_id
    assert otlp["spanId"] == span.span_id
    assert otlp["kind"] == "SPAN_KIND_LLM"
    assert otlp["status"]["code"] == "STATUS_CODE_OK"
    # Semantic fields are merged into OTLP attributes as KeyValue entries.
    keys = {a["key"] for a in otlp["attributes"]}
    assert "pytracer.agent_name" in keys
    assert "pytracer.token_count" in keys


@given(
    name=st.text(min_size=1, max_size=50),
    tokens=st.integers(min_value=0, max_value=10**6),
    cost=st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
)
def test_property_serialization_never_drops_data(name: str, tokens: int, cost: float) -> None:
    """Property-based fuzz: any span survives a JSON round trip unchanged."""
    span = _span(name=name, token_count=tokens, cost_usd=cost)
    assert Span.from_json(span.to_json()) == span
