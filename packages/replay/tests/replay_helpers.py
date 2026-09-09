"""Shared test helpers: span and result builders for replay tests."""

from __future__ import annotations

from pytracer_sdk import new_span_id, new_trace_id
from pytracer_sdk.schema import Resource, Span, SpanKind, SpanStatus

from pytracer_replay import ReplayMode, ReplayResult

RES = Resource(service_name="test", branch="main", commit="abc123")


def span(
    name: str,
    *,
    tool: str | None = None,
    kind: SpanKind = SpanKind.TOOL,
    status: SpanStatus = SpanStatus.OK,
    error_type: str | None = None,
    output: object = None,
    t: int = 0,
) -> Span:
    return Span(
        trace_id=new_trace_id(),
        span_id=new_span_id(),
        name=name,
        kind=kind,
        tool_name=tool,
        status=status,
        error_type=error_type,
        output=output,
        start_time_ns=t,
        end_time_ns=t + 1,
        resource=RES,
    )


def failing_result(sequence: list[str], error_type: str) -> ReplayResult:
    """A failing result whose last step errored with ``error_type``."""
    spans = []
    for i, name in enumerate(sequence):
        is_last = i == len(sequence) - 1
        spans.append(
            span(
                name,
                tool=name,
                status=SpanStatus.ERROR if is_last else SpanStatus.OK,
                error_type=error_type if is_last else None,
                t=i,
            )
        )
    return ReplayResult(
        return_value=None,
        status=SpanStatus.ERROR,
        mode=ReplayMode.FULL,
        spans=spans,
        error_type=error_type,
    )


def passing_result(sequence: list[str], return_value: object = "ok") -> ReplayResult:
    spans = [span(name, tool=name, t=i) for i, name in enumerate(sequence)]
    return ReplayResult(
        return_value=return_value,
        status=SpanStatus.OK,
        mode=ReplayMode.FULL,
        spans=spans,
    )
