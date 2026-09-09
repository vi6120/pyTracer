"""A tiny instrumented agent used as the target of CLI test collections."""

from __future__ import annotations

import pytracer_sdk as pytracer
from pytracer_sdk import SpanKind


@pytracer.op(kind=SpanKind.TOOL, tool_name="search")
def search(query: str) -> dict[str, list[str]]:
    # Stands in for a live tool call (a real one would hit the network).
    return {"results": [f"result for {query}"]}


@pytracer.op(kind=SpanKind.LLM)
def summarize(text: str) -> str:
    return f"summary: {text}"


@pytracer.op(kind=SpanKind.AGENT, agent_name="researcher")
def agent(task: str) -> str:
    hits = search(task)
    return summarize(str(hits))
