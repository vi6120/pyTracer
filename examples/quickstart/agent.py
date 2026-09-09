"""A tiny instrumented agent for the pyTracer quickstart.

A "researcher" agent that searches for documents and summarizes them. The tool
and model calls are stand-ins (no network, no API key) so the example runs
anywhere. Each function is instrumented with a single ``@op`` decorator, which
is all it takes for pyTracer to capture the run.
"""

from __future__ import annotations

import pytracer_sdk as pytracer
from pytracer_sdk import SpanKind


@pytracer.op(kind=SpanKind.TOOL, tool_name="search")
def search(query: str) -> list[str]:
    """Pretend to hit a search API and return documents."""
    return [f"Doc about {query} #1", f"Doc about {query} #2"]


@pytracer.op(kind=SpanKind.LLM)
def summarize(docs: list[str]) -> str:
    """Pretend to call a model to summarize the documents."""
    return f"Summary of {len(docs)} docs: " + "; ".join(docs)


@pytracer.op(kind=SpanKind.AGENT, agent_name="researcher")
def run(task: str) -> str:
    """The agent entrypoint: search, then summarize."""
    docs = search(task)
    return summarize(docs)
