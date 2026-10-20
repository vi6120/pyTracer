"""Analyst agent, FIXED variant for the three-verdict demo.

Only ``_headline`` differs from the baseline: it counts the sources instead of
indexing a missing one, so the agent returns a correct answer and the scenario
passes. Replaying the captured failure on this branch yields the verdict
**fixed**.
"""

from __future__ import annotations

import pytracer_sdk as pytracer
from pytracer_sdk import SpanKind


@pytracer.op(kind=SpanKind.TOOL, tool_name="search")
def search(query: str) -> list[str]:
    """Pretend to hit a search API and return sources (no network)."""
    return [f"Source on {query} #1", f"Source on {query} #2", f"Source on {query} #3"]


@pytracer.op(kind=SpanKind.LLM)
def summarize(sources: list[str]) -> str:
    """Pretend to call a model to summarize the sources (no API key)."""
    return "Rising temperatures are driven mainly by greenhouse gas emissions."


def _headline(summary: str, sources: list[str]) -> str:
    """Prefix the answer with how many sources backed it (counts them all)."""
    return f"Reviewed {len(sources)} sources. {summary}"


@pytracer.op(kind=SpanKind.AGENT, agent_name="analyst")
def run(task: str) -> str:
    """The agent entrypoint: search, summarize, then headline the answer."""
    sources = search(task)
    summary = summarize(sources)
    return _headline(summary, sources)
