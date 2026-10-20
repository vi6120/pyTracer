"""Analyst agent, DIVERGED variant for the three-verdict demo.

This branch removes the original ``IndexError`` but introduces a different
failure: it misreads the prose summary as a number, raising ``ValueError``. So
it fails, but differently than the baseline (a new failure fingerprint).
Replaying the captured failure here yields the verdict **diverged**: the branch
did not fix the bug, it replaced it with a new one. This is the case a plain
pass/fail tool reports identically to still-failing.
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
    """No longer indexes out of range, but misreads the summary as a count."""
    count = int(summary)  # ValueError: the summary is prose, not a number
    return f"Reviewed {count} sources. {summary}"


@pytracer.op(kind=SpanKind.AGENT, agent_name="analyst")
def run(task: str) -> str:
    """The agent entrypoint: search, summarize, then headline the answer."""
    sources = search(task)
    summary = summarize(sources)
    return _headline(summary, sources)
