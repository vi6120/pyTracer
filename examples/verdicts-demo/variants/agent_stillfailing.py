"""Analyst agent, STILL-FAILING variant for the three-verdict demo.

This branch only rewords the headline and leaves the ``IndexError`` in place, so
it fails the same way as the baseline (same failure fingerprint). Replaying the
captured failure here yields the verdict **still-failing**: the change was a
no-op for the bug. A plain pass/fail tool cannot tell this apart from a branch
that broke things a new way.
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
    """Reworded, but still indexes a fourth source that is not there."""
    primary = sources[3]  # still IndexError: the real bug was never addressed
    return f"According to {primary}: {summary}"


@pytracer.op(kind=SpanKind.AGENT, agent_name="analyst")
def run(task: str) -> str:
    """The agent entrypoint: search, summarize, then headline the answer."""
    sources = search(task)
    summary = summarize(sources)
    return _headline(summary, sources)
