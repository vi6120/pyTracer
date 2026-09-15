"""The analyst agent for the pyTracer cross-branch reproduction demo.

A tiny instrumented agent that searches for sources and writes a one-line
answer. ``search`` and ``summarize`` are ``@op``-instrumented, so in replay
their recorded outputs are injected from the frozen trace and the run stays
deterministic. The behavior the demo reproduces lives in ``_headline``, which
is plain glue code (not an ``@op``): it runs live during replay, so a one-line
fix to it on another branch actually flips the outcome while the mocks stay
identical.

run_demo.sh copies analyst_buggy.py onto the baseline branch and
analyst_fixed.py onto the fix branch; the two differ only inside ``_headline``.
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


def _headline(summary: str, n_sources: int) -> str:
    """Prefix the answer with how many sources backed it (buggy: off-by-one)."""
    return f"Reviewed {n_sources - 1} sources. {summary}"


@pytracer.op(kind=SpanKind.AGENT, agent_name="analyst")
def run(task: str) -> str:
    """The agent entrypoint: search, summarize, then headline the answer."""
    sources = search(task)
    summary = summarize(sources)
    return _headline(summary, len(sources))
