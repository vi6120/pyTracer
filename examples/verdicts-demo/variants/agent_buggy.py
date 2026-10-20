"""Analyst agent, baseline (buggy) variant for the three-verdict demo.

``search`` and ``summarize`` are ``@op``-instrumented, so their outputs are
injected from the frozen trace during replay and the run stays deterministic.
The behavior under test lives in ``_headline``, which is plain glue code (not an
``@op``), so it runs live during replay: a one-line change on another branch
actually changes the outcome while the mocked model and tool calls stay frozen.

Baseline bug: ``_headline`` indexes a fourth source that does not exist, so the
agent raises ``IndexError``. The fix / still-failing / diverged branches each
change only ``_headline``.
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
    """Prefix the answer with a primary source (buggy: there is no fourth one)."""
    primary = sources[3]  # IndexError: search returned only 3 sources
    return f"Primary source {primary}. {summary}"


@pytracer.op(kind=SpanKind.AGENT, agent_name="analyst")
def run(task: str) -> str:
    """The agent entrypoint: search, summarize, then headline the answer."""
    sources = search(task)
    summary = summarize(sources)
    return _headline(summary, sources)
