"""The pytest plugin: pyTracer collection files run as native pytest tests."""

from __future__ import annotations

import importlib
import sys

import pytest
import yaml
from cli_testkit import record_trace

# Load the plugin once for the sub-run, regardless of whether pytracer-cli is
# installed. When it is, the `pytest11` entry point auto-registers it as
# `pytracer`; block that so loading the module via `-p` is the only registration
# (pytest refuses the same module under two names). When it is not installed,
# `no:pytracer` is a harmless no-op and `-p` is what loads it.
PLUGIN_ARGS = ("-p", "no:pytracer", "-p", "pytracer_cli.pytest_plugin")

# A tiny instrumented agent written into the sandbox so the collection can name it.
AGENT_SRC = """
import pytracer_sdk as pytracer
from pytracer_sdk import SpanKind

@pytracer.op(kind=SpanKind.TOOL, tool_name="search")
def search(query):
    return {"results": [f"result for {query}"]}

@pytracer.op(kind=SpanKind.LLM)
def summarize(text):
    return f"summary: {text}"

@pytracer.op(kind=SpanKind.AGENT, agent_name="researcher")
def agent(task):
    return summarize(str(search(task)))
"""


def _setup(pytester: pytest.Pytester, assertions: list[dict[str, object]]) -> None:
    pytester.makepyfile(sample_agent=AGENT_SRC)
    sys.path.insert(0, str(pytester.path))
    sample_agent = importlib.import_module("sample_agent")
    record_trace(sample_agent.agent, {"task": "find cats"}, pytester.path / "trace.jsonl")
    (pytester.path / "agents.pytracer.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "kit",
                "scenarios": [
                    {
                        "name": "happy-path",
                        "agent": "sample_agent:agent",
                        "entry": {"task": "find cats"},
                        "trace": "trace.jsonl",
                        "mode": "full",
                        "assertions": assertions,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_plugin_runs_a_passing_scenario(pytester: pytest.Pytester) -> None:
    _setup(pytester, [{"type": "no_errors"}, {"type": "contains", "needle": "summary"}])
    result = pytester.runpytest("-v", *PLUGIN_ARGS)
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*happy-path*PASSED*"])  # each scenario is its own item


def test_plugin_reports_a_failing_scenario(pytester: pytest.Pytester) -> None:
    _setup(pytester, [{"type": "contains", "needle": "not-in-the-output"}])
    result = pytester.runpytest(*PLUGIN_ARGS)
    result.assert_outcomes(failed=1)


def test_plugin_ignores_non_matching_yaml(pytester: pytest.Pytester) -> None:
    pytester.makefile(".yaml", other="name: not-a-collection\n")
    result = pytester.runpytest(*PLUGIN_ARGS)
    result.assert_outcomes()  # nothing collected from a plain .yaml file
