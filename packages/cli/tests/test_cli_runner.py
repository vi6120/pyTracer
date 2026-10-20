"""Runner: recorded trace -> replay -> assertions -> scenario pass/fail."""

from __future__ import annotations

import sys
from pathlib import Path

import cli_sample_agent
import pytest
from cli_testkit import record_trace, write_collection

from pytracer_cli import run_path
from pytracer_cli.runner import load_agent


def _make(tmp_path: Path, assertions: list[dict[str, object]]) -> Path:
    record_trace(cli_sample_agent.agent, {"task": "find cats"}, tmp_path / "trace.jsonl")
    return write_collection(
        tmp_path,
        [
            {
                "name": "happy-path",
                "agent": "cli_sample_agent:agent",
                "entry": {"task": "find cats"},
                "trace": "trace.jsonl",
                "mode": "full",
                "assertions": assertions,
            }
        ],
    )


def test_passing_collection(tmp_path: Path) -> None:
    collection = _make(tmp_path, [{"type": "no_errors"}, {"type": "contains", "needle": "summary"}])
    result = run_path(collection)
    assert result.passed
    assert result.counts == {"total": 1, "passed": 1, "failed": 0}
    assert result.scenarios[0].fingerprint is None  # a passing run has no fingerprint


def test_failing_assertion(tmp_path: Path) -> None:
    collection = _make(tmp_path, [{"type": "contains", "needle": "not-in-output"}])
    result = run_path(collection)
    assert not result.passed
    assert result.counts["failed"] == 1
    assert not result.scenarios[0].assertions[0].passed


def test_bad_agent_path_is_config_error(tmp_path: Path) -> None:
    record_trace(cli_sample_agent.agent, {"task": "x"}, tmp_path / "trace.jsonl")
    collection = write_collection(
        tmp_path,
        [
            {
                "name": "broken",
                "agent": "nope_module:agent",
                "trace": "trace.jsonl",
                "assertions": [],
            }
        ],
    )
    result = run_path(collection)
    assert not result.passed
    assert result.scenarios[0].config_error is not None
    assert "could not import" in result.scenarios[0].config_error


def test_missing_trace_is_config_error(tmp_path: Path) -> None:
    collection = write_collection(
        tmp_path,
        [{"name": "s", "agent": "cli_sample_agent:agent", "trace": "absent.jsonl", "assertions": []}],
    )
    result = run_path(collection)
    assert not result.passed
    assert "trace file not found" in (result.scenarios[0].config_error or "")


def test_load_agent_imports_from_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # An agent module living in the project dir is importable when run from there.
    (tmp_path / "myagent.py").write_text("def run():\n    return 'ok'\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, "myagent", raising=False)
    agent = load_agent("myagent:run")
    assert agent() == "ok"


def test_deterministic_replay_does_not_call_live_tools(tmp_path: Path) -> None:
    # Recording captures real outputs; replay must reproduce them without new live effects.
    collection = _make(tmp_path, [{"type": "contains", "needle": "result for find cats"}])
    result = run_path(collection)
    assert result.passed
    assert result.scenarios[0].replay.return_value == "summary: {'results': ['result for find cats']}"


def test_full_mode_fails_on_unrecorded_call_instead_of_going_live(tmp_path: Path) -> None:
    # Record with one input, replay with a drifted input so the recorded call no
    # longer matches. In full mode this must FAIL with a clear reason, not silently
    # run the tool/model live (which would make the result depend on the machine).
    record_trace(cli_sample_agent.agent, {"task": "find cats"}, tmp_path / "trace.jsonl")
    collection = write_collection(
        tmp_path,
        [
            {
                "name": "prompt-drift",
                "agent": "cli_sample_agent:agent",
                "entry": {"task": "find dogs"},
                "trace": "trace.jsonl",
                "mode": "full",
                "assertions": [{"type": "no_errors"}],
            }
        ],
    )
    result = run_path(collection)
    assert not result.passed
    assert result.scenarios[0].replay.error_type == "MissingMockError"
    assert "no recording" in (result.scenarios[0].replay.error_message or "")


def test_scenario_strict_false_allows_live_fallback(tmp_path: Path) -> None:
    # With strict and sandbox turned off in the scenario, an unrecorded call runs
    # live instead of failing (the sample tool is pure Python, so it just runs).
    record_trace(cli_sample_agent.agent, {"task": "find cats"}, tmp_path / "trace.jsonl")
    collection = write_collection(
        tmp_path,
        [
            {
                "name": "prompt-drift-live",
                "agent": "cli_sample_agent:agent",
                "entry": {"task": "find dogs"},
                "trace": "trace.jsonl",
                "mode": "full",
                "strict": False,
                "sandbox": False,
                "assertions": [{"type": "no_errors"}],
            }
        ],
    )
    assert run_path(collection).passed


def test_run_path_strict_override_beats_full_mode_default(tmp_path: Path) -> None:
    # The CLI override (--no-strict/--no-sandbox) wins over the full-mode default.
    record_trace(cli_sample_agent.agent, {"task": "find cats"}, tmp_path / "trace.jsonl")
    collection = write_collection(
        tmp_path,
        [
            {
                "name": "drift",
                "agent": "cli_sample_agent:agent",
                "entry": {"task": "find dogs"},
                "trace": "trace.jsonl",
                "mode": "full",
                "assertions": [{"type": "no_errors"}],
            }
        ],
    )
    assert not run_path(collection).passed  # full-mode default is strict
    assert run_path(collection, strict=False, sandbox=False).passed  # override allows live


def test_structural_assertions_pass_from_a_collection(tmp_path: Path) -> None:
    # route / tools_used / step_count wired through the collection schema.
    collection = _make(
        tmp_path,
        [
            {"type": "route", "expected": ["researcher"]},
            {"type": "tools_used", "expected": ["search"]},
            {"type": "step_count", "kind": "tool", "expected": 1},
        ],
    )
    assert run_path(collection).passed


def test_route_assertion_catches_an_unexpected_route(tmp_path: Path) -> None:
    # A route expectation that does not match the run fails, catching a path
    # regression that an output-only assertion would miss.
    collection = _make(tmp_path, [{"type": "route", "expected": ["researcher", "verifier"]}])
    result = run_path(collection)
    assert not result.passed
    assert not result.scenarios[0].assertions[0].passed
