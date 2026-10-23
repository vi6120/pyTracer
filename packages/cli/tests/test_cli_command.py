"""The ``pytracer`` command: exit codes and output artifacts, via Typer's runner."""

from __future__ import annotations

from pathlib import Path

import cli_sample_agent
import pytest
from cli_testkit import record_trace, write_collection
from typer.testing import CliRunner

from pytracer_cli.main import app

runner = CliRunner()


def _collection(tmp_path: Path, needle: str) -> Path:
    record_trace(cli_sample_agent.agent, {"task": "find cats"}, tmp_path / "trace.jsonl")
    return write_collection(
        tmp_path,
        [
            {
                "name": "s1",
                "agent": "cli_sample_agent:agent",
                "entry": {"task": "find cats"},
                "trace": "trace.jsonl",
                "mode": "full",
                "assertions": [{"type": "contains", "needle": needle}],
            }
        ],
    )


def test_run_exit_zero_on_pass(tmp_path: Path) -> None:
    collection = _collection(tmp_path, "summary")
    result = runner.invoke(app, ["run", str(collection)])
    assert result.exit_code == 0
    assert "1/1 passed" in result.stdout


def test_run_exit_one_on_failure(tmp_path: Path) -> None:
    collection = _collection(tmp_path, "not-present")
    result = runner.invoke(app, ["run", str(collection)])
    assert result.exit_code == 1


def test_emit_notes_non_exact_matches(capsys: pytest.CaptureFixture[str]) -> None:
    # A fuzzy/positional substitution is surfaced in the run output, not left only
    # on a span attribute (R12).
    from pytracer_sdk import SpanStatus, new_span_id, new_trace_id
    from pytracer_sdk.schema import Span, SpanKind
    from pytracer_replay import ReplayMode, ReplayResult

    from pytracer_cli.main import _emit
    from pytracer_cli.runner import CollectionResult, ScenarioResult

    span = Span(
        trace_id=new_trace_id(), span_id=new_span_id(), name="search", kind=SpanKind.TOOL,
        start_time_ns=1, attributes={"pytracer.mock_match": "positional"},
    )
    replay = ReplayResult(
        return_value="x", status=SpanStatus.OK, mode=ReplayMode.FULL, spans=[span]
    )
    _emit(CollectionResult("suite", [ScenarioResult("s1", True, replay, [])]))
    assert "non-exact matches: 1×positional" in capsys.readouterr().out


def test_run_prints_the_failure_reason(tmp_path: Path) -> None:
    # A FAIL line must be followed by why it failed, not just [FAIL] (R3): here the
    # assertion that did not hold is named in the terminal output.
    collection = _collection(tmp_path, "not-present")
    result = runner.invoke(app, ["run", str(collection)])
    assert "[FAIL] s1" in result.stdout
    assert "contains" in result.stdout


def test_run_writes_artifacts(tmp_path: Path) -> None:
    collection = _collection(tmp_path, "summary")
    json_out = tmp_path / "out.json"
    junit_out = tmp_path / "out.xml"
    comment_out = tmp_path / "pr.md"
    result = runner.invoke(
        app,
        ["run", str(collection), "--json", str(json_out), "--junit", str(junit_out),
         "--comment", str(comment_out)],
    )
    assert result.exit_code == 0
    assert json_out.exists() and junit_out.exists() and comment_out.exists()
    assert "scenarios passed" in comment_out.read_text()
    assert "<testsuite" in junit_out.read_text()


def test_run_missing_file_exit_two(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 2


def test_version_command() -> None:
    from pytracer_cli import __version__

    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout
