"""The ``pytracer`` command: exit codes and output artifacts, via Typer's runner."""

from __future__ import annotations

from pathlib import Path

import cli_sample_agent
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
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.0.1" in result.stdout
