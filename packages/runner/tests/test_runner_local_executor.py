"""LocalExecutor: real subprocess execution, artifacts, exit codes, timeout."""

from __future__ import annotations

from pathlib import Path

from pytracer_runner import LocalExecutor


def test_runs_command_and_reads_artifact(tmp_path: Path) -> None:
    executor = LocalExecutor()
    result = executor.run(tmp_path, "echo hello && printf DATA > out.txt", artifacts=["out.txt"])
    assert result.ok
    assert "hello" in result.stdout
    assert result.artifacts["out.txt"] == "DATA"


def test_nonzero_exit_is_reported(tmp_path: Path) -> None:
    result = LocalExecutor().run(tmp_path, "exit 3")
    assert result.exit_code == 3
    assert not result.ok


def test_timeout_is_reported(tmp_path: Path) -> None:
    result = LocalExecutor(timeout=0.3).run(tmp_path, "sleep 5")
    assert result.timed_out
    assert result.exit_code == 124
    assert not result.ok


def test_missing_artifact_is_absent_not_error(tmp_path: Path) -> None:
    result = LocalExecutor().run(tmp_path, "true", artifacts=["never-written.json"])
    assert result.ok
    assert "never-written.json" not in result.artifacts
