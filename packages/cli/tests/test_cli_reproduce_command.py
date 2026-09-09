"""Command test for `pytracer reproduce` (runner orchestration is faked)."""

from __future__ import annotations

from typing import Any

import pytest
from pytracer_replay import Outcome
from typer.testing import CliRunner

from pytracer_cli.main import app

runner = CliRunner()


def _patch(monkeypatch: pytest.MonkeyPatch, result: Any) -> None:
    # The command imports run_reproduction from pytracer_cli.reproduce at call time,
    # so patching it there controls the (otherwise git+docker) reproduction.
    def fake(runner: Any, **kwargs: Any) -> Any:
        if isinstance(result, Exception):
            raise result
        return (None, None, result)

    monkeypatch.setattr("pytracer_cli.reproduce.run_reproduction", fake)


def test_all_fixed_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, {"s": Outcome.FIXED})
    result = runner.invoke(app, ["reproduce", "--branch", "feature", "--isolation", "local"])
    assert result.exit_code == 0
    assert "fixed" in result.stdout


def test_diverged_exits_one(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, {"s": Outcome.DIVERGED})
    result = runner.invoke(app, ["reproduce", "--branch", "feature", "--isolation", "local"])
    assert result.exit_code == 1
    assert "diverged" in result.stdout


def test_run_error_exits_two(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, RuntimeError("dependency install failed"))
    result = runner.invoke(app, ["reproduce", "--branch", "feature", "--isolation", "local"])
    assert result.exit_code == 2
