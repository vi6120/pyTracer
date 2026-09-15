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


@pytest.fixture(autouse=True)
def _clear_github_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Behave the same locally and inside GitHub Actions (which sets these).
    for var in ("GITHUB_REPOSITORY", "GITHUB_TOKEN", "PYTRACER_GITHUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)


def test_pr_dry_run_prints_comment(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, {"s": Outcome.FIXED})
    result = runner.invoke(
        app,
        ["reproduce", "--branch", "feature", "--isolation", "local", "--pr", "5", "--pr-dry-run"],
    )
    assert result.exit_code == 0
    assert "try on this branch" in result.stdout
    assert "fixed" in result.stdout


def test_pr_posts_comment(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, {"s": Outcome.FIXED})
    posted: dict[str, Any] = {}

    def fake_post(client: Any, *, number: int, **kwargs: Any) -> str:
        posted["number"] = number
        return "https://github.example/pr/5#comment-1"

    monkeypatch.setattr("pytracer_cli.pr_comment.post_reproduction_comment", fake_post)
    result = runner.invoke(
        app,
        ["reproduce", "--branch", "feature", "--isolation", "local",
         "--pr", "5", "--pr-repo", "o/n", "--token", "t"],
    )
    assert result.exit_code == 0
    assert posted["number"] == 5
    assert "posted verdict to https://github.example/pr/5#comment-1" in result.stdout


def test_pr_without_credentials_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, {"s": Outcome.FIXED})
    result = runner.invoke(
        app, ["reproduce", "--branch", "feature", "--isolation", "local", "--pr", "5"]
    )
    assert result.exit_code == 2
    assert "needs a repo" in result.stdout or "needs a repo" in (result.stderr or "")
