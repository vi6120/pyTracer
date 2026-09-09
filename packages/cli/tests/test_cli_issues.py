"""Command tests for `pytracer issues sync` with a stubbed GitHub client."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from pytracer_cli import main
from pytracer_cli.github import SyncResult
from pytracer_cli.main import app

runner = CliRunner()


def _results(tmp_path: Path, scenarios: list[dict[str, Any]]) -> Path:
    p = tmp_path / "results.json"
    p.write_text(json.dumps({"collection": "suite", "scenarios": scenarios}), encoding="utf-8")
    return p


_FAILED = {
    "name": "broken",
    "passed": False,
    "status": "error",
    "error_type": "KeyError",
    "fingerprint": "fp1",
    "assertions": [],
}
_PASSED = {"name": "ok", "passed": True, "status": "ok", "assertions": []}


@pytest.fixture(autouse=True)
def _clear_github_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # So tests behave the same locally and inside GitHub Actions (which sets these).
    for var in ("GITHUB_REPOSITORY", "GITHUB_TOKEN", "PYTRACER_GITHUB_TOKEN",
                "GITHUB_REF_NAME", "GITHUB_SHA"):
        monkeypatch.delenv(var, raising=False)


def test_sync_only_documents_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results = _results(tmp_path, [_FAILED, _PASSED])
    calls: list[str] = []
    monkeypatch.setattr(main, "GitHubClient", lambda *a, **k: object())

    def fake_sync(client: Any, *, fingerprint: str, title: str, body: str, labels: Any) -> SyncResult:
        calls.append(fingerprint)
        return SyncResult("created", 7, "https://github.example/7")

    monkeypatch.setattr(main, "sync_issue", fake_sync)
    res = runner.invoke(app, ["issues", "sync", str(results), "--repo", "o/n", "--token", "t"])
    assert res.exit_code == 0
    assert "created #7" in res.stdout
    assert calls == ["fp1"]  # the passing scenario is skipped


def test_sync_no_failures_is_a_noop(tmp_path: Path) -> None:
    results = _results(tmp_path, [_PASSED])
    res = runner.invoke(app, ["issues", "sync", str(results)])
    assert res.exit_code == 0
    assert "no failures" in res.stdout


def test_sync_dry_run_calls_no_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results = _results(tmp_path, [_FAILED])

    def explode(*a: Any, **k: Any) -> None:
        raise AssertionError("dry run must not construct a client")

    monkeypatch.setattr(main, "GitHubClient", explode)
    res = runner.invoke(app, ["issues", "sync", str(results), "--dry-run"])
    assert res.exit_code == 0
    assert "would sync" in res.stdout and "fp1" in res.stdout


def test_sync_requires_repo_and_token(tmp_path: Path) -> None:
    results = _results(tmp_path, [_FAILED])
    res = runner.invoke(app, ["issues", "sync", str(results)])
    assert res.exit_code == 2  # no repo/token available -> refuses before any API call


def test_sync_reads_repo_and_token_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = _results(tmp_path, [_FAILED])
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/n")
    monkeypatch.setenv("GITHUB_TOKEN", "envtoken")
    seen: dict[str, Any] = {}
    monkeypatch.setattr(main, "GitHubClient", lambda token, repo, **k: seen.update(repo=repo, token=token))

    def fake_sync(client: Any, **k: Any) -> SyncResult:
        return SyncResult("updated", 3, "https://github.example/3")

    monkeypatch.setattr(main, "sync_issue", fake_sync)
    res = runner.invoke(app, ["issues", "sync", str(results)])
    assert res.exit_code == 0
    assert seen == {"repo": "o/n", "token": "envtoken"}
    assert "updated #3" in res.stdout
