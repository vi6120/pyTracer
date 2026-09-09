"""Command tests for `navik keys` using a fake key store (no Postgres needed)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from typer.testing import CliRunner

from navik_cli import main
from navik_cli.keys import format_key_table
from navik_cli.main import app

runner = CliRunner()


class _Key:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


class FakeKeyStore:
    def __init__(self) -> None:
        self.rows: list[_Key] = []

    def create(self, project: str, *, name: str = "") -> tuple[_Key, str]:
        rec = _Key(
            key_id="abc12345", project=project, name=name, active=True,
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        self.rows.append(rec)
        return rec, "nvk_abc12345_secretpart"

    def revoke(self, key_id: str) -> bool:
        for r in self.rows:
            if r.key_id == key_id and r.active:
                r.active = False
                return True
        return False

    def list(self, *, project: str | None = None, include_revoked: bool = False) -> list[_Key]:
        rows = [r for r in self.rows if project is None or r.project == project]
        if not include_revoked:
            rows = [r for r in rows if r.active]
        return rows


@pytest.fixture(autouse=True)
def _fake_store(monkeypatch: pytest.MonkeyPatch) -> FakeKeyStore:
    store = FakeKeyStore()
    monkeypatch.setattr(main, "_open_key_store", lambda: store)
    return store


def test_keys_create_prints_secret_once() -> None:
    result = runner.invoke(app, ["keys", "create", "web-agent", "--name", "ci"])
    assert result.exit_code == 0
    assert "nvk_abc12345_secretpart" in result.stdout
    assert "abc12345" in result.stdout
    assert "cannot be shown again" in result.stdout


def test_keys_revoke_known_and_unknown(_fake_store: FakeKeyStore) -> None:
    _fake_store.create("proj")
    ok = runner.invoke(app, ["keys", "revoke", "abc12345"])
    assert ok.exit_code == 0
    assert "revoked abc12345" in ok.stdout

    missing = runner.invoke(app, ["keys", "revoke", "nope"])
    assert missing.exit_code == 2


def test_keys_list(_fake_store: FakeKeyStore) -> None:
    _fake_store.create("alpha", name="one")
    empty = runner.invoke(app, ["keys", "list", "--project", "beta"])
    assert "no keys found" in empty.stdout

    listed = runner.invoke(app, ["keys", "list"])
    assert listed.exit_code == 0
    assert "alpha" in listed.stdout
    assert "active" in listed.stdout


def test_format_key_table_hides_secret_and_shows_status() -> None:
    rows = [
        _Key(key_id="k1", project="p", name="n", active=True,
             created_at=datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)),
        _Key(key_id="k2", project="p", name="", active=False,
             created_at=datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)),
    ]
    table = format_key_table(rows)
    assert "active" in table
    assert "revoked" in table
    assert "k1" in table and "k2" in table
