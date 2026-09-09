"""Command tests for `navik traces` and `navik record` using a fake store."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml
from navik_sdk import new_span_id, new_trace_id
from navik_sdk.schema import Resource, Span, SpanKind, SpanStatus
from typer.testing import CliRunner

from navik_cli import main
from navik_cli.main import app

runner = CliRunner()
RES = Resource(service_name="t", branch="main", commit="abc123")
TRACE_ID = new_trace_id()


class _Summary:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


class FakeStore:
    def __init__(self) -> None:
        self.spans = [
            Span(trace_id=TRACE_ID, span_id=new_span_id(), name="agent", kind=SpanKind.AGENT,
                 agent_name="planner", start_time_ns=1, end_time_ns=2, resource=RES),
            Span(trace_id=TRACE_ID, span_id=new_span_id(), name="search", kind=SpanKind.TOOL,
                 tool_name="search", status=SpanStatus.ERROR, start_time_ns=3, end_time_ns=4, resource=RES),
        ]

    def list_traces(self, **kwargs: Any) -> list[_Summary]:
        return [
            _Summary(trace_id=TRACE_ID, root_agent="planner", branch="main",
                     span_count=2, failed=True, started_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
        ]

    def get_trace(self, trace_id: str, project: str | None = None) -> list[Span]:
        return self.spans if trace_id == TRACE_ID else []


@pytest.fixture(autouse=True)
def _fake_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "_open_trace_store", lambda: FakeStore())


def test_traces_list() -> None:
    result = runner.invoke(app, ["traces", "list"])
    assert result.exit_code == 0
    assert "planner" in result.stdout
    assert "FAIL" in result.stdout


def test_traces_show() -> None:
    result = runner.invoke(app, ["traces", "show", TRACE_ID])
    assert result.exit_code == 0
    assert "agent" in result.stdout
    assert "tool" in result.stdout


def test_traces_show_unknown_exits_two() -> None:
    result = runner.invoke(app, ["traces", "show", "does-not-exist"])
    assert result.exit_code == 2


def test_record_scaffolds_collection(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["record", TRACE_ID, "--agent", "myapp:run", "--entry", "task=hi", "--out", str(tmp_path)],
    )
    assert result.exit_code == 0
    collection = tmp_path / "collection.yaml"
    assert collection.exists()
    data = yaml.safe_load(collection.read_text())
    assert data["scenarios"][0]["agent"] == "myapp:run"
    assert data["scenarios"][0]["entry"] == {"task": "hi"}
    assert (tmp_path / "traces").is_dir()
