"""Hermetic tests for trace formatting and collection scaffolding."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
from navik_sdk import new_span_id, new_trace_id
from navik_sdk.schema import Resource, Span, SpanKind, SpanStatus

from navik_cli.traces import format_summary_table, format_trace_tree, parse_entry, record_trace

RES = Resource(service_name="t", branch="main", commit="abc123")


def _span(trace_id: str, name: str, kind: SpanKind, *, parent: str | None = None,
          span_id: str | None = None, status: SpanStatus = SpanStatus.OK, t: int = 0) -> Span:
    return Span(
        trace_id=trace_id, span_id=span_id or new_span_id(), parent_span_id=parent,
        name=name, kind=kind, tool_name=name if kind is SpanKind.TOOL else None,
        status=status, start_time_ns=t, end_time_ns=t + 1_000_000, resource=RES,
    )


class _Summary:
    def __init__(self, **kw: object) -> None:
        self.__dict__.update(kw)


def test_parse_entry_json_and_string() -> None:
    result = parse_entry(["task=summarize the doc", "n=3", "flag=true"])
    assert result == {"task": "summarize the doc", "n": 3, "flag": True}


def test_parse_entry_rejects_bad_item() -> None:
    with pytest.raises(ValueError):
        parse_entry(["noequals"])


def test_format_summary_table() -> None:
    rows = [
        _Summary(trace_id="abc123def456", root_agent="planner", branch="main",
                 span_count=5, failed=True, started_at=datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)),
    ]
    table = format_summary_table(rows)
    assert "planner" in table
    assert "FAIL" in table
    assert "abc123def456"[:12] in table


def test_format_trace_tree_nesting() -> None:
    tid = new_trace_id()
    root = _span(tid, "agent", SpanKind.AGENT, span_id=new_span_id(), t=0)
    child = _span(tid, "search", SpanKind.TOOL, parent=root.span_id, t=1)
    tree = format_trace_tree([child, root])
    lines = tree.splitlines()
    assert lines[0].startswith("[ ] agent")
    assert lines[1].startswith("  [ ] tool")  # indented child


def test_record_trace_creates_collection(tmp_path: Path) -> None:
    tid = new_trace_id()
    spans = [_span(tid, "agent", SpanKind.AGENT), _span(tid, "search", SpanKind.TOOL)]
    jsonl, collection = record_trace(
        spans, agent="myapp:run", entry={"task": "x"}, out_dir=tmp_path, name="happy"
    )
    assert jsonl.exists()
    assert len(jsonl.read_text().splitlines()) == 2
    data = yaml.safe_load(collection.read_text())
    assert data["scenarios"][0]["name"] == "happy"
    assert data["scenarios"][0]["agent"] == "myapp:run"
    assert data["scenarios"][0]["trace"].startswith("traces/")
    assert data["scenarios"][0]["assertions"] == [{"type": "no_errors"}]


def test_record_trace_appends_to_existing(tmp_path: Path) -> None:
    tid1, tid2 = new_trace_id(), new_trace_id()
    record_trace([_span(tid1, "a", SpanKind.AGENT)], agent="m:a", entry={}, out_dir=tmp_path, name="one")
    _, collection = record_trace(
        [_span(tid2, "b", SpanKind.AGENT)], agent="m:a", entry={}, out_dir=tmp_path, name="two"
    )
    data = yaml.safe_load(collection.read_text())
    assert [s["name"] for s in data["scenarios"]] == ["one", "two"]


def test_record_empty_trace_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        record_trace([], agent="m:a", entry={}, out_dir=tmp_path)
