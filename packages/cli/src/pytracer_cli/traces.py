"""Browse captured traces and turn one into a test collection.

These back the ``pytracer traces`` and ``pytracer record`` commands. The formatting
and scaffolding helpers are pure (they take spans/summaries), so they are tested
without a running trace store; the commands wire them to the store.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from pytracer_sdk.schema import Span


def parse_entry(items: list[str] | None) -> dict[str, Any]:
    """Parse repeated ``key=value`` options into a dict (values JSON-decoded)."""
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--entry must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def format_summary_table(rows: list[Any]) -> str:
    """Render trace summaries as a fixed-width table."""
    header = f"{'TRACE':<12} {'AGENT':<16} {'SPANS':>5} {'STATUS':<6} {'BRANCH':<12} STARTED"
    lines = [header]
    for r in rows:
        status = "FAIL" if r.failed else "ok"
        started = (
            r.started_at.strftime("%Y-%m-%d %H:%M")
            if hasattr(r.started_at, "strftime")
            else str(r.started_at)
        )
        lines.append(
            f"{r.trace_id[:12]:<12} {(r.root_agent or '-')[:16]:<16} "
            f"{r.span_count:>5} {status:<6} {(r.branch or '-')[:12]:<12} {started}"
        )
    return "\n".join(lines)


def format_trace_tree(spans: list[Span]) -> str:
    """Render a trace's spans as an indented tree, ordered by start time."""
    by_id = {s.span_id: s for s in spans}
    children: dict[str, list[Span]] = defaultdict(list)
    roots: list[Span] = []
    for span in sorted(spans, key=lambda s: s.start_time_ns):
        if span.parent_span_id and span.parent_span_id in by_id:
            children[span.parent_span_id].append(span)
        else:
            roots.append(span)

    lines: list[str] = []

    def render(span: Span, depth: int) -> None:
        mark = "x" if span.status.value == "error" else " "
        latency = f"{span.latency_ms:.0f}ms" if span.latency_ms is not None else "-"
        label = span.tool_name or span.agent_name or span.name
        lines.append(f"{'  ' * depth}[{mark}] {span.kind.value:<8} {label} ({latency})")
        for child in children[span.span_id]:
            render(child, depth + 1)

    for root in roots:
        render(root, 0)
    return "\n".join(lines)


def record_trace(
    spans: list[Span],
    *,
    agent: str,
    entry: dict[str, Any],
    out_dir: str | Path,
    name: str | None = None,
    mode: str = "full",
    assertions: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path]:
    """Write a trace's spans to JSONL and scaffold a collection scenario.

    Creates ``<out>/traces/<trace>.jsonl`` and creates or appends
    ``<out>/collection.yaml`` with a scenario that replays the trace against the
    given ``agent`` entrypoint. Returns (jsonl_path, collection_path).
    """
    if not spans:
        raise ValueError("cannot record an empty trace")
    trace_id = spans[0].trace_id
    scenario_name = name or f"trace-{trace_id[:8]}"

    out = Path(out_dir)
    traces_dir = out / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    jsonl = traces_dir / f"{trace_id[:16]}.jsonl"
    jsonl.write_text("\n".join(s.to_json() for s in spans) + "\n", encoding="utf-8")

    scenario = {
        "name": scenario_name,
        "agent": agent,
        "entry": entry,
        "trace": f"traces/{jsonl.name}",
        "mode": mode,
        "assertions": assertions or [{"type": "no_errors"}],
    }

    collection = out / "collection.yaml"
    if collection.exists():
        data = yaml.safe_load(collection.read_text(encoding="utf-8")) or {}
        data.setdefault("scenarios", []).append(scenario)
    else:
        data = {"name": "captured-suite", "scenarios": [scenario]}
    collection.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return jsonl, collection
