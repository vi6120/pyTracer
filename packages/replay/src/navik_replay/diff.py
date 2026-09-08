"""Structured diff between two runs.

Emits plain JSON-serializable structures (dicts/lists) so any frontend can
render where a change in prompt, model, or code caused two runs to diverge.
"""

from __future__ import annotations

from typing import Any

from navik_sdk.schema import Span

from .engine import ReplayResult


def _op_key(span: Span) -> str:
    return span.tool_name or span.name


def diff_traces(a: list[Span], b: list[Span]) -> dict[str, Any]:
    """Align two span sequences by position and report per-step differences."""
    a_ordered = sorted(a, key=lambda s: s.start_time_ns)
    b_ordered = sorted(b, key=lambda s: s.start_time_ns)
    steps: list[dict[str, Any]] = []
    diverged_at: int | None = None

    for i in range(max(len(a_ordered), len(b_ordered))):
        sa = a_ordered[i] if i < len(a_ordered) else None
        sb = b_ordered[i] if i < len(b_ordered) else None
        name_a = _op_key(sa) if sa else None
        name_b = _op_key(sb) if sb else None
        out_a = sa.output if sa else None
        out_b = sb.output if sb else None
        status_a = sa.status.value if sa else None
        status_b = sb.status.value if sb else None
        changed = (name_a != name_b) or (out_a != out_b) or (status_a != status_b)
        if changed and diverged_at is None:
            diverged_at = i
        steps.append(
            {
                "index": i,
                "name_a": name_a,
                "name_b": name_b,
                "output_a": out_a,
                "output_b": out_b,
                "status_a": status_a,
                "status_b": status_b,
                "changed": changed,
            }
        )

    return {
        "identical": diverged_at is None and len(a_ordered) == len(b_ordered),
        "span_count": {"a": len(a_ordered), "b": len(b_ordered)},
        "diverged_at": diverged_at,
        "steps": steps,
    }


def diff_results(a: ReplayResult, b: ReplayResult) -> dict[str, Any]:
    """Diff two replay results: return value, status, and the span traces."""
    trace_diff = diff_traces(a.spans, b.spans)
    return {
        "return_value_changed": a.return_value != b.return_value,
        "return_value_a": a.return_value,
        "return_value_b": b.return_value,
        "status_a": a.status.value,
        "status_b": b.status.value,
        "status_changed": a.status != b.status,
        "trace": trace_diff,
    }
