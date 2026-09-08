"""Render collection results for humans, machines, and CI.

- ``to_markdown``  → a PR-comment summary (pass/fail counts + per-scenario table)
- ``to_json``      → a machine-readable dict
- ``to_junit``     → JUnit XML for CI test reporters
- ``issue_body``   → an auto-documentation issue for a failure, carrying a
                     fingerprint marker so a repeat failure updates one issue
                     instead of opening duplicates (guide §7).
"""

from __future__ import annotations

import json
from typing import Any
from xml.sax.saxutils import escape

from .runner import CollectionResult, ScenarioResult

_PASS = "✅"
_FAIL = "❌"

# Hidden marker embedded in an auto-doc issue body; the fingerprint is the
# dedup key CI matches on to decide "update existing" vs "open new".
FINGERPRINT_MARKER = "<!-- navik-fingerprint: {fp} -->"


def to_json(result: CollectionResult) -> dict[str, Any]:
    return {
        "collection": result.name,
        "counts": result.counts,
        "passed": result.passed,
        "scenarios": [
            {
                "name": s.name,
                "passed": s.passed,
                "status": s.replay.status.value,
                "error_type": s.replay.error_type,
                "config_error": s.config_error,
                "fingerprint": s.fingerprint,
                "assertions": [
                    {"name": a.name, "passed": a.passed, "detail": a.detail} for a in s.assertions
                ],
            }
            for s in result.scenarios
        ],
    }


def _scenario_reason(s: ScenarioResult) -> str:
    if s.config_error:
        return f"config error: {s.config_error}"
    if s.replay.failed:
        return f"replay error: {s.replay.error_type}"
    failed = [a for a in s.assertions if not a.passed]
    if failed:
        return "; ".join(f"{a.name} ({a.detail})" if a.detail else a.name for a in failed)
    return ""


def to_markdown(result: CollectionResult) -> str:
    c = result.counts
    header = _PASS if result.passed else _FAIL
    lines = [
        f"## {header} Navik agent tests — `{result.name}`",
        "",
        f"**{c['passed']}/{c['total']} scenarios passed**"
        + (f", {c['failed']} failed" if c["failed"] else ""),
        "",
        "| Scenario | Result | Details |",
        "| --- | --- | --- |",
    ]
    for s in result.scenarios:
        mark = _PASS if s.passed else _FAIL
        detail = _scenario_reason(s).replace("|", "\\|") or "—"
        lines.append(f"| `{s.name}` | {mark} | {detail} |")
    return "\n".join(lines) + "\n"


def to_junit(result: CollectionResult) -> str:
    c = result.counts
    cases: list[str] = []
    for s in result.scenarios:
        classname = escape(result.name)
        name = escape(s.name)
        if s.passed:
            cases.append(f'    <testcase classname="{classname}" name="{name}" />')
        else:
            reason = escape(_scenario_reason(s) or "failed")
            cases.append(
                f'    <testcase classname="{classname}" name="{name}">\n'
                f'      <failure message="{reason}"></failure>\n'
                f"    </testcase>"
            )
    body = "\n".join(cases)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<testsuite name="{escape(result.name)}" tests="{c["total"]}" failures="{c["failed"]}">\n'
        f"{body}\n"
        "</testsuite>\n"
    )


def issue_body(
    scenario: ScenarioResult,
    *,
    branch: str | None = None,
    commit: str | None = None,
    trace_id: str | None = None,
) -> str:
    """Auto-documentation body for a failed scenario, with a dedup marker."""
    reason = _scenario_reason(scenario) or "unknown failure"
    lines = [
        f"### Agent test failure: `{scenario.name}`",
        "",
        f"- **Status:** {scenario.replay.status.value}",
        f"- **Reason:** {reason}",
    ]
    if scenario.replay.error_type:
        lines.append(f"- **Error type:** `{scenario.replay.error_type}`")
    if branch:
        lines.append(f"- **Branch:** `{branch}`")
    if commit:
        lines.append(f"- **Commit:** `{commit}`")
    if trace_id:
        lines.append(f"- **Trace:** `{trace_id}`")
    if scenario.fingerprint:
        lines.append(f"- **Fingerprint:** `{scenario.fingerprint}`")
    lines.append("")
    lines.append(FINGERPRINT_MARKER.format(fp=scenario.fingerprint or "none"))
    return "\n".join(lines) + "\n"


def dumps_json(result: CollectionResult) -> str:
    return json.dumps(to_json(result), indent=2, default=str)
