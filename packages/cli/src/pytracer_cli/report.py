"""Render collection results for humans, machines, and CI.

- ``to_markdown``  -> a PR-comment summary (pass/fail counts + per-scenario table)
- ``to_json``      -> a machine-readable dict
- ``to_junit``     -> JUnit XML for CI test reporters
- ``issue_body``   -> an auto-documentation issue for a failure, carrying a
                     fingerprint marker so a repeat failure updates one issue
                     instead of opening duplicates (guide section 7).
"""

from __future__ import annotations

import json
from typing import Any
from xml.sax.saxutils import escape

from .runner import CollectionResult, ScenarioResult

_PASS = "PASS"
_FAIL = "FAIL"

# Hidden marker embedded in an auto-doc issue body; the fingerprint is the
# dedup key CI matches on to decide "update existing" vs "open new".
FINGERPRINT_MARKER = "<!-- pytracer-fingerprint: {fp} -->"


def to_json(result: CollectionResult) -> dict[str, Any]:
    """Machine-readable result: counts, pass flag, and per-scenario detail."""
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
    """Render a PR-comment summary: header, counts, and a per-scenario table."""
    c = result.counts
    status = _PASS if result.passed else _FAIL
    lines = [
        f"## pyTracer agent tests: `{result.name}` ({status})",
        "",
        f"**{c['passed']}/{c['total']} scenarios passed**"
        + (f", {c['failed']} failed" if c["failed"] else ""),
        "",
        "| Scenario | Result | Details |",
        "| --- | --- | --- |",
    ]
    for s in result.scenarios:
        mark = _PASS if s.passed else _FAIL
        detail = _scenario_reason(s).replace("|", "\\|") or "-"
        lines.append(f"| `{s.name}` | {mark} | {detail} |")
    return "\n".join(lines) + "\n"


def to_junit(result: CollectionResult) -> str:
    """Render JUnit XML so CI test reporters can surface scenario results."""
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


def issue_title(name: str) -> str:
    """Title for an auto-documentation issue about a failed scenario."""
    return f"Agent test failure: {name}"


def _render_issue(
    *,
    name: str,
    status: str,
    reason: str,
    error_type: str | None,
    fingerprint: str | None,
    branch: str | None,
    commit: str | None,
    trace_id: str | None,
) -> str:
    """Shared renderer for the auto-doc issue body (dedup marker last)."""
    lines = [
        f"### Agent test failure: `{name}`",
        "",
        f"- **Status:** {status}",
        f"- **Reason:** {reason}",
    ]
    if error_type:
        lines.append(f"- **Error type:** `{error_type}`")
    if branch:
        lines.append(f"- **Branch:** `{branch}`")
    if commit:
        lines.append(f"- **Commit:** `{commit}`")
    if trace_id:
        lines.append(f"- **Trace:** `{trace_id}`")
    if fingerprint:
        lines.append(f"- **Fingerprint:** `{fingerprint}`")
    lines.append("")
    lines.append(FINGERPRINT_MARKER.format(fp=fingerprint or "none"))
    return "\n".join(lines) + "\n"


def issue_body(
    scenario: ScenarioResult,
    *,
    branch: str | None = None,
    commit: str | None = None,
    trace_id: str | None = None,
) -> str:
    """Auto-documentation body for a failed scenario, with a dedup marker."""
    return _render_issue(
        name=scenario.name,
        status=scenario.replay.status.value,
        reason=_scenario_reason(scenario) or "unknown failure",
        error_type=scenario.replay.error_type,
        fingerprint=scenario.fingerprint,
        branch=branch,
        commit=commit,
        trace_id=trace_id,
    )


def _reason_from_json(s: dict[str, Any]) -> str:
    """Mirror of :func:`_scenario_reason` for a scenario dict from :func:`to_json`."""
    if s.get("config_error"):
        return f"config error: {s['config_error']}"
    if s.get("status") == "error":
        return f"replay error: {s.get('error_type')}"
    failed = [a for a in s.get("assertions", []) if not a["passed"]]
    if failed:
        return "; ".join(
            f"{a['name']} ({a['detail']})" if a.get("detail") else a["name"] for a in failed
        )
    return ""


def issue_body_from_json(
    scenario: dict[str, Any],
    *,
    branch: str | None = None,
    commit: str | None = None,
    trace_id: str | None = None,
) -> str:
    """Build an issue body from a scenario dict (as emitted by :func:`to_json`)."""
    return _render_issue(
        name=scenario["name"],
        status=scenario["status"],
        reason=_reason_from_json(scenario) or "unknown failure",
        error_type=scenario.get("error_type"),
        fingerprint=scenario.get("fingerprint"),
        branch=branch,
        commit=commit,
        trace_id=trace_id,
    )


def dumps_json(result: CollectionResult) -> str:
    """Serialize :func:`to_json` output as an indented JSON string."""
    return json.dumps(to_json(result), indent=2, default=str)
