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

from pytracer_replay.assertions import describe_requests, repeated_requests
from pytracer_sdk.codec import encode_value
from pytracer_sdk.schema import Span, SpanStatus

from .runner import CollectionResult, ScenarioResult

_PASS = "PASS"
_FAIL = "FAIL"

# Hidden marker embedded in an auto-doc issue body; the fingerprint is the
# dedup key CI matches on to decide "update existing" vs "open new".
FINGERPRINT_MARKER = "<!-- pytracer-fingerprint: {fp} -->"


def _json_value(value: Any) -> Any:
    """A return value in the recording's JSON form (types tagged), or its text when
    it has no JSON form."""
    encoded = encode_value(value)
    try:
        json.dumps(encoded)
    except (TypeError, ValueError):
        return repr(value)
    return encoded


def retry_chains(spans: list[Span]) -> list[dict[str, Any]]:
    """Each op called more than once with the same input under the same parent,
    with how every attempt ended (the tracer numbers them ``pytracer.attempt``)."""
    groups: dict[str, list[Span]] = {}
    for span in spans:
        key = json.dumps(
            [span.parent_span_id, span.kind.value, span.name, span.tool_name, span.input],
            sort_keys=True, default=repr,
        )
        groups.setdefault(key, []).append(span)
    chains: list[dict[str, Any]] = [
        {
            "op": calls[0].tool_name or calls[0].name,
            "kind": calls[0].kind.value,
            "attempts": [
                {
                    "span_id": c.span_id,
                    "status": c.status.value,
                    "error_type": c.error_type,
                    "retry": "pytracer.retry_of" in c.attributes,
                }
                for c in calls
            ],
        }
        for calls in groups.values()
        if any("pytracer.attempt" in c.attributes for c in calls)
    ]
    # Requests a call sent more than once: a retry below the call (the tool's own,
    # or a model SDK's), recorded by the SDK's HTTP capture.
    for span in spans:
        for requests in repeated_requests(span):
            chains.append({
                "op": span.tool_name or span.name,
                "kind": "http",
                "request": f"{requests[0].get('method')} {requests[0].get('url')}",
                "attempts": [
                    {
                        "status": r.get("status"),
                        "error_type": r.get("error"),
                        "retry": bool(r.get("retry")),
                    }
                    for r in requests
                ],
                "text": describe_requests(requests),
            })
    return chains


def _chain_text(chain: dict[str, Any]) -> str:
    if chain["kind"] == "http":
        return f"{chain['op']} repeated {chain['text']}"
    parts = []
    for number, attempt in enumerate(chain["attempts"], 1):
        failed = attempt["status"] == SpanStatus.ERROR.value
        outcome = f"failed ({attempt['error_type'] or 'error'})" if failed else "ok"
        parts.append(f"#{number}{' retry' if attempt['retry'] else ''} {outcome}")
    return f"{chain['op']}: " + ", ".join(parts)


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
                "error_message": s.replay.error_message,
                "config_error": s.config_error,
                "fingerprint": s.fingerprint,
                "side_effects": s.replay.side_effects.summary(),
                "non_idempotent": [
                    {"target": e.target, "kind": e.kind.value}
                    for e in s.replay.side_effects.non_idempotent()
                ],
                "changed_models": s.replay.changed_models,
                # Where the run left its recording, and which of those calls ran
                # live; cross-branch verdicts never trust a run that left it.
                "unrecorded_calls": s.replay.unrecorded_calls,
                "live_fallbacks": s.replay.live_fallbacks,
                "divergences": s.replay.divergences,
                "evaluated_changed_prompts": s.replay.evaluated_changed_prompts,
                "concurrency_ambiguities": s.replay.concurrency_ambiguities,
                "assertions": [
                    {"name": a.name, "passed": a.passed, "detail": a.detail} for a in s.assertions
                ],
                # What the agent returned, every span of the replayed run, and the
                # ops it called more than once (with how each attempt ended).
                "return_value": _json_value(s.replay.return_value),
                "spans": [json.loads(span.to_json()) for span in s.replay.spans],
                "retries": retry_chains(s.replay.spans),
            }
            for s in result.scenarios
        ],
    }


def scenario_reason(s: ScenarioResult) -> str:
    """One-line explanation of why a scenario failed (empty if it passed).

    For a replay error the message is included, not just the type, so a strict
    miss shows which call had no recording (the ``MissingMockError`` text) rather
    than a bare ``MissingMockError``.
    """
    if s.config_error:
        return f"config error: {s.config_error}"
    if s.replay.failed:
        head = s.replay.error_type or "error"
        detail = s.replay.error_message
        return f"replay error: {head}: {detail}" if detail else f"replay error: {head}"
    failed = [a for a in s.assertions if not a.passed]
    if failed:
        return "; ".join(f"{a.name} ({a.detail})" if a.detail else a.name for a in failed)
    return ""


# Backwards-compatible alias (the name was private before it was surfaced in the CLI).
_scenario_reason = scenario_reason


def changed_model_note(s: ScenarioResult) -> str:
    """Explain a verdict driven by a changed model input (a prompt edit), or ''.

    When a frozen model op's input no longer matches a recording, a full-mode
    replay cannot judge the step: the recorded response does not answer the new
    prompt. Say so, so a real fix that edits a prompt is not read as a plain
    'diverged' with no explanation.
    """
    changed = s.replay.changed_models
    if not changed:
        return ""
    names = ", ".join(changed)
    if not s.replay.evaluated_changed_prompts:
        return (
            f"a model input changed ({names}); the recorded response no longer applies, "
            f"so a frozen replay cannot judge it - re-run in partial mode, or with "
            f"--evaluate-changed-prompts, to evaluate the change"
        )
    return (
        f"a model input changed ({names}); it was run live to evaluate the change. The "
        f"verdict only holds if nothing after the changed call also changed; a later step "
        f"not in the recording still cannot be judged"
    )


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
    lines += _changed_model_lines(result)
    lines += _side_effect_lines(result)
    lines += _retry_lines(result)
    return "\n".join(lines) + "\n"


def _retry_lines(result: CollectionResult) -> list[str]:
    """A footer listing ops called more than once, with how each attempt ended:
    the retry evidence a passing run otherwise hides (a duplicated write)."""
    notes = [
        f"- `{s.name}`: {_chain_text(chain)}"
        for s in result.scenarios
        for chain in retry_chains(s.replay.spans)
    ]
    if not notes:
        return []
    return ["", "**Calls repeated during the run:**", *notes]


def _changed_model_lines(result: CollectionResult) -> list[str]:
    """A footer noting scenarios whose verdict was driven by a changed model input."""
    notes = [f"- `{s.name}`: {changed_model_note(s)}" for s in result.scenarios if changed_model_note(s)]
    if not notes:
        return []
    return ["", "**Model inputs changed (e.g. a prompt edit):**", *notes]


def _side_effect_lines(result: CollectionResult) -> list[str]:
    """A footer noting any live external effects seen during replay.

    A full replay should touch nothing live; when one does (a live/partial mode,
    or a non-strict miss that ran a tool for real), surface it so it is not silent,
    and call out writes replayed more than once as a non-idempotency risk.
    """
    notes: list[str] = []
    for s in result.scenarios:
        log = s.replay.side_effects
        if not len(log):
            continue
        kinds = ", ".join(f"{v}×{k}" for k, v in sorted(log.summary().items()))
        note = f"- `{s.name}`: {kinds}"
        repeated = log.non_idempotent()
        if repeated:
            targets = ", ".join(sorted({e.target for e in repeated}))
            note += f" — non-idempotent write replayed: {targets}"
        notes.append(note)
    if not notes:
        return []
    return ["", "**Live side effects during replay:**", *notes]


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


def dedup_key(name: str, fingerprint: str | None) -> str:
    """The stable key used to BOTH embed the dedup marker and find the issue.

    A failure fingerprint dedups by failure; when there is none (an assertion-only
    failure has no fingerprint), fall back to the scenario name so recurrences
    update one issue and distinct scenarios still get distinct issues. The marker
    and the search must use this same key, or a new issue opens every run.
    """
    return fingerprint or f"name:{name}"


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
    # Marker must match the search key (see dedup_key), or dedup fails.
    lines.append(FINGERPRINT_MARKER.format(fp=dedup_key(name, fingerprint)))
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
        head = s.get("error_type") or "error"
        detail = s.get("error_message")
        return f"replay error: {head}: {detail}" if detail else f"replay error: {head}"
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
