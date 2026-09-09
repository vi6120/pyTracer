"""Report formatting: markdown, JSON, JUnit, and auto-doc issue body."""

from __future__ import annotations

from pytracer_replay import AssertionResult, ReplayMode, ReplayResult
from pytracer_sdk import SpanStatus

from pytracer_cli import (
    CollectionResult,
    ScenarioResult,
    issue_body,
    to_json,
    to_junit,
    to_markdown,
)


def _ok_scenario() -> ScenarioResult:
    r = ReplayResult(return_value="summary: x", status=SpanStatus.OK, mode=ReplayMode.FULL)
    return ScenarioResult("happy", True, r, [AssertionResult("contains", True)])


def _failed_scenario() -> ScenarioResult:
    r = ReplayResult(
        return_value=None, status=SpanStatus.ERROR, mode=ReplayMode.FULL, error_type="KeyError"
    )
    return ScenarioResult(
        "broken", False, r, [AssertionResult("no_errors", False, "KeyError")], fingerprint="abc123"
    )


def test_markdown_summary() -> None:
    result = CollectionResult("suite", [_ok_scenario(), _failed_scenario()])
    md = to_markdown(result)
    assert "1/2 scenarios passed" in md
    assert "`happy`" in md and "`broken`" in md
    assert "FAIL" in md and "PASS" in md


def test_json_shape() -> None:
    result = CollectionResult("suite", [_ok_scenario(), _failed_scenario()])
    payload = to_json(result)
    assert payload["counts"] == {"total": 2, "passed": 1, "failed": 1}
    assert payload["passed"] is False
    assert payload["scenarios"][1]["fingerprint"] == "abc123"


def test_junit_reports_failures() -> None:
    result = CollectionResult("suite", [_ok_scenario(), _failed_scenario()])
    xml = to_junit(result)
    assert 'tests="2"' in xml
    assert 'failures="1"' in xml
    assert "<failure" in xml


def test_issue_body_carries_fingerprint_marker() -> None:
    body = issue_body(_failed_scenario(), branch="feature/x", commit="deadbeef", trace_id="t1")
    assert "abc123" in body
    assert "<!-- pytracer-fingerprint: abc123 -->" in body  # dedup marker for CI
    assert "feature/x" in body
    assert "deadbeef" in body
