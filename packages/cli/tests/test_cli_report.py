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
from pytracer_cli.report import changed_model_note, issue_body_from_json


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


def _changed_model_scenario() -> ScenarioResult:
    r = ReplayResult(
        return_value=None, status=SpanStatus.ERROR, mode=ReplayMode.FULL,
        error_type="MissingMockError", model_input_changes=["call_model"],
    )
    return ScenarioResult("prompt-edit", False, r, [])


def test_changed_model_note_and_json_explain_a_prompt_edit() -> None:
    # A failure driven by a changed model input is explained, not a bare diverge (X4).
    s = _changed_model_scenario()
    note = changed_model_note(s)
    assert "call_model" in note and "partial mode" in note
    result = CollectionResult("suite", [s])
    assert to_json(result)["scenarios"][0]["changed_models"] == ["call_model"]
    assert "Model inputs changed" in to_markdown(result)


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


def test_json_and_issue_body_carry_the_replay_error_message() -> None:
    # R3: the --json report, and the issue body built from it, must carry the full
    # replay reason so issues sync / CI can see which call had no recording.
    msg = "no recording for llm 'llm.complete' (occurrence 0): its input did not match"
    r = ReplayResult(
        return_value=None,
        status=SpanStatus.ERROR,
        mode=ReplayMode.FULL,
        error_type="MissingMockError",
        error_message=msg,
    )
    sj = to_json(CollectionResult("suite", [ScenarioResult("miss", False, r, [])]))["scenarios"][0]
    assert sj["error_type"] == "MissingMockError"
    assert sj["error_message"] == msg
    body = issue_body_from_json(sj, branch="b", commit="c")
    assert "llm.complete" in body  # the "which call" detail reaches the issue
    assert "MissingMockError" in body


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
