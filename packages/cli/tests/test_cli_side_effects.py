"""Side-effect assertions from a collection, and the retry evidence in the reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cli_refund_agent
import pytest
from cli_testkit import record_trace, write_collection

from pytracer_cli import run_path
from pytracer_cli.report import to_json, to_markdown


def _make(tmp_path: Path, assertions: list[dict[str, Any]], writes: list[str] | None = None) -> Path:
    cli_refund_agent.TIMEOUTS[:] = [TimeoutError("payment gateway timed out")]
    record_trace(cli_refund_agent.agent, {"order_id": "A1"}, tmp_path / "trace.jsonl")
    scenario: dict[str, Any] = {
        "name": "refund",
        "agent": "cli_refund_agent:agent",
        "entry": {"order_id": "A1"},
        "trace": "trace.jsonl",
        "mode": "full",
        "assertions": assertions,
    }
    if writes is not None:
        scenario["writes"] = writes
    return write_collection(tmp_path, [scenario])


def test_a_duplicate_refund_fails_the_scenario_with_its_retry_evidence(tmp_path: Path) -> None:
    collection = _make(
        tmp_path,
        [
            {"type": "no_errors"},
            {"type": "returns", "expected": {"status": "refunded"}},
            {"type": "writes", "expected": [{"tool": "refund_payment", "times": 1, "args": {"order_id": "A1"}}]},
            {"type": "no_duplicate_writes"},
        ],
        writes=["refund_payment"],
    )
    result = run_path(collection)
    assert not result.passed
    checks = {a.name: a for a in result.scenarios[0].assertions}
    assert checks["no_errors"].passed and checks["returns"].passed  # it looks fine
    assert not checks["writes"].passed
    assert "#1 failed (TimeoutError), #2 retry ok" in checks["writes"].detail
    assert not checks["no_duplicate_writes"].passed


def test_run_json_carries_the_return_value_spans_and_retries(tmp_path: Path) -> None:
    result = run_path(_make(tmp_path, [{"type": "no_errors"}]))
    scenario = to_json(result)["scenarios"][0]
    assert scenario["return_value"] == {"order_id": "A1", "status": "refunded"}
    tools = [s["tool_name"] for s in scenario["spans"] if s["kind"] == "tool"]
    assert tools == ["lookup_order", "refund_payment", "refund_payment"]
    [chain] = scenario["retries"]
    assert chain["op"] == "refund_payment"
    assert [(a["status"], a["error_type"], a["retry"]) for a in chain["attempts"]] == [
        ("error", "TimeoutError", False), ("ok", None, True),
    ]


def test_the_markdown_report_lists_repeated_calls(tmp_path: Path) -> None:
    markdown = to_markdown(run_path(_make(tmp_path, [{"type": "no_errors"}])))
    assert "**Calls repeated during the run:**" in markdown
    assert "`refund`: refund_payment: #1 failed (TimeoutError), #2 retry ok" in markdown


def test_only_needs_declared_write_tools(tmp_path: Path) -> None:
    collection = _make(tmp_path, [{"type": "writes", "only": True, "expected": ["refund_payment"]}])
    result = run_path(collection)
    assert "'only' needs the scenario's 'writes' declared" in (result.scenarios[0].config_error or "")


@pytest.mark.parametrize(
    ("assertion", "message"),
    [
        ({"type": "no_duplicate_writes"}, "needs 'tools' or the scenario's 'writes'"),
        ({"type": "writes", "expected": [{"tool": "refund_payment", "times": -1}]}, "non-negative"),
        ({"type": "writes", "expected": [{"tool": "x", "count": 1}]}, "unknown fields"),
        ({"type": "returns"}, "'returns' requires 'expected'"),
    ],
)
def test_bad_side_effect_assertions_are_config_errors(
    tmp_path: Path, assertion: dict[str, Any], message: str
) -> None:
    result = run_path(_make(tmp_path, [assertion]))
    assert message in (result.scenarios[0].config_error or "")
