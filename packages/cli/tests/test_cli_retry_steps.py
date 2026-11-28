"""Retry steps in reports: attempts, the accepted one, and the tokens wasted."""

from __future__ import annotations

from typing import Any

import pytracer_sdk as pytracer
from pytracer_cli.report import retry_steps, step_text
from pytracer_cli.traces import format_trace_tree
from pytracer_replay import MockSet, ReplayEngine, ReplayMode
from pytracer_sdk import SpanKind
from pytracer_sdk.schema import Span

TOKENS = [1200, 1300, 1100]  # per attempt


def _agent(replies: list[str], ask: str = "Extract the order id.") -> Any:
    """Asks a model until it answers with an order id, inside a retry step."""

    @pytracer.op(kind=SpanKind.LLM, name="extractor")
    def model(prompt: str) -> str:
        return replies[prompt.count("Fix")]

    @pytracer.op(kind=SpanKind.AGENT, agent_name="extract")
    def agent() -> str:
        prompt = ask
        with pytracer.step("extract_order") as step:
            for _ in range(3):
                answer = model(prompt)
                if answer.startswith("#W"):
                    return answer
                step.reject(ValueError(f"not an order id: {answer}"))
                prompt += " Fix: it must start with #W."
        return ""

    return agent


def _recorded(replies: list[str]) -> list[Span]:
    spans = ReplayEngine(ReplayMode.LIVE).replay(_agent(replies), MockSet()).spans
    calls = [s for s in spans if s.kind is SpanKind.LLM]
    for span, tokens in zip(calls, TOKENS, strict=False):
        span.token_count = tokens  # the model seam would count these live
        span.cost_usd = tokens / 1_000_000 * 5
    return spans


def test_a_live_step_reports_its_waste() -> None:
    (step,) = retry_steps(_recorded(["W1", "W1", "#W2378156"]))
    assert (step["attempts"], step["accepted"], step["rejected"]) == (3, 3, 2)
    assert step["tokens"] == 3600 and step["wasted_tokens"] == 2500
    assert abs(step["wasted_cost_usd"] - 2500 / 1_000_000 * 5) < 1e-12
    assert [r["tokens_from"] for r in step["attempt_detail"]] == ["run"] * 3
    assert step_text(step) == (
        "extract_order: 3 attempts, #3 accepted; #1 rejected (ValueError); "
        "#2 rejected (ValueError); 2,500 of 3,600 tokens wasted ($0.0125)"
    )


def test_a_step_with_nothing_accepted_wastes_every_attempt() -> None:
    (step,) = retry_steps(_recorded(["W1", "W1", "W1"]))
    assert step["accepted"] is None and step["wasted_tokens"] == step["tokens"] == 3600


def test_a_frozen_replay_takes_tokens_from_its_recording() -> None:
    replies = ["W1", "#W2378156"]
    recording = _recorded(replies)
    replay = ReplayEngine(ReplayMode.FULL, strict=True).replay(
        _agent(replies), MockSet.from_trace(recording)  # type: ignore[attr-defined]
    )
    (step,) = retry_steps(replay.spans, recording)
    assert step["wasted_tokens"] == 1200 and step["tokens"] == 2500
    assert [r["tokens_from"] for r in step["attempt_detail"]] == ["recording"] * 2
    # Without the recording a frozen call's cost is unknown, and says so.
    (blind,) = retry_steps(replay.spans)
    assert blind["wasted_tokens"] is None
    assert "tokens wasted unknown" in step_text(blind)


def test_a_replay_that_left_its_recording_does_not_borrow_tokens() -> None:
    # A changed prompt misses the recording; with evaluate_changed_prompts the model
    # runs live (no token counter here), so its cost is unknown, not borrowed.
    recording = _recorded(["W1", "#W2378156"])
    replay = ReplayEngine(
        ReplayMode.FULL, strict=False, sandbox=False, evaluate_changed_prompts=True
    ).replay(_agent(["W1", "#W2378156"], ask="Find the order id."), MockSet.from_trace(recording))  # type: ignore[attr-defined]
    (step,) = retry_steps(replay.spans, recording)
    assert step["attempts"] == 2
    assert [r["tokens_from"] for r in step["attempt_detail"]] == [None, None]
    assert step["wasted_tokens"] is None


def test_traces_show_marks_the_step_and_each_attempt() -> None:
    tree = format_trace_tree(_recorded(["W1", "W1", "#W2378156"]))
    assert "extract_order" in tree and "3 attempts, #3 accepted" in tree
    assert "attempt 1, rejected (ValueError), 1,200 tokens" in tree
    assert "attempt 3, accepted, 1,100 tokens" in tree


def test_a_retry_step_over_a_provider_response_reports_its_waste_with_no_token_code() -> None:
    # The model call returns an OpenAI-style response; its usage is read by default.
    from types import SimpleNamespace

    answers = iter(["W1", "#W2378156"])

    @pytracer.op(kind=SpanKind.LLM, name="extractor")
    def model(prompt: str) -> Any:
        return SimpleNamespace(text=next(answers), usage=SimpleNamespace(total_tokens=1500))

    @pytracer.op(kind=SpanKind.AGENT, agent_name="extract")
    def agent() -> str:
        with pytracer.step("extract_order") as step:
            for _ in range(2):
                answer = model("Extract the order id.").text
                if answer.startswith("#W"):
                    return answer
                step.reject(ValueError("not an order id"))
        return ""

    spans = ReplayEngine(ReplayMode.LIVE).replay(agent, MockSet()).spans
    (step,) = retry_steps(spans)
    assert step["tokens"] == 3000 and step["wasted_tokens"] == 1500
