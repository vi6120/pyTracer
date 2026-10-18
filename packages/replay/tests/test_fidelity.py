"""Replay fidelity helpers: streaming, concurrency, nondeterminism, side effects."""

from __future__ import annotations

import random
import threading
import time

import pytest
import pytracer_sdk as pytracer
from pytracer_sdk import SpanKind
from pytracer_replay import (
    Chunk,
    ConcurrencyLedger,
    MockSet,
    NondeterminismControls,
    RecordedStream,
    ReplayEngine,
    ReplayMode,
    RetryDeduper,
    SideEffectKind,
    SideEffectLog,
    controlled,
    input_hash,
    stream_from_text,
)


# --- streaming -------------------------------------------------------------

def test_recorded_stream_text_orders_by_index() -> None:
    stream = RecordedStream([Chunk(2, "c"), Chunk(0, "a"), Chunk(1, "b")])
    assert stream.text == "abc"


def test_recorded_stream_replay_yields_in_order() -> None:
    calls: list[float] = []
    stream = RecordedStream([Chunk(1, "b"), Chunk(0, "a")])
    assert list(stream.replay(sleep=calls.append)) == ["a", "b"]
    assert calls == []  # no timing unless asked


def test_recorded_stream_replay_honors_timing() -> None:
    calls: list[float] = []
    stream = RecordedStream([Chunk(0, "a", 10.0), Chunk(1, "b", 20.0)])
    assert list(stream.replay(honor_timing=True, sleep=calls.append)) == ["a", "b"]
    assert calls == [0.01, 0.02]


def test_stream_from_text_chunks() -> None:
    stream = stream_from_text("hello", chunk_size=2)
    assert [c.text for c in sorted(stream.chunks, key=lambda c: c.index)] == ["he", "ll", "o"]
    assert stream.text == "hello"


def test_stream_from_text_rejects_bad_chunk_size() -> None:
    with pytest.raises(ValueError):
        stream_from_text("x", chunk_size=0)


# --- concurrency -----------------------------------------------------------

def test_concurrency_ledger_numbers_occurrences() -> None:
    ledger = ConcurrencyLedger()
    k0 = ledger.key_for("search", {"q": "x"})
    k1 = ledger.key_for("search", {"q": "x"})
    k2 = ledger.key_for("search", {"q": "y"})
    assert (k0.occurrence, k1.occurrence, k2.occurrence) == (0, 1, 0)
    assert k0.input_hash == k1.input_hash
    assert k0.input_hash != k2.input_hash


def test_concurrency_ledger_is_thread_safe() -> None:
    ledger = ConcurrencyLedger()
    results: list[int] = []
    guard = threading.Lock()

    def worker() -> None:
        key = ledger.key_for("tool", {"a": 1})
        with guard:
            results.append(key.occurrence)

    threads = [threading.Thread(target=worker) for _ in range(50)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(results) == list(range(50))  # every occurrence unique, no races


# --- nondeterminism controls -----------------------------------------------

def test_controlled_seeds_random_reproducibly() -> None:
    with controlled(NondeterminismControls(seed=42)):
        first = [random.random() for _ in range(5)]
    with controlled(NondeterminismControls(seed=42)):
        second = [random.random() for _ in range(5)]
    assert first == second


def test_controlled_freezes_and_restores_time() -> None:
    original = time.time
    with controlled(NondeterminismControls(freeze_time=1000.0)):
        assert time.time() == 1000.0
        assert time.monotonic() == 0.0
    assert time.time is original
    assert time.time() != 1000.0  # real clock restored


def test_retry_deduper_flags_second_identical_call() -> None:
    deduper = RetryDeduper()
    assert deduper.is_retry("llm", {"p": 1}) is False
    assert deduper.is_retry("llm", {"p": 1}) is True
    assert deduper.is_retry("llm", {"p": 2}) is False


# --- side effects ----------------------------------------------------------

def test_side_effect_log_detects_non_idempotent() -> None:
    log = SideEffectLog()
    log.record("crm.lead", SideEffectKind.WRITE, {"id": 7})
    log.record("crm.lead", SideEffectKind.WRITE, {"id": 7})
    log.record("crm.lead", SideEffectKind.WRITE, {"id": 8})
    bad = log.non_idempotent()
    assert len(bad) == 1
    assert bad[0].target == "crm.lead"
    assert bad[0].input_hash == input_hash({"id": 7})


def test_side_effect_log_flags_conflicting_targets() -> None:
    log = SideEffectLog()
    log.record("file:/a", SideEffectKind.WRITE, {"v": 1})
    log.record("file:/a", SideEffectKind.WRITE, {"v": 2})
    log.record("file:/b", SideEffectKind.WRITE, {"v": 1})
    assert log.conflicting_targets() == ["file:/a"]


# --- engine integration ----------------------------------------------------

def test_engine_applies_nondeterminism_controls() -> None:
    @pytracer.op(kind=SpanKind.AGENT, agent_name="clock")
    def agent() -> float:
        return time.time()

    engine = ReplayEngine(ReplayMode.LIVE, controls=NondeterminismControls(freeze_time=123.0))
    result = engine.replay(agent, MockSet())
    assert result.return_value == 123.0
    assert time.time() != 123.0  # clock restored after the replay


def test_engine_seeds_random_for_reproducible_live_runs() -> None:
    @pytracer.op(kind=SpanKind.AGENT, agent_name="rng")
    def agent() -> float:
        return random.random()

    engine = ReplayEngine(ReplayMode.LIVE, controls=NondeterminismControls(seed=7))
    first = engine.replay(agent, MockSet()).return_value
    second = engine.replay(agent, MockSet()).return_value
    assert first == second


def test_engine_records_live_tool_calls_as_side_effects() -> None:
    @pytracer.op(kind=SpanKind.TOOL, tool_name="writer")
    def writer(x: int) -> str:
        return f"wrote-{x}"

    @pytracer.op(kind=SpanKind.AGENT, agent_name="a")
    def agent() -> str:
        writer(1)
        writer(1)
        writer(2)
        return "done"

    result = ReplayEngine(ReplayMode.LIVE).replay(agent, MockSet())
    targets = [e.target for e in result.side_effects.effects]
    assert targets.count("writer") == 3
    assert len(result.side_effects.non_idempotent()) == 1  # writer(1) ran twice


def test_engine_records_no_side_effects_when_fully_mocked() -> None:
    @pytracer.op(kind=SpanKind.TOOL, tool_name="t")
    def tool(x: int) -> int:
        return x * 2

    @pytracer.op(kind=SpanKind.AGENT, agent_name="a")
    def agent() -> int:
        return tool(3)

    recording = ReplayEngine(ReplayMode.LIVE).replay(agent, MockSet())
    mocks = MockSet.from_trace(recording.spans)
    result = ReplayEngine(ReplayMode.FULL).replay(agent, mocks)
    assert result.side_effects.effects == []
