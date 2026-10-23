"""Replay fidelity helpers: streaming, concurrency, nondeterminism, side effects."""

from __future__ import annotations

import random
import threading
import time
# Bound by name at module import, to prove controlled() pins these too (R10).
from datetime import datetime as _dt_byname
from time import time as _time_byname
from uuid import uuid4 as _uuid4_byname

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


def test_controlled_freezes_time_ns_and_datetime() -> None:
    import datetime

    original_ns = time.time_ns
    with controlled(NondeterminismControls(freeze_time=1000.0)):
        assert time.time_ns() == 1_000_000_000_000
        assert datetime.datetime.now().timestamp() == pytest.approx(1000.0, abs=1.0)  # noqa: DTZ005
        assert datetime.datetime.now(datetime.timezone.utc).timestamp() == 1000.0
    assert time.time_ns is original_ns
    assert time.time_ns() != 1_000_000_000_000  # real clock restored


def test_controlled_pins_symbols_imported_by_name() -> None:
    # `from datetime import datetime` / `from time import time` / `from uuid import
    # uuid4` are pinned too, not only the module-qualified forms (R10).
    with controlled(NondeterminismControls(freeze_time=1000.0, seed=3)):
        assert _time_byname() == 1000.0
        assert _dt_byname.now().timestamp() == pytest.approx(1000.0, abs=1.0)  # noqa: DTZ005
        first = str(_uuid4_byname())
    with controlled(NondeterminismControls(freeze_time=1000.0, seed=3)):
        assert str(_uuid4_byname()) == first  # same seed -> same id
    assert _time_byname() != 1000.0  # real clock restored


def test_controlled_datetime_swap_preserves_isinstance() -> None:
    # datetime is swapped process-wide; a real datetime must still pass isinstance,
    # or replacing the class would break type checks in the agent being replayed.
    import datetime as dt

    real = dt.datetime(2020, 1, 1)  # noqa: DTZ001
    with controlled(NondeterminismControls(freeze_time=1000.0)):
        assert isinstance(real, dt.datetime)
        assert isinstance(dt.datetime.now(), dt.datetime)  # noqa: DTZ005
        assert issubclass(dt.datetime, dt.datetime)
    assert isinstance(real, dt.datetime)  # still fine after restore


def test_controlled_makes_uuid4_deterministic_and_restores() -> None:
    import uuid

    original = uuid.uuid4
    with controlled(NondeterminismControls(seed=5)):
        first = [str(uuid.uuid4()) for _ in range(3)]
    with controlled(NondeterminismControls(seed=5)):
        second = [str(uuid.uuid4()) for _ in range(3)]
    assert first == second
    assert uuid.uuid4 is original  # real uuid4 restored


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


def test_engine_records_declared_writes_and_flags_non_idempotent() -> None:
    @pytracer.op(kind=SpanKind.TOOL, tool_name="writer")
    def writer(x: int) -> str:
        return f"wrote-{x}"

    @pytracer.op(kind=SpanKind.AGENT, agent_name="a")
    def agent() -> str:
        writer(1)
        writer(1)
        writer(2)
        return "done"

    # The tool is declared a writer, so its live calls log as writes and the
    # repeated write(1) is flagged non-idempotent.
    result = ReplayEngine(ReplayMode.LIVE, writes=["writer"]).replay(agent, MockSet())
    targets = [e.target for e in result.side_effects.effects]
    assert targets.count("writer") == 3
    assert all(e.kind is SideEffectKind.WRITE for e in result.side_effects.effects)
    assert len(result.side_effects.non_idempotent()) == 1  # writer(1) ran twice


def test_engine_does_not_flag_repeated_reads_as_non_idempotent() -> None:
    @pytracer.op(kind=SpanKind.TOOL, tool_name="reader")
    def reader(q: str) -> str:
        return "catalog"

    @pytracer.op(kind=SpanKind.AGENT, agent_name="a")
    def agent() -> str:
        reader("same")
        reader("same")
        reader("same")
        return "done"

    # An undeclared tool logs as an unclassified external call, so reading the
    # same thing three times is surfaced but never flagged non-idempotent (R11).
    result = ReplayEngine(ReplayMode.LIVE).replay(agent, MockSet())
    assert len(result.side_effects) == 3
    assert all(e.kind is SideEffectKind.EXTERNAL_CALL for e in result.side_effects.effects)
    assert result.side_effects.non_idempotent() == []


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


def test_engine_replays_repeated_same_input_calls_in_sequence() -> None:
    produced = iter([10, 20, 30])

    @pytracer.op(kind=SpanKind.TOOL, tool_name="counter")
    def counter(k: int) -> int:
        return next(produced)

    @pytracer.op(kind=SpanKind.AGENT, agent_name="a")
    def agent() -> list[int]:
        return [counter(1), counter(1), counter(1)]  # same input, different outputs

    recording = ReplayEngine(ReplayMode.LIVE).replay(agent, MockSet())
    assert recording.return_value == [10, 20, 30]
    mocks = MockSet.from_trace(recording.spans)
    result = ReplayEngine(ReplayMode.FULL).replay(agent, mocks)
    assert result.return_value == [10, 20, 30]  # recordings replayed in order


def test_engine_replays_a_recorded_stream_as_chunks() -> None:
    seen: list[str] = []

    @pytracer.op(kind=SpanKind.LLM)
    def model(prompt: str) -> str:
        return "hello"  # live "stream": a string iterated character by character

    @pytracer.op(kind=SpanKind.AGENT, agent_name="a")
    def agent() -> str:
        out = ""
        for chunk in model("hi"):
            seen.append(chunk)
            out += chunk
        return out

    recording = ReplayEngine(ReplayMode.LIVE).replay(agent, MockSet())
    model_span = next(s for s in recording.spans if s.kind is SpanKind.LLM)
    mocks = MockSet()
    mocks.add("model", model_span.input, stream_from_text("hello", chunk_size=1))

    seen.clear()
    result = ReplayEngine(ReplayMode.FULL).replay(agent, mocks)
    assert result.return_value == "hello"
    assert seen == ["h", "e", "l", "l", "o"]  # delivered chunk by chunk, in order
