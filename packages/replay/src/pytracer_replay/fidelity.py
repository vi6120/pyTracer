"""Replay fidelity helpers for high-faithfulness replay.

Deterministic replay (``engine.py``) injects recorded model and tool responses so
a run reproduces exactly. These helpers harden it for the cases that still drift:

  - **Streaming** - re-emit a recorded model response chunk by chunk, optionally
    with its captured timing (``RecordedStream``).
  - **Concurrency** - give each op call a stable ``CallKey`` so concurrent tool
    calls resolve to the right recording regardless of arrival order
    (``ConcurrencyLedger``).
  - **Nondeterminism controls** - pin the clock and seed the RNG around agent
    execution, and collapse idempotent retries (``controlled``, ``RetryDeduper``).
  - **External side effects** - record writes seen during replay and flag
    non-idempotent or conflicting ones (``SideEffectLog``).

Each is a standalone utility the engine and runner compose: the interceptor can
consult a ``ConcurrencyLedger``; the runner can wrap ``agent()`` in
``controlled(...)``; the sandbox can feed a ``SideEffectLog``. They keep no state
in the engine, so nothing in the core replay path changes.
"""

from __future__ import annotations

import datetime as _datetime
import random
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .hashing import input_hash

__all__ = [
    "CallKey",
    "Chunk",
    "ConcurrencyLedger",
    "NondeterminismControls",
    "RecordedStream",
    "RetryDeduper",
    "SideEffect",
    "SideEffectKind",
    "SideEffectLog",
    "controlled",
    "stream_from_text",
]


# --- streaming responses ---------------------------------------------------

@dataclass(frozen=True)
class Chunk:
    """One recorded piece of a streamed model response."""

    index: int
    text: str
    delay_ms: float = 0.0  # delay before this chunk, kept for timing fidelity


@dataclass
class RecordedStream:
    """A model response captured as an ordered list of streamed chunks."""

    chunks: list[Chunk] = field(default_factory=list)

    def _ordered(self) -> list[Chunk]:
        return sorted(self.chunks, key=lambda c: c.index)

    @property
    def text(self) -> str:
        """The full response, chunks joined in index order."""
        return "".join(c.text for c in self._ordered())

    def replay(
        self, *, honor_timing: bool = False, sleep: Callable[[float], Any] = time.sleep
    ) -> Iterator[str]:
        """Yield chunk texts in order; with ``honor_timing`` reproduce the delays.

        ``sleep`` is injectable so timing can be driven or faked in tests.
        """
        for chunk in self._ordered():
            if honor_timing and chunk.delay_ms > 0:
                sleep(chunk.delay_ms / 1000.0)
            yield chunk.text


def stream_from_text(text: str, *, chunk_size: int = 1, delay_ms: float = 0.0) -> RecordedStream:
    """Build a ``RecordedStream`` by splitting ``text`` into fixed-size chunks."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    parts = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    return RecordedStream([Chunk(i, part, delay_ms) for i, part in enumerate(parts)])


# --- async / concurrent tool-call ordering ---------------------------------

@dataclass(frozen=True)
class CallKey:
    """Stable identity for one op call, independent of concurrent arrival order."""

    identifier: str
    input_hash: str
    occurrence: int


@dataclass
class ConcurrencyLedger:
    """Thread-safe per-call occurrence numbering, assigned in arrival order.

    Each call to ``(identifier, input)`` gets the next occurrence index, so repeat
    calls resolve to their recordings in sequence. This is correct for sequential
    repeats, and for concurrent calls whose inputs differ (each keys to its own
    recording regardless of arrival order).

    It does NOT disambiguate concurrent calls that share the same input: those are
    numbered by the order they happen to arrive, which need not match the recording
    order, so if their recorded responses differ the two can be swapped. That case
    cannot be resolved from arrival order alone; it needs a caller-supplied ordering
    key, which replay does not yet capture. When the shared-input responses are
    identical (the common case) the swap is harmless.
    """

    _counts: dict[tuple[str, str], int] = field(default_factory=dict)
    _identifier_counts: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def key_for(self, identifier: str, inp: object) -> CallKey:
        """Return the stable key for the next call of ``identifier`` with ``inp``."""
        ih = input_hash(inp)
        with self._lock:
            n = self._counts.get((identifier, ih), 0)
            self._counts[(identifier, ih)] = n + 1
        return CallKey(identifier=identifier, input_hash=ih, occurrence=n)

    def positional_for(self, identifier: str) -> int:
        """Return the next call index for ``identifier``, ignoring the input.

        Positional matching serves the Nth recording for the Nth call to a tool,
        so it needs the per-identifier call order, not the per-(identifier, input)
        one: in partial mode a live model re-words the input every call, so each
        call would otherwise look like occurrence 0 and always get the first
        recording.
        """
        with self._lock:
            n = self._identifier_counts.get(identifier, 0)
            self._identifier_counts[identifier] = n + 1
        return n


# --- nondeterminism controls -----------------------------------------------

@dataclass(frozen=True)
class NondeterminismControls:
    """What to pin so a replay is reproducible beyond the mocked responses."""

    freeze_time: float | None = None  # unix seconds to pin time.time() to
    seed: int | None = None           # RNG seed for the stdlib random module


@contextmanager
def controlled(controls: NondeterminismControls) -> Iterator[None]:
    """Pin the clock, the calendar, UUIDs and the RNG for the block, then restore.

    With ``freeze_time`` set, freezes the ``time`` module clocks
    (``time``/``time_ns``/``monotonic``/``monotonic_ns``/``perf_counter``/
    ``perf_counter_ns``) and ``datetime.datetime.now``/``utcnow`` to that instant.
    With ``seed`` set, seeds the stdlib ``random`` module and makes ``uuid.uuid4``
    draw from a seeded stream so ids repeat run to run. Restoration is guaranteed
    on exit.

    These patch the process-global modules while active, so run them only around a
    replay. The patch reaches code that looks the symbol up on its module at call
    time (``time.time()``, ``datetime.datetime.now()``, ``uuid.uuid4()``); code
    that bound a name at import (``from datetime import datetime``) keeps the real
    one, and third-party clocks/RNGs (numpy, ``os.urandom``) are out of scope.
    """
    restore: list[tuple[Any, str, Any]] = []

    def patch(target: Any, name: str, fresh: Any) -> None:
        if hasattr(target, name):
            restore.append((target, name, getattr(target, name)))
            setattr(target, name, fresh)

    if controls.seed is not None:
        random.seed(controls.seed)
        # A separate stream so draining uuids does not perturb the agent's own RNG.
        uuid_rng = random.Random(controls.seed)
        patch(uuid, "uuid4", lambda: uuid.UUID(int=uuid_rng.getrandbits(128), version=4))
    if controls.freeze_time is not None:
        frozen = controls.freeze_time
        frozen_ns = int(frozen * 1_000_000_000)
        patch(time, "time", lambda: frozen)
        patch(time, "time_ns", lambda: frozen_ns)
        patch(time, "monotonic", lambda: 0.0)
        patch(time, "monotonic_ns", lambda: 0)
        patch(time, "perf_counter", lambda: 0.0)
        patch(time, "perf_counter_ns", lambda: 0)

        frozen_aware = _datetime.datetime.fromtimestamp(frozen, tz=_datetime.timezone.utc)
        # Naive local wall-clock for that instant, matching what real now() returns.
        frozen_local = _datetime.datetime.fromtimestamp(frozen)  # noqa: DTZ006

        class _FrozenDateTime(_datetime.datetime):
            # Overrides return the frozen instant; the base datetime return type is
            # intentional (the pinned value is a plain datetime, not this subclass).
            @classmethod
            def now(cls, tz: Any = None) -> _datetime.datetime:  # type: ignore[override]
                # now(None) -> naive local; now(tz) -> aware in tz (as stdlib does).
                return frozen_aware.astimezone(tz) if tz is not None else frozen_local

            @classmethod
            def utcnow(cls) -> _datetime.datetime:  # type: ignore[override]
                return frozen_aware.replace(tzinfo=None)

        patch(_datetime, "datetime", _FrozenDateTime)
    try:
        yield
    finally:
        for target, name, original in reversed(restore):
            setattr(target, name, original)


@dataclass
class RetryDeduper:
    """Collapse repeated identical calls (an agent's idempotent retries).

    A second call with the same identifier and input is a retry, so it need not
    consume a fresh recording.
    """

    _seen: set[str] = field(default_factory=set)

    def is_retry(self, identifier: str, inp: object) -> bool:
        """True if (identifier, input) was seen before; records it either way."""
        token = f"{identifier}:{input_hash(inp)}"
        if token in self._seen:
            return True
        self._seen.add(token)
        return False


# --- external side-effect detection ----------------------------------------

class SideEffectKind(str, Enum):
    """A category of external effect observed during replay.

    ``READ`` is read-only (a query, a GET); replaying it twice is harmless.
    ``WRITE`` / ``DELETE`` mutate an external system, so a repeat is the signal
    that matters. ``EXTERNAL_CALL`` is a live call we could not classify: it is
    surfaced, but not treated as a mutation, because assuming it writes would flag
    every read as non-idempotent.
    """

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXTERNAL_CALL = "external_call"


# The kinds that mutate external state; only these can be "non-idempotent" or
# "conflicting". A read or an unclassified call is never flagged on its own.
MUTATING_KINDS: frozenset[SideEffectKind] = frozenset(
    {SideEffectKind.WRITE, SideEffectKind.DELETE}
)


@dataclass(frozen=True)
class SideEffect:
    """An external effect seen while replaying (a DB row, a file, an API call)."""

    target: str
    kind: SideEffectKind
    input_hash: str


@dataclass
class SideEffectLog:
    """Records external effects seen during a replay, in order."""

    effects: list[SideEffect] = field(default_factory=list)

    def record(self, target: str, kind: SideEffectKind, inp: object) -> None:
        """Append one observed side effect."""
        self.effects.append(SideEffect(target, kind, input_hash(inp)))

    def mutations(self) -> list[SideEffect]:
        """The effects that mutate external state (writes and deletes)."""
        return [e for e in self.effects if e.kind in MUTATING_KINDS]

    def non_idempotent(self) -> list[SideEffect]:
        """Distinct mutating effects whose (target, kind, input) occur more than once.

        The same write replayed twice is the signal for a non-idempotent effect.
        Reads and unclassified calls are excluded: a tool read three times is not
        a non-idempotent write, and flagging it as one was a false alarm.
        """
        counts: dict[tuple[str, str, str], int] = {}
        for e in self.effects:
            if e.kind not in MUTATING_KINDS:
                continue
            key = (e.target, e.kind.value, e.input_hash)
            counts[key] = counts.get(key, 0) + 1
        out: list[SideEffect] = []
        emitted: set[tuple[str, str, str]] = set()
        for e in self.effects:
            if e.kind not in MUTATING_KINDS:
                continue
            key = (e.target, e.kind.value, e.input_hash)
            if counts[key] > 1 and key not in emitted:
                out.append(e)
                emitted.add(key)
        return out

    def conflicting_targets(self) -> list[str]:
        """Targets mutated with two or more distinct inputs (write-write risk)."""
        by_target: dict[str, set[str]] = {}
        for e in self.effects:
            if e.kind not in MUTATING_KINDS:
                continue
            by_target.setdefault(e.target, set()).add(e.input_hash)
        return sorted(t for t, inputs in by_target.items() if len(inputs) > 1)

    def summary(self) -> dict[str, int]:
        """Count of observed side effects by kind, for surfacing to the user."""
        out: dict[str, int] = {}
        for e in self.effects:
            out[e.kind.value] = out.get(e.kind.value, 0) + 1
        return out

    def __len__(self) -> int:
        return len(self.effects)
