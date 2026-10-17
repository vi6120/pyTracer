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

import random
import threading
import time
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
    """Deterministic, thread-safe per-call occurrence numbering.

    Concurrent tool calls arrive in a nondeterministic order, so matching them to
    recordings by arrival is flaky. This assigns each (identifier, input) its next
    occurrence index, giving a stable ``CallKey`` to look up the right recording.
    """

    _counts: dict[tuple[str, str], int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def key_for(self, identifier: str, inp: object) -> CallKey:
        """Return the stable key for the next call of ``identifier`` with ``inp``."""
        ih = input_hash(inp)
        with self._lock:
            n = self._counts.get((identifier, ih), 0)
            self._counts[(identifier, ih)] = n + 1
        return CallKey(identifier=identifier, input_hash=ih, occurrence=n)


# --- nondeterminism controls -----------------------------------------------

@dataclass(frozen=True)
class NondeterminismControls:
    """What to pin so a replay is reproducible beyond the mocked responses."""

    freeze_time: float | None = None  # unix seconds to pin time.time() to
    seed: int | None = None           # RNG seed for the stdlib random module


@contextmanager
def controlled(controls: NondeterminismControls) -> Iterator[None]:
    """Pin the clock and seed the RNG for the duration of the block, then restore.

    Freezes ``time.time`` / ``time.monotonic`` / ``time.perf_counter`` and seeds
    ``random``. Restoration is guaranteed on exit. Note: this patches the process
    clock globally while active, so run it only around a replay; ``datetime.now``
    and third-party RNGs (numpy, uuid) are out of scope here.
    """
    if controls.seed is not None:
        random.seed(controls.seed)
    restore: list[tuple[str, Any]] = []
    if controls.freeze_time is not None:
        frozen = controls.freeze_time
        patches: list[tuple[str, Callable[[], float]]] = [
            ("time", lambda: frozen),
            ("monotonic", lambda: 0.0),
            ("perf_counter", lambda: 0.0),
        ]
        for name, fresh in patches:
            restore.append((name, getattr(time, name)))
            setattr(time, name, fresh)
    try:
        yield
    finally:
        for name, original in restore:
            setattr(time, name, original)


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
    """A category of external write observed during replay."""

    WRITE = "write"
    DELETE = "delete"
    EXTERNAL_CALL = "external_call"


@dataclass(frozen=True)
class SideEffect:
    """An external write seen while replaying (a DB row, a file, an API call)."""

    target: str
    kind: SideEffectKind
    input_hash: str


@dataclass
class SideEffectLog:
    """Records external writes seen during a replay, in order."""

    effects: list[SideEffect] = field(default_factory=list)

    def record(self, target: str, kind: SideEffectKind, inp: object) -> None:
        """Append one observed side effect."""
        self.effects.append(SideEffect(target, kind, input_hash(inp)))

    def non_idempotent(self) -> list[SideEffect]:
        """Distinct effects whose (target, kind, input) occur more than once.

        The same write replayed twice is the signal for a non-idempotent effect.
        """
        counts: dict[tuple[str, str, str], int] = {}
        for e in self.effects:
            key = (e.target, e.kind.value, e.input_hash)
            counts[key] = counts.get(key, 0) + 1
        out: list[SideEffect] = []
        emitted: set[tuple[str, str, str]] = set()
        for e in self.effects:
            key = (e.target, e.kind.value, e.input_hash)
            if counts[key] > 1 and key not in emitted:
                out.append(e)
                emitted.add(key)
        return out

    def conflicting_targets(self) -> list[str]:
        """Targets written with two or more distinct inputs (write-write risk)."""
        by_target: dict[str, set[str]] = {}
        for e in self.effects:
            by_target.setdefault(e.target, set()).add(e.input_hash)
        return sorted(t for t, inputs in by_target.items() if len(inputs) > 1)
