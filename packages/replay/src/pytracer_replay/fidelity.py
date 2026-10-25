"""Replay fidelity helpers for high-faithfulness replay.

Deterministic replay (``engine.py``) injects recorded model and tool responses so
a run reproduces exactly. These helpers harden it for the cases that still drift:

  - **Streaming** - re-emit a recorded model response chunk by chunk, optionally
    with its captured timing (``RecordedStream``).
  - **Concurrency** - give each op call a ``CallKey`` so repeat and concurrent
    tool calls resolve to their recordings in issue order (sequential and async
    execution); same-input calls racing on different OS threads cannot be ordered
    from arrival alone and are flagged instead of silently swapped
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
import sys
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
    calls resolve to their recordings in sequence. Arrival order equals the order
    the agent issued the calls for **sequential** code and for **single-threaded
    async** (an ``asyncio`` event loop runs the interceptor in creation order), so
    those replay in order correctly. Concurrent calls whose inputs differ also
    resolve correctly, each to its own recording.

    Calls on **different OS threads** that share the same input reach the
    interceptor in a race-determined order. When they come from different callers
    (each thread's own ``@op``, called with its own input), :meth:`scoped_key_for`
    numbers them per caller, so each resolves to the recording made under that
    caller whatever order the threads arrive in. What remains is calls with an
    identical code path: same op, same input, under the same caller with the same
    input. Those cannot be told apart, so if their recorded responses differ the two
    can be swapped; when they are identical (the common case) order is irrelevant.
    :meth:`multi_thread_keys` reports the keys reached from several threads so the
    engine can surface exactly this case, not silently.
    """

    _counts: dict[tuple[str, ...], int] = field(default_factory=dict)
    _identifier_counts: dict[str, int] = field(default_factory=dict)
    # Threads that requested each key, to tell single-threaded (sequential/async,
    # deterministic) access apart from multi-threaded (race-ordered) access. A key
    # is (identifier, input_hash), or with the caller's (identifier, input_hash)
    # appended for a caller-scoped call.
    _threads: dict[tuple[str, ...], set[int]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def key_for(self, identifier: str, inp: object, *, track_threads: bool = True) -> CallKey:
        """Return the stable key for the next call of ``identifier`` with ``inp``.

        ``track_threads=False`` numbers the call without counting its thread, for a
        call that is resolved by its caller-scoped key instead.
        """
        ih = input_hash(inp)
        return CallKey(identifier, ih, self._next((identifier, ih), track_threads))

    def scoped_key_for(self, identifier: str, inp: object, parent: tuple[str, str]) -> CallKey:
        """Return the key for the next call of ``identifier`` with ``inp`` under
        the caller ``parent`` (its identifier and input hash)."""
        ih = input_hash(inp)
        return CallKey(identifier, ih, self._next((identifier, ih, *parent), True))

    def _next(self, key: tuple[str, ...], track_threads: bool) -> int:
        # Identify the thread by its live object, not threading.get_ident(): the OS
        # thread id is reused once a short thread exits, which would make two
        # genuinely different threads look like one.
        thread_id = id(threading.current_thread())
        with self._lock:
            n = self._counts.get(key, 0)
            self._counts[key] = n + 1
            if track_threads:
                self._threads.setdefault(key, set()).add(thread_id)
        return n

    def multi_thread_keys(self) -> set[tuple[str, ...]]:
        """Keys requested from more than one OS thread (race-ordered arrival)."""
        with self._lock:
            return {key for key, threads in self._threads.items() if len(threads) > 1}

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

    With ``freeze_time`` set, freezes ``time.time`` / ``time.time_ns`` /
    ``time.monotonic`` / ``perf_counter`` and ``datetime.datetime.now`` / ``utcnow``
    to that instant. With ``seed`` set, seeds the stdlib ``random`` module and makes
    ``uuid.uuid4`` draw from a seeded stream so ids repeat run to run.

    The wall-clock symbols (``time.time``, ``time.time_ns``, ``datetime.datetime``,
    ``uuid.uuid4``) are replaced wherever they are bound, so a module that did
    ``from datetime import datetime`` is pinned too, not only ``datetime.datetime``
    accessed through the module. ``datetime`` is swapped for a subclass whose
    ``isinstance`` / ``issubclass`` still answer against the real class, so pinning
    it process-wide does not break type checks. Restoration is guaranteed on exit.

    Run this only around a replay: it freezes the process clock while active, so
    code that waits on real time to advance will not. Third-party clocks and RNGs
    (numpy, ``os.urandom``) are out of scope.
    """
    restore: list[tuple[Any, str, Any]] = []

    def set_attr(target: Any, name: str, fresh: Any) -> None:
        restore.append((target, name, getattr(target, name)))
        setattr(target, name, fresh)

    def fixed(value: Any) -> Callable[[], Any]:
        return lambda: value

    # (original, replacement) for symbols callers also bind by name; replaced on
    # their home module and in every other module global that holds them.
    deep_pairs: list[tuple[Any, Any]] = []

    if controls.seed is not None:
        random.seed(controls.seed)
        # A separate stream so draining uuids does not perturb the agent's own RNG.
        uuid_rng = random.Random(controls.seed)

        def fake_uuid4() -> uuid.UUID:
            return uuid.UUID(int=uuid_rng.getrandbits(128), version=4)

        deep_pairs.append((uuid.uuid4, fake_uuid4))

    if controls.freeze_time is not None:
        frozen = controls.freeze_time
        frozen_ns = int(frozen * 1_000_000_000)
        frozen_aware = _datetime.datetime.fromtimestamp(frozen, tz=_datetime.timezone.utc)
        # Naive local wall-clock for that instant, matching what real now() returns.
        frozen_local = _datetime.datetime.fromtimestamp(frozen)  # noqa: DTZ006
        real_datetime = _datetime.datetime

        class _FrozenMeta(type):
            # Keep isinstance/issubclass answering against the real datetime, so
            # swapping the class in every module does not break type checks.
            def __instancecheck__(cls, obj: Any) -> bool:
                return isinstance(obj, real_datetime)

            def __subclasscheck__(cls, sub: type) -> bool:
                return issubclass(sub, real_datetime)

        class _FrozenDateTime(_datetime.datetime, metaclass=_FrozenMeta):
            # Overrides return the frozen instant; the base datetime return type is
            # intentional (the pinned value is a plain datetime, not this subclass).
            @classmethod
            def now(cls, tz: Any = None) -> _datetime.datetime:  # type: ignore[override]
                # now(None) -> naive local; now(tz) -> aware in tz (as stdlib does).
                return frozen_aware.astimezone(tz) if tz is not None else frozen_local

            @classmethod
            def utcnow(cls) -> _datetime.datetime:  # type: ignore[override]
                return frozen_aware.replace(tzinfo=None)

        deep_pairs.append((time.time, fixed(frozen)))
        deep_pairs.append((time.time_ns, fixed(frozen_ns)))
        deep_pairs.append((_datetime.datetime, _FrozenDateTime))
        # monotonic / perf_counter: pin on the time module only. They are rarely
        # imported by name, and deep-patching them could freeze timing internals
        # elsewhere in the process.
        set_attr(time, "monotonic", fixed(0.0))
        set_attr(time, "monotonic_ns", fixed(0))
        set_attr(time, "perf_counter", fixed(0.0))
        set_attr(time, "perf_counter_ns", fixed(0))

    if deep_pairs:
        by_id = {id(orig): (orig, repl) for orig, repl in deep_pairs}
        for module in list(sys.modules.values()):
            if module is None:
                continue
            try:
                members = list(vars(module).items())
            except Exception:  # noqa: BLE001 - some modules expose no writable dict
                continue
            for name, value in members:
                pair = by_id.get(id(value))
                if pair is not None and value is pair[0]:
                    try:
                        set_attr(module, name, pair[1])
                    except Exception:  # noqa: BLE001 - read-only attributes
                        restore.pop()  # set_attr recorded before setattr raised
    try:
        yield
    finally:
        for target, name, original in reversed(restore):
            try:
                setattr(target, name, original)
            except Exception:  # noqa: BLE001
                pass


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
