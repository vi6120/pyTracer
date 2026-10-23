"""MockSet - recorded op outputs keyed by (identifier, input hash).

A mock set is the deterministic substitute for live tool/model calls during
replay. It is built from a captured trace's recorded spans, or assembled by
hand for a test scenario.

Lookup is exact by default (identifier + input hash), which is the highest
fidelity. Real agents drift, though: a timestamp in a prompt, reordered kwargs,
or a whitespace change makes an input miss an otherwise-correct recording. So
lookup also supports an opt-in fallback: when the exact key misses, match by
identifier - use the sole recording for that identifier, or the nearest one by
input similarity - so replay stays deterministic through small, expected drift.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from pytracer_sdk.schema import Span, SpanKind, SpanStatus

from .hashing import canonical_json, input_hash

# Kinds that carry a mockable external effect.
MOCKABLE_KINDS: frozenset[SpanKind] = frozenset({SpanKind.TOOL, SpanKind.LLM})


@dataclass(frozen=True)
class RecordedError:
    """A recorded op failure. The engine re-raises this on replay so the agent's
    error path runs exactly as it did when captured, instead of a recorded failure
    being served as a (different, or successful) output.
    """

    error_type: str
    error_message: str | None = None

# When several recordings share an identifier, a fuzzy candidate must be at least
# this similar to the requested input to be used, so replay never injects a wildly
# different response.
DEFAULT_FUZZY_THRESHOLD = 0.6

# Positional matching serves the Nth recording for the Nth call, tolerating a model
# that re-words a tool's input each run. It still requires the input to be at least
# loosely related to the recording at that position (this floor), so a genuinely
# different call is reported as a miss rather than served the wrong recording. The
# floor is below the fuzzy threshold on purpose: positional is meant to survive
# heavier re-wording than fuzzy, but not a different question entirely.
POSITIONAL_MATCH_FLOOR = 0.4


def _identifier(kind: SpanKind, name: str, tool_name: str | None) -> str:
    """Stable key for an op: tool name when present, else the span name."""
    return tool_name or name


@dataclass
class MockSet:
    """Maps (identifier, input_hash) -> recorded outputs in capture order.

    Repeated calls with the same input return the recordings in sequence (the Nth
    call gets the Nth recording), so a tool called several times with the same
    input replays its distinct responses in order. Once the sequence is exhausted
    the last recording repeats. A fuzzy fallback handles drifted inputs.
    """

    _sequences: dict[tuple[str, str], list[Any]] = field(default_factory=dict)
    # identifier -> recordings in capture order, for the identifier/fuzzy fallback.
    _by_identifier: dict[str, list[tuple[Any, Any]]] = field(default_factory=dict)

    @classmethod
    def from_trace(
        cls, spans: Iterable[Span], *, kinds: frozenset[SpanKind] = MOCKABLE_KINDS
    ) -> MockSet:
        """Build a mock set from a trace's recorded tool/model spans."""
        mocks = cls()
        for span in spans:
            if span.kind in kinds:
                # A span that failed is recorded as an error to re-raise on replay,
                # not as its (empty) output served as a success.
                recorded: Any = (
                    RecordedError(span.error_type or "Exception", span.error_message)
                    if span.status is SpanStatus.ERROR
                    else span.output
                )
                mocks.add(_identifier(span.kind, span.name, span.tool_name), span.input, recorded)
        return mocks

    def add(self, identifier: str, input: Any, output: Any) -> None:
        self._sequences.setdefault((identifier, input_hash(input)), []).append(output)
        self._by_identifier.setdefault(identifier, []).append((input, output))

    def lookup(
        self,
        identifier: str,
        input: Any,
        *,
        occurrence: int = 0,
        positional_occurrence: int | None = None,
        fuzzy: bool = False,
        positional: bool = False,
        threshold: float = DEFAULT_FUZZY_THRESHOLD,
    ) -> tuple[bool, Any, str]:
        """Return ``(hit, output, match)``.

        ``match`` is ``"exact"``, ``"positional"``, ``"fuzzy"``, or ``"miss"``.
        Exact match first: the recording at ``occurrence`` for this
        (identifier, input). When the call count runs past the recorded sequence
        the result is a ``miss``, not the last recording again: an over-run is a
        real divergence (the agent called the op more often than it did when
        captured), and silently reusing a stale response would hide it. With
        ``positional`` on, fall back to this identifier's Nth recording by call
        order (for a live model that words a tool's input differently each run),
        where N is ``positional_occurrence`` (the per identifier call index) when
        given, else ``occurrence``. The positional recording is served only when
        the input is at least loosely related to it (``POSITIONAL_MATCH_FLOOR``);
        a genuinely different input, or an over-run past the recordings, is a miss,
        not the wrong recording served silently. With ``fuzzy`` on, fall back to
        the nearest recording for this identifier by input similarity if it clears
        ``threshold``. A non-exact match is reported so the caller can mark it.
        """
        seq = self._sequences.get((identifier, input_hash(input)))
        if seq is not None:
            if occurrence < len(seq):
                return True, seq[occurrence], "exact"
            return False, None, "miss"
        candidates = self._by_identifier.get(identifier)
        if positional and candidates:
            idx = occurrence if positional_occurrence is None else positional_occurrence
            if idx < len(candidates):
                candidate_input, candidate_output = candidates[idx]
                ratio = SequenceMatcher(
                    None, canonical_json(input), canonical_json(candidate_input)
                ).ratio()
                if ratio >= POSITIONAL_MATCH_FLOOR:
                    return True, candidate_output, "positional"
            # No recording at this position, or the one there is for a very
            # different input: a miss, not the wrong recording served silently.
            return False, None, "miss"
        if fuzzy and candidates:
            found, output = self._best_candidate(input, candidates, threshold)
            if found:
                return True, output, "fuzzy"
        return False, None, "miss"

    @staticmethod
    def _best_candidate(
        input: Any, candidates: list[tuple[Any, Any]], threshold: float
    ) -> tuple[bool, Any]:
        """Pick the nearest recording for ``input`` if it clears ``threshold``.

        The threshold applies even when there is a single recording, so a wildly
        different input is not served the sole recording of a different question.
        """
        target = canonical_json(input)
        best_ratio = -1.0
        best_output: Any = None
        for candidate_input, candidate_output in candidates:
            ratio = SequenceMatcher(None, target, canonical_json(candidate_input)).ratio()
            if ratio > best_ratio:
                best_ratio, best_output = ratio, candidate_output
        if best_ratio >= threshold:
            return True, best_output
        return False, None

    def __len__(self) -> int:
        return len(self._sequences)
