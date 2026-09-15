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

from pytracer_sdk.schema import Span, SpanKind

from .hashing import canonical_json, input_hash

# Kinds that carry a mockable external effect.
MOCKABLE_KINDS: frozenset[SpanKind] = frozenset({SpanKind.TOOL, SpanKind.LLM})

# When several recordings share an identifier, a fuzzy candidate must be at least
# this similar to the requested input to be used, so replay never injects a wildly
# different response.
DEFAULT_FUZZY_THRESHOLD = 0.6


def _identifier(kind: SpanKind, name: str, tool_name: str | None) -> str:
    """Stable key for an op: tool name when present, else the span name."""
    return tool_name or name


@dataclass
class MockSet:
    """Maps (identifier, input_hash) -> recorded output, with a fuzzy fallback."""

    _outputs: dict[tuple[str, str], Any] = field(default_factory=dict)
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
                mocks.add(_identifier(span.kind, span.name, span.tool_name), span.input, span.output)
        return mocks

    def add(self, identifier: str, input: Any, output: Any) -> None:
        self._outputs[(identifier, input_hash(input))] = output
        self._by_identifier.setdefault(identifier, []).append((input, output))

    def lookup(
        self,
        identifier: str,
        input: Any,
        *,
        fuzzy: bool = False,
        threshold: float = DEFAULT_FUZZY_THRESHOLD,
    ) -> tuple[bool, Any]:
        """Return (hit, output).

        Exact match first. With ``fuzzy`` on, fall back to the recordings for this
        identifier when the exact key misses: the sole recording if there is only
        one, otherwise the nearest by input similarity if it clears ``threshold``.
        ``hit`` is False when nothing suitable was recorded.
        """
        key = (identifier, input_hash(input))
        if key in self._outputs:
            return True, self._outputs[key]
        if fuzzy:
            candidates = self._by_identifier.get(identifier)
            if candidates:
                found, output = self._best_candidate(input, candidates, threshold)
                if found:
                    return True, output
        return False, None

    @staticmethod
    def _best_candidate(
        input: Any, candidates: list[tuple[Any, Any]], threshold: float
    ) -> tuple[bool, Any]:
        """Pick a fallback output for ``input`` from an identifier's recordings."""
        if len(candidates) == 1:
            # Only one recording for this identifier: the drift is in the input,
            # so use it rather than falling through to a live call.
            return True, candidates[0][1]
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
        return len(self._outputs)
