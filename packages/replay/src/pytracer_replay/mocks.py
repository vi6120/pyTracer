"""MockSet - recorded op outputs keyed by (identifier, input hash).

A mock set is the deterministic substitute for live tool/model calls during
replay. It is built from a captured trace's recorded spans, or assembled by
hand for a test scenario.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from pytracer_sdk.schema import Span, SpanKind

from .hashing import input_hash

# Kinds that carry a mockable external effect.
MOCKABLE_KINDS: frozenset[SpanKind] = frozenset({SpanKind.TOOL, SpanKind.LLM})


def _identifier(kind: SpanKind, name: str, tool_name: str | None) -> str:
    """Stable key for an op: tool name when present, else the span name."""
    return tool_name or name


@dataclass
class MockSet:
    """Maps (identifier, input_hash) -> recorded output."""

    _outputs: dict[tuple[str, str], Any] = field(default_factory=dict)

    @classmethod
    def from_trace(
        cls, spans: Iterable[Span], *, kinds: frozenset[SpanKind] = MOCKABLE_KINDS
    ) -> MockSet:
        """Build a mock set from a trace's recorded tool/model spans."""
        outputs: dict[tuple[str, str], Any] = {}
        for span in spans:
            if span.kind in kinds:
                key = (_identifier(span.kind, span.name, span.tool_name), input_hash(span.input))
                outputs[key] = span.output
        return cls(outputs)

    def add(self, identifier: str, input: Any, output: Any) -> None:
        self._outputs[(identifier, input_hash(input))] = output

    def lookup(self, identifier: str, input: Any) -> tuple[bool, Any]:
        """Return (hit, output). ``hit`` is False when nothing was recorded."""
        key = (identifier, input_hash(input))
        if key in self._outputs:
            return True, self._outputs[key]
        return False, None

    def __len__(self) -> int:
        return len(self._outputs)
