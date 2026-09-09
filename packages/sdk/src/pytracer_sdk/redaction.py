"""Local secret / PII redaction, applied at capture time.

Runs on span input/output *before* serialization, so nothing sensitive ever
reaches the network layer. The ruleset is intentionally conservative regex -
API keys, emails, and card numbers - and is applied recursively to nested
structures. Redacted values are replaced with a stable placeholder so the
shape of the data is preserved for debugging.
"""

from __future__ import annotations

import re
from re import Pattern
from typing import Any

PLACEHOLDER = "[redacted:{label}]"
MAX_DEPTH = 20  # guard against pathological / cyclic structures


class RedactionRule:
    """A named regex whose matches are replaced with a labeled placeholder."""

    __slots__ = ("label", "pattern")

    def __init__(self, label: str, pattern: str | Pattern[str]) -> None:
        self.label = label
        self.pattern: Pattern[str] = (
            re.compile(pattern) if isinstance(pattern, str) else pattern
        )

    def apply(self, text: str) -> str:
        return self.pattern.sub(PLACEHOLDER.format(label=self.label), text)


# Order matters: more specific patterns first.
DEFAULT_RULES: list[RedactionRule] = [
    # Common provider API keys (OpenAI sk-..., Anthropic, GitHub, AWS, Slack, generic bearer).
    RedactionRule("api_key", r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    RedactionRule("api_key", r"\b(?:gh[pousr]|github_pat)_[A-Za-z0-9_]{20,}\b"),
    RedactionRule("api_key", r"\bAKIA[0-9A-Z]{16}\b"),
    RedactionRule("api_key", r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    RedactionRule("credit_card", r"\b(?:\d[ -]?){13,19}\b"),
    RedactionRule("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
]


class Redactor:
    """Applies a set of redaction rules to arbitrary JSON-like values."""

    def __init__(self, rules: list[RedactionRule] | None = None, *, enabled: bool = True) -> None:
        self.rules = DEFAULT_RULES if rules is None else rules
        self.enabled = enabled

    def redact_text(self, text: str) -> str:
        if not self.enabled:
            return text
        for rule in self.rules:
            text = rule.apply(text)
        return text

    def redact(self, value: Any, _depth: int = 0) -> Any:
        """Recursively redact strings inside dicts, lists, tuples, and scalars."""
        if not self.enabled or _depth > MAX_DEPTH:
            return value
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, dict):
            return {k: self.redact(v, _depth + 1) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            redacted = [self.redact(v, _depth + 1) for v in value]
            return type(value)(redacted) if isinstance(value, tuple) else redacted
        # Non-string scalars (int, float, bool, None) carry no textual secrets.
        return value
