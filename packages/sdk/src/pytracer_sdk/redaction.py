"""Local secret / PII redaction, applied at capture time.

Runs on span input/output *before* serialization, so nothing sensitive ever
reaches the network layer. The ruleset is intentionally conservative regex -
API keys, emails, and card numbers - and is applied recursively to nested
structures. Redacted values are replaced with a stable placeholder so the
shape of the data is preserved for debugging.

The placeholder keeps the length of what it replaces when it can: a value at least
as long as ``[redacted:<label>]`` is padded with ``*`` to its own length. Code that
slices or truncates text (``result[:400]``) then cuts a replayed, redacted value at
the same place it cut the original, so the replay sends the same input as the
recording. A shorter value gets the bare placeholder. The cost is that the length
of a redacted value is kept; ``Redactor(preserve_length=False)`` drops it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from re import Pattern
from typing import Any

PLACEHOLDER = "[redacted:{label}]"
# Every placeholder starts with this, so replay can tell a redacted value apart.
MARKER = "[redacted:"
PAD = "*"
MAX_DEPTH = 20  # guard against pathological / cyclic structures


class RedactionRule:
    """A named regex whose matches are replaced with a labeled placeholder.

    An optional ``validate`` callable gates replacement: a match is redacted only
    when ``validate(match_text)`` is true. This lets a broad pattern (such as the
    card-number regex) avoid false positives on ordinary long numbers.
    """

    __slots__ = ("label", "pattern", "validate")

    def __init__(
        self,
        label: str,
        pattern: str | Pattern[str],
        *,
        validate: Callable[[str], bool] | None = None,
    ) -> None:
        self.label = label
        self.pattern: Pattern[str] = (
            re.compile(pattern) if isinstance(pattern, str) else pattern
        )
        self.validate = validate

    def apply(self, text: str, *, preserve_length: bool = False) -> str:
        placeholder = PLACEHOLDER.format(label=self.label)
        validate = self.validate

        def _repl(match: re.Match[str]) -> str:
            value = match.group(0)
            if validate is not None and not validate(value):
                return value
            if preserve_length and len(value) > len(placeholder):
                return placeholder + PAD * (len(value) - len(placeholder))
            return placeholder

        return self.pattern.sub(_repl, text)


def _luhn_ok(text: str) -> bool:
    """True if the digits in ``text`` form a Luhn-valid number of card length.

    Real card numbers (13-19 digits) satisfy the Luhn checksum, while ordinary
    long numbers such as order ids, ISBNs, and millisecond timestamps almost
    never do, so this removes the card rule's false positives.
    """
    digits = [int(c) for c in text if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


# Order matters: more specific patterns first.
DEFAULT_RULES: list[RedactionRule] = [
    # Common provider API keys (OpenAI sk-..., Anthropic, GitHub, AWS, Slack, generic bearer).
    RedactionRule("api_key", r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    RedactionRule("api_key", r"\b(?:gh[pousr]|github_pat)_[A-Za-z0-9_]{20,}\b"),
    RedactionRule("api_key", r"\bAKIA[0-9A-Z]{16}\b"),
    RedactionRule("api_key", r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    RedactionRule("credit_card", r"\b(?:\d[ -]?){13,19}\b", validate=_luhn_ok),
    RedactionRule("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
]


class Redactor:
    """Applies a set of redaction rules to arbitrary JSON-like values."""

    def __init__(
        self,
        rules: list[RedactionRule] | None = None,
        *,
        enabled: bool = True,
        preserve_length: bool = True,
    ) -> None:
        self.rules = DEFAULT_RULES if rules is None else rules
        self.enabled = enabled
        self.preserve_length = preserve_length

    def redact_text(self, text: str) -> str:
        if not self.enabled:
            return text
        for rule in self.rules:
            text = rule.apply(text, preserve_length=self.preserve_length)
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
