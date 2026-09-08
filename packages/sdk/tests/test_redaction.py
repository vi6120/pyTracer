"""Redaction: seeded fake secrets must never survive capture."""

from __future__ import annotations

import pytest

from navik_sdk.redaction import Redactor

SECRETS = [
    ("openai key", "sk-abc123DEF456ghi789JKL012mno", "sk-abc123"),
    ("github pat", "ghp_0123456789abcdefghijklmnopqrstuvwx", "ghp_"),
    ("aws key", "AKIAIOSFODNN7EXAMPLE", "AKIA"),
    ("email", "alice.smith@example.com", "@example.com"),
    ("credit card", "4111 1111 1111 1111", "4111"),
]


@pytest.mark.parametrize("label,secret,fragment", SECRETS)
def test_secret_is_removed_from_text(label: str, secret: str, fragment: str) -> None:
    out = Redactor().redact_text(f"here is the value: {secret} done")
    assert secret not in out
    assert fragment not in out
    assert "[redacted:" in out


def test_redaction_recurses_into_nested_structures() -> None:
    payload = {
        "messages": [
            {"role": "user", "content": "my key is sk-abc123DEF456ghi789JKL012mno"},
            {"role": "system", "email": "bob@corp.io"},
        ],
        "nested": {"deep": ["contact alice@example.com now"]},
    }
    out = Redactor().redact(payload)
    flat = str(out)
    assert "sk-abc123" not in flat
    assert "bob@corp.io" not in flat
    assert "alice@example.com" not in flat


def test_non_string_scalars_pass_through() -> None:
    payload = {"count": 42, "ratio": 0.5, "ok": True, "empty": None}
    assert Redactor().redact(payload) == payload


def test_redaction_can_be_disabled() -> None:
    secret = "sk-abc123DEF456ghi789JKL012mno"
    assert Redactor(enabled=False).redact_text(secret) == secret
