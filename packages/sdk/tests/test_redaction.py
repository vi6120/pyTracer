"""Redaction: seeded fake secrets must never survive capture."""

from __future__ import annotations

import pytest

from pytracer_sdk.redaction import Redactor

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


def test_credit_card_rule_ignores_ordinary_long_numbers() -> None:
    r = Redactor()
    # A Luhn-valid card number is still redacted.
    assert "[redacted:credit_card]" in r.redact_text("card 4242 4242 4242 4242")
    # Ordinary long numbers that are not Luhn-valid are left intact, so recorded
    # inputs/outputs (and thus replay mocks) are not corrupted.
    for ordinary in ("order 1111111111111111", "isbn 9780306406157", "ts 1700000000000"):
        assert r.redact_text(ordinary) == ordinary


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


def test_a_long_value_keeps_its_length_so_slicing_replays_the_same() -> None:
    # Code that truncates text (result[:N]) must cut a replayed, redacted value at
    # the same place it cut the original, or the replay sends another input.
    record = '{"email": "yusuf.rossi7110@example.com", "last_four": "2478"}, "orders": [1]}'
    redact = Redactor().redact_text
    out = redact(record)
    assert len(out) == len(record)
    assert "[redacted:email]" in out and "example.com" not in out
    # Cuts before the email and after it (a cut through it is the remaining edge).
    for cut in (8, 45, 60, len(record)):
        # Recorded: the original cut, then redacted. Replayed: the redacted value cut.
        assert redact(record[:cut]) == redact(out[:cut])


def test_a_value_shorter_than_the_placeholder_gets_the_bare_placeholder() -> None:
    assert Redactor().redact_text("a@b.co") == "[redacted:email]"


def test_length_preservation_can_be_turned_off() -> None:
    out = Redactor(preserve_length=False).redact_text("yusuf.rossi7110@example.com")
    assert out == "[redacted:email]"


def test_redacting_twice_changes_nothing() -> None:
    once = Redactor().redact_text(
        "mail yusuf.rossi7110@example.com, key sk-abc123DEF456ghi789JKL012mno"
    )
    assert Redactor().redact_text(once) == once


def _corpus() -> list[str]:
    """Seeded text with secrets, near misses and noise, for the pre-check tests."""
    import random

    rng = random.Random(20261128)
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_.@#:/{}\",+%"
    texts = [f"value: {secret} done" for _, secret, _ in SECRETS]
    texts += [
        "4242-4242-4242-4242", "4 2 4 2 4 2 4 2 4 2 4 2 4 2 4 2", "order #W2378156 on 2026-11-28",
        "sk-short", "ghp_tooShort", "github_pat_" + "a" * 30, "AKIAIOSFODNN7EXAMPL", "xoxb-1234567890",
        "user@host", "a@b.co", '{"email": "yusuf.rossi7110@example.com", "last_four": "2478"}',
    ]
    for _ in range(3000):
        noise = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 120)))
        if rng.random() < 0.3:
            secret = rng.choice(SECRETS)[1]
            cut = rng.randint(0, len(noise))
            noise = noise[:cut] + secret + noise[cut:]
        texts.append(noise)
    return texts


def test_pre_checks_never_change_what_is_redacted() -> None:
    from pytracer_sdk.redaction import DEFAULT_RULES, RedactionRule

    plain = [RedactionRule(r.label, r.pattern, validate=r.validate) for r in DEFAULT_RULES]
    for preserve in (True, False):
        fast, slow = Redactor(preserve_length=preserve), Redactor(plain, preserve_length=preserve)
        for text in _corpus():
            assert fast.redact_text(text) == slow.redact_text(text), text


def test_each_pre_check_holds_wherever_its_pattern_matches() -> None:
    from pytracer_sdk.redaction import DEFAULT_RULES

    for rule in DEFAULT_RULES:
        assert rule.may_match is not None
        for text in _corpus():
            for match in rule.pattern.finditer(text):
                assert rule.may_match(match.group(0)), (rule.label, match.group(0))
