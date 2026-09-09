"""Formatting for the ``navik keys`` commands.

The command bodies live in ``main.py`` and wire these pure helpers to the
PostgreSQL ``ApiKeyStore``; keeping the formatting here lets it be tested without
a running store.
"""

from __future__ import annotations

from typing import Any


def format_key_table(records: list[Any]) -> str:
    """Render API-key records as a fixed-width table (never shows the secret)."""
    header = f"{'KEY ID':<10} {'PROJECT':<20} {'NAME':<16} {'STATUS':<8} CREATED"
    lines = [header]
    for r in records:
        status = "active" if r.active else "revoked"
        created = (
            r.created_at.strftime("%Y-%m-%d %H:%M")
            if hasattr(r.created_at, "strftime")
            else str(r.created_at)
        )
        lines.append(
            f"{r.key_id:<10} {r.project[:20]:<20} {(r.name or '-')[:16]:<16} "
            f"{status:<8} {created}"
        )
    return "\n".join(lines)


def format_new_key(record: Any, full_key: str) -> str:
    """Render the one-time output shown when a key is created."""
    return (
        f"key_id:  {record.key_id}\n"
        f"project: {record.project}\n"
        f"api key: {full_key}\n"
        "store this key now; only its hash is kept, so it cannot be shown again."
    )
