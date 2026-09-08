"""Test helpers: build artifacts and scenario bodies."""

from __future__ import annotations

from typing import Any

from navik_registry import CollectionArtifact


def scenario(needle: str = "x") -> dict[str, Any]:
    return {
        "agent": "m:a",
        "trace": "t.jsonl",
        "mode": "full",
        "assertions": [{"type": "contains", "needle": needle}],
    }


def artifact(
    name: str = "c",
    *,
    framework: str | None = None,
    use_case: str | None = None,
    scenarios: dict[str, Any] | None = None,
) -> CollectionArtifact:
    return CollectionArtifact(
        name=name, framework=framework, use_case=use_case, scenarios=scenarios or {}
    )
