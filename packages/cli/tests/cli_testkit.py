"""Test helpers: record a trace and write a collection file to a temp dir."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pytracer_replay import MockSet, ReplayEngine, ReplayMode


def record_trace(agent: Any, entry: dict[str, Any], path: Path) -> None:
    """Run ``agent`` live and write its spans as a JSONL recording."""
    result = ReplayEngine(ReplayMode.LIVE).replay(lambda: agent(**entry), MockSet())
    path.write_text("\n".join(s.to_json() for s in result.spans), encoding="utf-8")


def write_collection(directory: Path, scenarios: list[dict[str, Any]], name: str = "kit") -> Path:
    """Write a collection YAML into ``directory`` and return its path."""
    path = directory / "collection.yaml"
    path.write_text(yaml.safe_dump({"name": name, "scenarios": scenarios}), encoding="utf-8")
    return path
