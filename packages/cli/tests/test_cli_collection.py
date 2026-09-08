"""Collection loader: YAML and JSON, path resolution, validation."""

from __future__ import annotations

import json
from pathlib import Path

from navik_replay import ReplayMode

from navik_cli import load_collection


def test_loads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        "name: t\nscenarios:\n"
        "  - name: s\n    agent: m:a\n    trace: t.jsonl\n    mode: partial\n",
        encoding="utf-8",
    )
    collection, base = load_collection(path)
    assert collection.name == "t"
    assert collection.scenarios[0].mode is ReplayMode.PARTIAL
    assert base == tmp_path


def test_loads_json(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text(
        json.dumps(
            {"name": "t", "scenarios": [{"name": "s", "agent": "m:a", "trace": "t.jsonl"}]}
        ),
        encoding="utf-8",
    )
    collection, _ = load_collection(path)
    assert collection.scenarios[0].mode is ReplayMode.FULL  # default


def test_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("name: t\nbogus: 1\nscenarios: []\n", encoding="utf-8")
    try:
        load_collection(path)
        raise AssertionError("expected validation error")
    except ValueError:
        pass
