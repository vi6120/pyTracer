"""Bundle a collection and its recorded traces into one self-contained artifact,
and materialize one back into a runnable collection directory.

This is what makes a collection shared through the registry runnable: the traces
travel with the scenarios, so whoever pulls it does not end up with scenarios that
point at ``traces/*.jsonl`` files they never received. The bundle's shape matches
the registry's ``CollectionArtifact`` (name, scenarios keyed by name, and the
traces), so the registry service stores and versions the same self-contained form.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def bundle_collection(collection_path: str | Path) -> dict[str, Any]:
    """Read a collection file and its referenced traces into a self-contained bundle."""
    path = Path(collection_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"collection must be a mapping, got {type(data).__name__}")  # noqa: TRY004
    base = path.parent
    scenarios: dict[str, dict[str, Any]] = {}
    traces: dict[str, str] = {}
    for scenario in data.get("scenarios", []):
        name = scenario["name"]
        scenarios[name] = scenario
        rel = scenario.get("trace")
        if rel and rel not in traces:
            trace_file = base / rel
            if not trace_file.is_file():
                raise FileNotFoundError(f"trace file not found for scenario {name!r}: {trace_file}")
            traces[rel] = trace_file.read_text(encoding="utf-8")
    return {
        "name": data.get("name", path.stem),
        "description": data.get("description", ""),
        "scenarios": scenarios,
        "traces": traces,
    }


def materialize_collection(bundle: dict[str, Any], out_dir: str | Path) -> Path:
    """Write a bundle's scenarios and traces into a runnable directory.

    Returns the path to the written ``collection.yaml``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for rel, content in (bundle.get("traces") or {}).items():
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
    collection = {
        "name": bundle.get("name", "collection"),
        "scenarios": list((bundle.get("scenarios") or {}).values()),
    }
    collection_path = out / "collection.yaml"
    collection_path.write_text(yaml.safe_dump(collection, sort_keys=False), encoding="utf-8")
    return collection_path


def read_bundle(bundle_path: str | Path) -> dict[str, Any]:
    """Load a bundle artifact written as JSON."""
    data = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"bundle must be a JSON object, got {type(data).__name__}")  # noqa: TRY004
    return data


def write_bundle(bundle: dict[str, Any], bundle_path: str | Path) -> None:
    """Write a bundle artifact as JSON."""
    Path(bundle_path).write_text(json.dumps(bundle, indent=2), encoding="utf-8")
