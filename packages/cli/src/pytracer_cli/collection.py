"""Test collection schema and loader.

A collection is a diffable YAML/JSON bundle of scenarios. Each scenario names an
agent entrypoint, a recorded trace to mock from, a replay mode, and declarative
assertions. Keeping it data (not code) is what lets collections be versioned,
forked, and shared later by the registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pytracer_replay import ReplayMode

# Assertion types expressible declaratively in a file (predicate / llm_judge
# need Python callables and are used programmatically, not from a collection).
DECLARATIVE_ASSERTIONS = frozenset(
    {"no_errors", "exact_match", "contains", "semantic_similarity"}
)


class AssertionSpec(BaseModel):
    """A declarative assertion (type plus the fields that type needs)."""

    model_config = ConfigDict(extra="forbid")

    type: str
    expected: Any = None
    needle: str | None = None
    threshold: float = 0.7
    name: str | None = None


class ScenarioSpec(BaseModel):
    """One test scenario: which agent to run, against which trace, and how to judge it."""

    model_config = ConfigDict(extra="forbid")

    name: str
    agent: str  # "module:callable"
    entry: dict[str, Any] = Field(default_factory=dict)
    trace: str  # path to a JSONL span recording, relative to the collection file
    mode: ReplayMode = ReplayMode.FULL
    assertions: list[AssertionSpec] = Field(default_factory=list)


class Collection(BaseModel):
    """A named bundle of scenarios, the unit the CLI runs and the registry versions."""

    model_config = ConfigDict(extra="forbid")

    name: str
    scenarios: list[ScenarioSpec]


def load_collection(path: str | Path) -> tuple[Collection, Path]:
    """Load and validate a collection file. Returns (collection, base_dir).

    ``base_dir`` is the collection file's directory, used to resolve the
    relative ``trace`` paths inside each scenario. YAML and JSON are both
    accepted (JSON is a subset of YAML).
    """
    file_path = Path(path).resolve()
    raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        # ValueError (not TypeError) so the CLI's config-error handler catches it
        # alongside pydantic's ValidationError.
        raise ValueError(f"collection must be a mapping, got {type(raw).__name__}")  # noqa: TRY004
    return Collection.model_validate(raw), file_path.parent
