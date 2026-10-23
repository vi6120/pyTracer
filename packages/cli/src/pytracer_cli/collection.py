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
    {
        "no_errors",
        "exact_match",
        "contains",
        "semantic_similarity",
        # Structural assertions on the span tree (how the agent got there):
        "route",  # ordered agents visited
        "tools_used",  # tools called
        "step_count",  # number of steps (optionally of one kind)
    }
)


class AssertionSpec(BaseModel):
    """A declarative assertion (type plus the fields that type needs)."""

    model_config = ConfigDict(extra="forbid")

    type: str
    expected: Any = None
    needle: str | None = None
    threshold: float = 0.7
    name: str | None = None
    # Structural assertions: `tools_used` uses `exact`; `step_count` uses `kind`
    # (agent / llm / tool / reasoning / handoff) to count one kind of span.
    exact: bool = False
    kind: str | None = None


class ScenarioSpec(BaseModel):
    """One test scenario: which agent to run, against which trace, and how to judge it."""

    model_config = ConfigDict(extra="forbid")

    name: str
    agent: str  # "module:callable"
    entry: dict[str, Any] = Field(default_factory=dict)
    trace: str  # path to a JSONL span recording, relative to the collection file
    mode: ReplayMode = ReplayMode.FULL
    # Fall back to fuzzy mock matching when an exact input misses (for agents
    # whose op inputs drift slightly between capture and replay).
    fuzzy: bool = False
    # Replay safety. ``strict`` fails a scenario on an unrecorded call instead of
    # silently running it live; ``sandbox`` blocks outbound network during replay.
    # ``None`` means use the mode default: both on in ``full`` mode (so a full
    # replay really touches no live services), off in ``partial`` / ``live``.
    strict: bool | None = None
    sandbox: bool | None = None
    # Selective freeze: `live` names ops to run live (freezing the rest of the run),
    # `freeze` names ops to serve from the recording (running the rest per mode).
    # Ops are matched by identifier (tool name, else op/agent name). Use `live` to
    # freeze a whole run except the one step under test.
    live: list[str] = Field(default_factory=list)
    freeze: list[str] = Field(default_factory=list)
    # Tools that mutate external state (by identifier). A live call to one is
    # logged as a write, so a repeat is flagged non-idempotent; other live tool
    # calls are logged as unclassified, so read-only tools are not falsely flagged.
    writes: list[str] = Field(default_factory=list)
    # Nondeterminism controls, for an agent that reads the clock or generates ids
    # or random numbers directly (not only through mocked ops). ``freeze_time``
    # pins ``time.*`` and ``datetime.now`` to that unix instant; ``seed`` seeds
    # ``random`` and ``uuid.uuid4``, so the run reproduces beyond the recording.
    freeze_time: float | None = None
    seed: int | None = None
    # Run a frozen model op live when its input changed (a prompt edit) instead of
    # failing, so the change is actually evaluated. Makes a live model call.
    evaluate_changed_prompts: bool = False
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
