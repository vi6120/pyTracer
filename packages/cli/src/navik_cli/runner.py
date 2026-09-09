"""Run a collection: import agents, replay scenarios, evaluate assertions.

This is the shared core that both the local CLI and the CI wrapper call, so a
run behaves identically on a laptop and in a GitHub Action.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from navik_replay import (
    Assertion,
    AssertionResult,
    Contains,
    ExactMatch,
    MockSet,
    NoErrors,
    ReplayEngine,
    ReplayResult,
    SemanticSimilarity,
    all_passed,
    fingerprint,
    run_assertions,
)
from navik_sdk.schema import Span, SpanStatus

from .collection import AssertionSpec, Collection, ScenarioSpec, load_collection


class ScenarioError(RuntimeError):
    """Raised when a scenario is misconfigured (bad agent path, missing trace)."""


@dataclass
class ScenarioResult:
    """Outcome of one scenario: pass flag, replay result, and assertion results.

    ``config_error`` is set (and ``passed`` is False) when the scenario itself
    was misconfigured, e.g. a bad agent path or a missing trace file.
    """

    name: str
    passed: bool
    replay: ReplayResult
    assertions: list[AssertionResult] = field(default_factory=list)
    fingerprint: str | None = None
    config_error: str | None = None


@dataclass
class CollectionResult:
    """Aggregate outcome of running every scenario in a collection."""

    name: str
    scenarios: list[ScenarioResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(s.passed for s in self.scenarios)

    @property
    def counts(self) -> dict[str, int]:
        total = len(self.scenarios)
        failed = sum(1 for s in self.scenarios if not s.passed)
        return {"total": total, "passed": total - failed, "failed": failed}


def load_agent(path: str) -> Callable[..., object]:
    """Import an agent callable from a ``"module:attribute"`` path.

    The current working directory is put on the import path, so an agent module
    that lives in the project being tested is importable when ``navik run`` is
    invoked from that project (the console script does not add cwd by default).
    """
    module_name, sep, attr = path.partition(":")
    if not sep or not attr:
        raise ScenarioError(f"agent must be 'module:callable', got {path!r}")
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ScenarioError(f"could not import agent module {module_name!r}: {exc}") from exc
    try:
        agent = getattr(module, attr)
    except AttributeError as exc:
        raise ScenarioError(f"{module_name!r} has no attribute {attr!r}") from exc
    if not callable(agent):
        raise ScenarioError(f"agent {path!r} is not callable")
    return agent  # type: ignore[no-any-return]


def load_trace(path: Path) -> list[Span]:
    """Load a JSONL span recording (one Span per line)."""
    if not path.exists():
        raise ScenarioError(f"trace file not found: {path}")
    spans: list[Span] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            spans.append(Span.from_json(line))
    return spans


def build_assertion(spec: AssertionSpec) -> Assertion:
    """Map a declarative assertion spec to a replay assertion object."""
    name = spec.name
    if spec.type == "no_errors":
        return NoErrors(name=name or "no_errors")
    if spec.type == "exact_match":
        return ExactMatch(spec.expected, name=name or "exact_match")
    if spec.type == "contains":
        if spec.needle is None:
            raise ScenarioError("'contains' assertion requires 'needle'")
        return Contains(spec.needle, name=name or "contains")
    if spec.type == "semantic_similarity":
        if not isinstance(spec.expected, str):
            raise ScenarioError("'semantic_similarity' requires string 'expected'")
        return SemanticSimilarity(
            spec.expected, threshold=spec.threshold, name=name or "semantic_similarity"
        )
    raise ScenarioError(f"unsupported assertion type in collection: {spec.type!r}")


def run_scenario(scenario: ScenarioSpec, base_dir: Path) -> ScenarioResult:
    """Replay one scenario and evaluate its assertions."""
    try:
        agent = load_agent(scenario.agent)
        spans = load_trace(base_dir / scenario.trace)
        assertions = [build_assertion(a) for a in scenario.assertions]
    except ScenarioError as exc:
        empty = ReplayResult(return_value=None, status=SpanStatus.ERROR, mode=scenario.mode)
        return ScenarioResult(scenario.name, False, empty, [], None, str(exc))

    mocks = MockSet.from_trace(spans)
    result = ReplayEngine(scenario.mode).replay(lambda: agent(**scenario.entry), mocks)
    assertion_results = run_assertions(result, assertions)
    # A scenario passes when it did not error and every assertion passed.
    passed = not result.failed and all_passed(assertion_results)
    return ScenarioResult(
        name=scenario.name,
        passed=passed,
        replay=result,
        assertions=assertion_results,
        fingerprint=fingerprint(result),
    )


def run_collection(collection: Collection, base_dir: Path) -> CollectionResult:
    """Run every scenario in a collection and aggregate the results."""
    results = [run_scenario(s, base_dir) for s in collection.scenarios]
    return CollectionResult(collection.name, results)


def run_path(path: str | Path) -> CollectionResult:
    """Load a collection file and run it end to end."""
    collection, base_dir = load_collection(path)
    return run_collection(collection, base_dir)
