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

from pytracer_replay import (
    Assertion,
    AssertionResult,
    Contains,
    ExactMatch,
    MockSet,
    NoDuplicateWrites,
    NoErrors,
    NondeterminismControls,
    ReplayEngine,
    ReplayMode,
    ReplayResult,
    Returns,
    Route,
    SemanticSimilarity,
    StepCount,
    ToolsUsed,
    WriteCall,
    Writes,
    all_passed,
    fingerprint,
    run_assertions,
)
from pytracer_sdk.schema import Span, SpanKind, SpanStatus

from .collection import AssertionSpec, Collection, ScenarioSpec, load_collection
from .trace_io import read_trace_text


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
    # The recording the scenario replayed, so a report can read what a frozen call
    # cost when it was recorded (a replayed call costs nothing).
    recording: list[Span] = field(default_factory=list)


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
    that lives in the project being tested is importable when ``pytracer run`` is
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
    """Load a JSONL span recording (one Span per line), zstd-compressed if it ends in .zst."""
    if not path.exists():
        raise ScenarioError(f"trace file not found: {path}")
    try:
        text = read_trace_text(path)
    except RuntimeError as exc:  # a .zst trace with no zstd installed
        raise ScenarioError(str(exc)) from exc
    spans: list[Span] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            spans.append(Span.from_json(line))
    return spans


def _write_calls(expected: object) -> list[WriteCall]:
    if not isinstance(expected, list) or not expected:
        raise ScenarioError("'writes' requires 'expected' to be a list of {tool, times, args}")
    calls = []
    for item in expected:
        if isinstance(item, str):
            item = {"tool": item}
        if not isinstance(item, dict) or not isinstance(item.get("tool"), str):
            raise ScenarioError("each 'writes' entry needs a 'tool' name")
        unknown = set(item) - {"tool", "times", "args"}
        if unknown:
            raise ScenarioError(f"'writes' entry has unknown fields {sorted(unknown)}")
        times, args = item.get("times"), item.get("args")
        if times is not None and (not isinstance(times, int) or isinstance(times, bool) or times < 0):
            raise ScenarioError("'writes' 'times' must be a non-negative integer")
        if args is not None and not isinstance(args, dict):
            raise ScenarioError("'writes' 'args' must be a mapping of argument names to values")
        calls.append(WriteCall(item["tool"], times=times, args=args))
    return calls


def build_assertion(spec: AssertionSpec, writes: list[str] | None = None) -> Assertion:
    """Map a declarative assertion spec to a replay assertion object.

    ``writes`` is the scenario's declared write tools, which `writes` (with
    ``only``) and `no_duplicate_writes` use.
    """
    name = spec.name
    declared = list(writes or [])
    if spec.type == "no_errors":
        if spec.scope not in ("root", "any"):
            raise ScenarioError("'no_errors' scope must be 'root' or 'any'")
        return NoErrors(scope=spec.scope, name=name or "no_errors")
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
    if spec.type == "route":
        if not isinstance(spec.expected, list) or not all(isinstance(x, str) for x in spec.expected):
            raise ScenarioError("'route' requires 'expected' to be a list of agent names")
        return Route(spec.expected, name=name or "route")
    if spec.type == "tools_used":
        if not isinstance(spec.expected, list) or not all(isinstance(x, str) for x in spec.expected):
            raise ScenarioError("'tools_used' requires 'expected' to be a list of tool names")
        return ToolsUsed(spec.expected, exact=spec.exact, name=name or "tools_used")
    if spec.type == "step_count":
        if not isinstance(spec.expected, int) or isinstance(spec.expected, bool):
            raise ScenarioError("'step_count' requires an integer 'expected'")
        kind = None
        if spec.kind is not None:
            try:
                kind = SpanKind(spec.kind)
            except ValueError as exc:
                raise ScenarioError(f"'step_count' has unknown kind {spec.kind!r}") from exc
        return StepCount(spec.expected, kind=kind, name=name or "step_count")
    if spec.type == "writes":
        if spec.only and not declared:
            raise ScenarioError("'writes' with 'only' needs the scenario's 'writes' declared")
        return Writes(
            _write_calls(spec.expected), write_tools=declared, only=spec.only, name=name or "writes"
        )
    if spec.type == "no_duplicate_writes":
        tools = spec.tools if spec.tools is not None else declared
        if not tools:
            raise ScenarioError(
                "'no_duplicate_writes' needs 'tools' or the scenario's 'writes' declared"
            )
        return NoDuplicateWrites(list(tools), name=name or "no_duplicate_writes")
    if spec.type == "returns":
        if spec.expected is None:
            raise ScenarioError("'returns' requires 'expected'")
        return Returns(spec.expected, name=name or "returns")
    raise ScenarioError(f"unsupported assertion type in collection: {spec.type!r}")


def _effective(override: bool | None, scenario_value: bool | None, mode_default: bool) -> bool:
    """Resolve a safety flag: CLI override wins, then the scenario field, then the mode default."""
    if override is not None:
        return override
    if scenario_value is not None:
        return scenario_value
    return mode_default


def run_scenario(
    scenario: ScenarioSpec,
    base_dir: Path,
    *,
    strict: bool | None = None,
    sandbox: bool | None = None,
    evaluate_changed_prompts: bool | None = None,
    cut_point: bool | None = None,
) -> ScenarioResult:
    """Replay one scenario and evaluate its assertions.

    ``strict`` / ``sandbox`` are optional run-wide overrides (from the CLI). When
    unset they fall back to the scenario's own fields, then to the mode default:
    both on in ``full`` mode, off in ``partial`` / ``live``.

    ``evaluate_changed_prompts`` runs a frozen model op live when its input changed
    (a prompt edit) instead of failing, so the change is actually evaluated; it
    makes a live model call, so it also drops the network sandbox by default.

    ``cut_point`` (experimental) runs the first call that misses the recording, and
    every tool and model call after it, live; it also drops the sandbox by default.
    """
    try:
        overlap = set(scenario.live) & set(scenario.freeze)
        if overlap:
            raise ScenarioError(f"ops cannot be both live and frozen: {sorted(overlap)}")
        agent = load_agent(scenario.agent)
        spans = load_trace(base_dir / scenario.trace)
        assertions = [build_assertion(a, scenario.writes) for a in scenario.assertions]
    except ScenarioError as exc:
        empty = ReplayResult(return_value=None, status=SpanStatus.ERROR, mode=scenario.mode)
        return ScenarioResult(scenario.name, False, empty, [], None, str(exc))

    full = scenario.mode is ReplayMode.FULL
    strict_eff = _effective(strict, scenario.strict, full)
    evaluate_eff = _effective(
        evaluate_changed_prompts, scenario.evaluate_changed_prompts, False
    )
    cut_eff = _effective(cut_point, scenario.cut_point, False)
    # Ops listed in `live` are meant to run for real, so don't sandbox the network by
    # default when any op is live, or when a changed prompt is evaluated live, or the
    # step under test would be blocked.
    sandbox_default = full and not scenario.live and not evaluate_eff and not cut_eff
    sandbox_eff = _effective(sandbox, scenario.sandbox, sandbox_default)
    # Pin the clock / RNG only when the scenario asks, so most runs keep the real ones.
    controls = (
        NondeterminismControls(freeze_time=scenario.freeze_time, seed=scenario.seed)
        if scenario.freeze_time is not None or scenario.seed is not None
        else None
    )
    mocks = MockSet.from_trace(spans)
    result = ReplayEngine(
        scenario.mode,
        strict=strict_eff,
        sandbox=sandbox_eff,
        fuzzy_match=scenario.fuzzy,
        live_ops=scenario.live,
        frozen_ops=scenario.freeze,
        writes=scenario.writes,
        evaluate_changed_prompts=evaluate_eff,
        controls=controls,
        cut_point=cut_eff,
    ).replay(lambda: agent(**scenario.entry), mocks)
    if result.uninstrumented:
        # Recordings existed but the agent hit no @op seam, so nothing was injected
        # and it ran live. Fail loudly instead of a misleading green.
        return ScenarioResult(
            scenario.name,
            False,
            result,
            [],
            fingerprint(result),
            "the agent made no @op calls during replay, so nothing was injected and it "
            "ran live; if the run was captured via an adapter, note that replay needs "
            "@op instrumentation",
        )
    assertion_results = run_assertions(result, assertions)
    # A scenario passes when it did not error and every assertion passed.
    passed = not result.failed and all_passed(assertion_results)
    return ScenarioResult(
        name=scenario.name,
        passed=passed,
        replay=result,
        assertions=assertion_results,
        fingerprint=fingerprint(result),
        recording=spans,
    )


def run_collection(
    collection: Collection,
    base_dir: Path,
    *,
    strict: bool | None = None,
    sandbox: bool | None = None,
    evaluate_changed_prompts: bool | None = None,
    cut_point: bool | None = None,
) -> CollectionResult:
    """Run every scenario in a collection and aggregate the results."""
    results = [
        run_scenario(
            s, base_dir, strict=strict, sandbox=sandbox,
            evaluate_changed_prompts=evaluate_changed_prompts, cut_point=cut_point,
        )
        for s in collection.scenarios
    ]
    return CollectionResult(collection.name, results)


def run_path(
    path: str | Path,
    *,
    strict: bool | None = None,
    sandbox: bool | None = None,
    evaluate_changed_prompts: bool | None = None,
    cut_point: bool | None = None,
) -> CollectionResult:
    """Load a collection file and run it end to end."""
    collection, base_dir = load_collection(path)
    return run_collection(
        collection, base_dir, strict=strict, sandbox=sandbox,
        evaluate_changed_prompts=evaluate_changed_prompts, cut_point=cut_point,
    )
