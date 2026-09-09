"""Cross-branch reproduction: back the ``navik reproduce`` command.

Given a collection and a candidate branch, this replays the collection (with its
frozen recorded trace) on a baseline ref and on the candidate ref in isolation,
then classifies each scenario as fixed, still failing, or diverged. The runner
and classifier come from ``navik-runner``; this module holds the orchestration
and formatting so it can be tested with a fake runner.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol


class _Runner(Protocol):
    def reproduce(
        self,
        ref: str,
        collection_path: str,
        *,
        frozen_trace: str | None = None,
        trace_dest: str | None = None,
    ) -> Any: ...


def run_reproduction(
    runner: _Runner,
    *,
    collection_path: str,
    baseline_ref: str,
    candidate_ref: str,
    classify: Callable[[Any, Any], dict[str, Any]],
    frozen_trace: str | None = None,
    trace_dest: str | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Reproduce the collection on both refs and classify each scenario.

    Returns (baseline_result, candidate_result, verdicts). Raises if either run
    produced no results (for example a failed dependency install).
    """
    baseline = runner.reproduce(
        baseline_ref, collection_path, frozen_trace=frozen_trace, trace_dest=trace_dest
    )
    candidate = runner.reproduce(
        candidate_ref, collection_path, frozen_trace=frozen_trace, trace_dest=trace_dest
    )
    if baseline.results is None:
        raise RuntimeError(f"baseline run on {baseline_ref!r} produced no results: {baseline.stderr[:200]}")
    if candidate.results is None:
        raise RuntimeError(f"candidate run on {candidate_ref!r} produced no results: {candidate.stderr[:200]}")
    verdicts = classify(baseline.results, candidate.results)
    return baseline, candidate, verdicts


def format_verdicts(verdicts: dict[str, Any]) -> str:
    """Render per-scenario cross-branch outcomes as a table."""
    if not verdicts:
        return "no scenarios"
    lines = [f"{'SCENARIO':<28} OUTCOME"]
    for name in sorted(verdicts):
        outcome = verdicts[name]
        lines.append(f"{name[:28]:<28} {getattr(outcome, 'value', outcome)}")
    return "\n".join(lines)


def all_fixed(verdicts: dict[str, Any]) -> bool:
    """True if every scenario's outcome is 'fixed' (candidate cleanly fixes all)."""
    return all(getattr(v, "value", v) == "fixed" for v in verdicts.values())
