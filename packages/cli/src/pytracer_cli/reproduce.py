"""Cross-branch reproduction: back the ``pytracer reproduce`` command.

Given a collection and a candidate branch, this replays the collection (with its
frozen recorded trace) on a baseline ref and on the candidate ref in isolation,
then classifies each scenario as fixed, still failing, or diverged. The runner
and classifier come from ``pytracer-runner``; this module holds the orchestration
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
        env_file: str | None = None,
        env_dest: str = ".env",
        evaluate_changed_prompts: bool = False,
        install: bool | None = None,
    ) -> Any: ...


def _tail(text: str, limit: int = 2000) -> str:
    """The last ``limit`` characters of ``text`` (so the real error is not cut off)."""
    text = (text or "").strip()
    return text if len(text) <= limit else "...\n" + text[-limit:]


def run_reproduction(
    runner: _Runner,
    *,
    collection_path: str,
    baseline_ref: str,
    candidate_ref: str,
    classify: Callable[[Any, Any], dict[str, Any]],
    frozen_trace: str | None = None,
    trace_dest: str | None = None,
    env_file: str | None = None,
    env_dest: str = ".env",
    evaluate_changed_prompts: bool = False,
    install: bool | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Reproduce the collection on both refs and classify each scenario.

    Returns (baseline_result, candidate_result, verdicts). Raises if either run
    produced no results (for example a failed dependency install); the error
    carries the run's stderr so the cause is visible, not truncated.
    """
    baseline = runner.reproduce(
        baseline_ref,
        collection_path,
        frozen_trace=frozen_trace,
        trace_dest=trace_dest,
        env_file=env_file,
        env_dest=env_dest,
        evaluate_changed_prompts=evaluate_changed_prompts,
        install=install,
    )
    candidate = runner.reproduce(
        candidate_ref,
        collection_path,
        frozen_trace=frozen_trace,
        trace_dest=trace_dest,
        env_file=env_file,
        env_dest=env_dest,
        evaluate_changed_prompts=evaluate_changed_prompts,
        install=install,
    )
    if baseline.results is None:
        raise RuntimeError(
            f"baseline run on {baseline_ref!r} produced no results:\n{_tail(baseline.stderr)}"
        )
    if candidate.results is None:
        raise RuntimeError(
            f"candidate run on {candidate_ref!r} produced no results:\n{_tail(candidate.stderr)}"
        )
    verdicts = classify(baseline.results, candidate.results)
    return baseline, candidate, verdicts


def format_verdicts(verdicts: dict[str, Any]) -> str:
    """Render per-scenario cross-branch outcomes as a table, with a hint for the
    inconclusive outcomes so they are not read as a plain pass or fail."""
    if not verdicts:
        return "no scenarios"
    lines = [f"{'SCENARIO':<28} OUTCOME"]
    values: set[str] = set()
    for name in sorted(verdicts):
        value = str(getattr(verdicts[name], "value", verdicts[name]))
        values.add(value)
        lines.append(f"{name[:28]:<28} {value}")
    if "cannot_reproduce" in values:
        lines.append("")
        lines.append(
            "cannot_reproduce: the baseline left its recordings (a frozen call "
            "found none), so it never reproduced the failure; re-run with "
            "--env-file so both reach the bug."
        )
    if "cannot_judge" in values:
        lines.append("")
        lines.append(
            "cannot_judge: the candidate left its recordings (a frozen call found "
            "none, or ran live), so its outcome, pass or fail, is not a frozen "
            "result. If a prompt edit changed a model input, try reproduce "
            "--evaluate-changed-prompts."
        )
    return "\n".join(lines)


def all_fixed(verdicts: dict[str, Any]) -> bool:
    """True if every scenario's outcome is 'fixed' (candidate cleanly fixes all)."""
    return all(getattr(v, "value", v) == "fixed" for v in verdicts.values())
