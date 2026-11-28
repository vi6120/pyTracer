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
        cut_point: bool = False,
    ) -> Any: ...


def _tail(text: str, limit: int = 2000) -> str:
    """The last ``limit`` characters of ``text`` (so the real error is not cut off)."""
    text = (text or "").strip()
    return text if len(text) <= limit else "...\n" + text[-limit:]


def _no_results(which: str, ref: str, run: Any) -> str:
    """Why a run produced no results. A run stopped by the time limit says so
    first: its output ends wherever the agent was, which looks like any other
    failure."""
    if getattr(run, "timed_out", False):
        return (
            f"{which} run on {ref!r} timed out before it finished (raise the limit "
            f"with --timeout); its output up to then:\n{_tail(run.stderr)}"
        )
    return f"{which} run on {ref!r} produced no results:\n{_tail(run.stderr)}"


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
    cut_point: bool = False,
) -> tuple[Any, Any, dict[str, Any]]:
    """Reproduce the collection on both refs and classify each scenario.

    Returns (baseline_result, candidate_result, verdicts). Raises if either run
    produced no results (for example a failed dependency install); the error
    carries the run's stderr so the cause is visible, not truncated.
    """
    # Passed only when on, so a runner written before cut-point replay still works.
    extra: dict[str, Any] = {"cut_point": True} if cut_point else {}
    baseline = runner.reproduce(
        baseline_ref,
        collection_path,
        frozen_trace=frozen_trace,
        trace_dest=trace_dest,
        env_file=env_file,
        env_dest=env_dest,
        evaluate_changed_prompts=evaluate_changed_prompts,
        install=install,
        **extra,
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
        **extra,
    )
    for which, ref, run in (("baseline", baseline_ref, baseline), ("candidate", candidate_ref, candidate)):
        if run.results is None:
            raise RuntimeError(_no_results(which, ref, run))
    verdicts = classify(baseline.results, candidate.results)
    return baseline, candidate, verdicts


def format_verdicts(verdicts: dict[str, Any], reasons: dict[str, str] | None = None) -> str:
    """Render per-scenario cross-branch outcomes as a table, with the reason for each
    inconclusive outcome (``reasons``, from ``explain_reproduction``) so it is not
    read as a plain pass or fail; a general hint where no reason is known."""
    if not verdicts:
        return "no scenarios"
    lines = [f"{'SCENARIO':<28} OUTCOME"]
    values: set[str] = set()
    for name in sorted(verdicts):
        value = str(getattr(verdicts[name], "value", verdicts[name]))
        values.add(value)
        lines.append(f"{name[:28]:<28} {value}")
    reasons = reasons or {}
    if reasons:
        lines.append("")
        lines += [f"{name}: {reasons[name]}" for name in sorted(reasons)]
    unexplained = {
        str(getattr(verdicts[n], "value", verdicts[n])) for n in verdicts if n not in reasons
    }
    values &= unexplained
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
