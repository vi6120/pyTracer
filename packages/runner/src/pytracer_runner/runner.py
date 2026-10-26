"""Cross-branch runner and reproduction classification.

Ties a :class:`SourceProvider` (checkout a ref) and an :class:`Executor` (run a
command) together: each run happens in its own fresh temp workspace, so refs
never share state. ``reproduce`` runs a collection at a ref with a frozen trace
injected, and ``classify_reproduction`` turns two runs' results into the
fixed / still-failing / diverged verdicts that "try on this branch" reports.
"""

from __future__ import annotations

import json
import shlex
import shutil
import sys
import tempfile
from collections.abc import Mapping
from importlib import metadata as importlib_metadata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pytracer_replay import Outcome

from .executor import ExecResult, Executor
from .source import SourceProvider

_RESULT_ARTIFACT = "pytracer-out.json"


def _cli_install_step(python: str) -> str:
    """Pip step that provisions pytracer-cli, pinned to the running version if known."""
    try:
        version = importlib_metadata.version("pytracer-cli")
    except importlib_metadata.PackageNotFoundError:
        return f"{python} -m pip install -q pytracer-cli"
    return f"{python} -m pip install -q 'pytracer-cli=={version}'"


@dataclass
class ReproduceResult:
    """Result of reproducing a collection at one ref."""

    ref: str
    exit_code: int
    results: dict[str, Any] | None  # parsed `pytracer run --json` output, or None if missing
    stderr: str = ""
    timed_out: bool = False


class CrossBranchRunner:
    """Runs commands against arbitrary refs, each in an isolated workspace."""

    def __init__(self, source: SourceProvider, executor: Executor) -> None:
        self._source = source
        self._executor = executor

    def run_ref(
        self,
        ref: str,
        command: str,
        *,
        artifacts: list[str] | None = None,
        inject: Mapping[str, Path] | None = None,
    ) -> ExecResult:
        """Check out ``ref`` into a fresh workspace, optionally inject files, run.

        ``inject`` maps a repo-relative destination path to a source file copied
        in after checkout - used to freeze the recorded trace/mocks so only the
        agent code varies between refs.
        """
        workdir = Path(tempfile.mkdtemp(prefix="pytracer-xbranch-"))
        try:
            self._source.checkout(ref, workdir)
            for dest_rel, src in (inject or {}).items():
                dest = workdir / dest_rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
            return self._executor.run(workdir, command, artifacts=artifacts or [])
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def reproduce(
        self,
        ref: str,
        collection_path: str,
        *,
        frozen_trace: str | Path | None = None,
        trace_dest: str | None = None,
        env_file: str | Path | None = None,
        env_dest: str = ".env",
        evaluate_changed_prompts: bool = False,
        install: str | bool | None = None,
        provision_cli: bool | None = None,
    ) -> ReproduceResult:
        """Run a collection at ``ref`` with the frozen trace injected.

        The command installs the ref's own dependencies (in its clean workspace),
        then runs the collection emitting JSON results, which are parsed back.
        ``env_file`` carries the run's settings (e.g. a ``.env``) into the clean
        checkout so both refs reach the bug instead of failing on configuration.
        ``provision_cli`` also installs ``pytracer-cli`` into the ref's environment -
        pinned to the running version - so ``python -m pytracer_cli`` works even when
        the project under test does not depend on it. Install output is not silenced,
        so a failure is visible in ``stderr``.

        ``install`` is the install command, ``True`` for ``pip install -e .``, or
        ``False`` to install nothing. By default (``None``) an isolated executor
        (Docker) installs the project and provisions the CLI in its throwaway
        container, while a local one installs nothing and runs with the current
        interpreter: its environment is the caller's own, which already has the
        project's dependencies, and installing there would change it (re-point an
        editable install at a temporary checkout, or downgrade pyTracer to a pin).
        """
        inject: dict[str, Path] = {}
        if frozen_trace is not None and trace_dest is not None:
            inject[trace_dest] = Path(frozen_trace)
        if env_file is not None:
            inject[env_dest] = Path(env_file)
        isolated = getattr(self._executor, "isolated", True)
        # Local runs use this interpreter, not whichever `python` is first on PATH.
        python = "python" if isolated else shlex.quote(sys.executable)
        if install is None:
            install = isolated
        if provision_cli is None:
            provision_cli = isolated
        steps: list[str] = []
        if install is True:
            steps.append(f"{python} -m pip install -e . -q")
        elif install:
            steps.append(install)
        if provision_cli:
            steps.append(_cli_install_step(python))
        # `python -m` works whenever pytracer-cli is importable, even if the
        # project does not put the `pytracer` script on PATH.
        run_cmd = f"{python} -m pytracer_cli run {collection_path} --json {_RESULT_ARTIFACT}"
        if evaluate_changed_prompts:
            run_cmd += " --evaluate-changed-prompts"
        steps.append(run_cmd)
        command = " && ".join(steps)
        result = self.run_ref(ref, command, artifacts=[_RESULT_ARTIFACT], inject=inject)
        parsed = (
            json.loads(result.artifacts[_RESULT_ARTIFACT])
            if _RESULT_ARTIFACT in result.artifacts
            else None
        )
        return ReproduceResult(
            ref=ref,
            exit_code=result.exit_code,
            results=parsed,
            stderr=result.stderr,
            timed_out=result.timed_out,
        )


def classify_reproduction(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Outcome]:
    """Per-scenario cross-branch verdicts from two `pytracer run --json` outputs.

    A verdict is only drawn from runs that stayed on their recordings, the same way
    whether the candidate passed or failed:

    - STILL_FAILING: the candidate fails exactly as the baseline did (the same
      fingerprint, or the same assertions failing the same way). The bug showed
      again, so this holds even if something else drifted.
    - CANNOT_REPRODUCE: the baseline left its recordings (a frozen call found no
      recording, or ran live), so it never reproduced the captured failure and
      there is nothing to compare.
    - CANNOT_JUDGE: the candidate left its recordings, passing or failing: its
      outcome came from calls that were not frozen, so it says nothing reliable
      about the fix. A model call evaluated live on request
      (``--evaluate-changed-prompts``) does not count as leaving them.
    - FIXED: it passes; DIVERGED: it fails differently.
    """
    base_by_name = {s["name"]: s for s in baseline.get("scenarios", [])}
    verdicts: dict[str, Outcome] = {}
    for scenario in candidate.get("scenarios", []):
        name = scenario["name"]
        base = base_by_name.get(name)
        if (
            base is not None
            and not base.get("passed")
            and not scenario.get("passed")
            and not _missed(base)
            and not _missed(scenario)
            and _same_failure(scenario, base)
        ):
            verdicts[name] = Outcome.STILL_FAILING
        elif base is not None and _left_recording(base):
            verdicts[name] = Outcome.CANNOT_REPRODUCE
        elif _left_recording(scenario):
            verdicts[name] = Outcome.CANNOT_JUDGE
        elif scenario.get("passed"):
            verdicts[name] = Outcome.FIXED
        else:
            verdicts[name] = Outcome.DIVERGED
    return verdicts


def explain_reproduction(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], verdicts: Mapping[str, Any]
) -> dict[str, str]:
    """Why each scenario has no verdict, in words a developer can act on.

    For ``cannot_judge`` it names where the candidate first left its recording and
    what kind of change that was; for ``cannot_reproduce``, where the baseline did.
    Only a changed model input gets the ``--evaluate-changed-prompts`` advice: a
    changed tool call, or one the recording never made, needs a re-recording.
    """
    base_by_name = {s["name"]: s for s in baseline.get("scenarios", [])}
    cand_by_name = {s["name"]: s for s in candidate.get("scenarios", [])}
    reasons: dict[str, str] = {}
    for name, verdict in verdicts.items():
        value = getattr(verdict, "value", verdict)
        if value == "cannot_judge":
            reasons[name] = _why(cand_by_name.get(name, {}), "the branch")
        elif value == "cannot_reproduce":
            reasons[name] = _why(base_by_name.get(name, {}), "the baseline") + (
                " The baseline never reproduced the captured failure: pass the run's "
                "settings with --env-file, or re-record from the baseline."
            )
    return reasons


_WHAT = {
    ("tool", "changed_input"): "calls tool '{op}' with arguments the recording does not have",
    ("tool", "extra_call"): "calls tool '{op}' more often than the recording does",
    ("tool", "not_recorded"): "calls tool '{op}', which the recording never called",
    ("llm", "changed_input"): "sends model '{op}' an input the recording does not have",
    ("llm", "extra_call"): "calls model '{op}' more often than the recording does",
    ("llm", "not_recorded"): "calls model '{op}', which the recording never called",
}


def _why(scenario: Mapping[str, Any], who: str) -> str:
    divergences = scenario.get("divergences") or []
    if not divergences:
        # Runs reported before divergences were recorded.
        changed = scenario.get("changed_models") or []
        if changed:
            return (
                f"A model input changed on {who} ({', '.join(changed)}). If it edits a "
                "prompt, re-run with --evaluate-changed-prompts to judge it live."
            )
        return f"{who[0].upper()}{who[1:]} left its recording, so its result is not frozen."
    first = divergences[0]
    kind, reason = str(first.get("kind")), str(first.get("reason"))
    what = _WHAT.get((kind, reason), "makes call {op}, which the recording cannot serve")
    sentence = f"{who[0].upper()}{who[1:]} {what.format(op=first.get('op'))}"
    if first.get("ran_live"):
        sentence += " (it ran live)"
    sentence += "."
    later = len(divergences) - 1
    if later:
        sentence += (
            f" {later} later call(s) left it too, after that one."
        )
    if who == "the branch":
        if (kind, reason) == ("llm", "changed_input"):
            sentence += (
                " If the branch edits that prompt, re-run with --evaluate-changed-prompts "
                "to judge it live."
            )
        else:
            sentence += (
                " A frozen replay cannot judge a change to the calls themselves; "
                "re-record the scenario on this branch to judge it."
            )
    return sentence


def _missed(scenario: Mapping[str, Any]) -> bool:
    """True when the run failed on a missing recording (not on the agent's bug)."""
    return scenario.get("error_type") == "MissingMockError"


def _left_recording(scenario: Mapping[str, Any]) -> bool:
    """True when a run did not stay on its recordings: a frozen call found no
    recording (and failed, or ran live). A model call evaluated live on request is
    a deliberate exception."""
    unrecorded = scenario.get("unrecorded_calls")
    evaluated = bool(scenario.get("evaluated_changed_prompts"))
    if unrecorded is not None:
        return any(not (evaluated and call.startswith("llm ")) for call in unrecorded)
    # Runs reported before these fields existed.
    changed = bool(scenario.get("changed_models")) and not evaluated
    return changed or _missed(scenario)


def _same_failure(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    """Whether two failed runs of a scenario failed the same way."""
    fingerprint = candidate.get("fingerprint")
    if fingerprint is not None or baseline.get("fingerprint") is not None:
        return fingerprint == baseline.get("fingerprint")
    # Nothing raised (a fingerprint identifies an error), so the failure is the
    # assertions: the same ones failing with the same detail is the same failure.
    # With the recordings frozen, the same bug produces the same detail.
    failed = _failed_assertions(candidate)
    return bool(failed) and failed == _failed_assertions(baseline)


def _failed_assertions(scenario: Mapping[str, Any]) -> list[tuple[Any, Any]]:
    return sorted(
        (str(a.get("name")), str(a.get("detail")))
        for a in scenario.get("assertions", [])
        if not a.get("passed")
    )


