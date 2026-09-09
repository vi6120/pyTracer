"""Cross-branch runner and reproduction classification.

Ties a :class:`SourceProvider` (checkout a ref) and an :class:`Executor` (run a
command) together: each run happens in its own fresh temp workspace, so refs
never share state. ``reproduce`` runs a collection at a ref with a frozen trace
injected, and ``classify_reproduction`` turns two runs' results into the
fixed / still-failing / diverged verdicts that "try on this branch" reports.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pytracer_replay import Outcome

from .executor import ExecResult, Executor
from .source import SourceProvider

_RESULT_ARTIFACT = "pytracer-out.json"


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
        install: str = "pip install -e . >/dev/null 2>&1",
    ) -> ReproduceResult:
        """Run a collection at ``ref`` with the frozen trace injected.

        The command installs the ref's own dependencies (in its clean workspace),
        then runs ``pytracer run`` emitting JSON results, which are parsed back.
        """
        inject: dict[str, Path] = {}
        if frozen_trace is not None and trace_dest is not None:
            inject[trace_dest] = Path(frozen_trace)
        command = f"{install} && pytracer run {collection_path} --json {_RESULT_ARTIFACT}"
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

    For each scenario in the candidate run: if it now passes it is FIXED; if it
    still fails with the same failure fingerprint as the baseline it is
    STILL_FAILING; otherwise it is DIVERGED (a new, different failure).
    """
    base_by_name = {s["name"]: s for s in baseline.get("scenarios", [])}
    verdicts: dict[str, Outcome] = {}
    for scenario in candidate.get("scenarios", []):
        name = scenario["name"]
        if scenario.get("passed"):
            verdicts[name] = Outcome.FIXED
            continue
        base = base_by_name.get(name)
        fingerprint = scenario.get("fingerprint")
        if (
            base is not None
            and not base.get("passed")
            and fingerprint is not None
            and base.get("fingerprint") == fingerprint
        ):
            verdicts[name] = Outcome.STILL_FAILING
        else:
            verdicts[name] = Outcome.DIVERGED
    return verdicts
