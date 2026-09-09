"""End-to-end 'try on this branch' with a real local git repo.

Builds a throwaway repo where a bug is fixed on a feature branch, then uses the
runner to materialize each ref in isolation, run a check in it, and classify the
feature branch as having fixed the failure. Skipped when git is unavailable.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pytracer_replay import Outcome

from pytracer_runner import (
    CrossBranchRunner,
    GitSourceProvider,
    LocalExecutor,
    classify_reproduction,
)

GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(GIT is None, reason="git not available")

# Emits a pytracer-run-style result file based on whether bug.check() is fixed.
_RUN_CHECK = (
    "import json, bug\n"
    "ok = bug.check()\n"
    "json.dump({'collection': 'c', 'scenarios': ["
    "{'name': 'fixes-bug', 'passed': ok, 'fingerprint': None if ok else 'fp1'}]},"
    " open('pytracer-out.json', 'w'))\n"
)


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    )
    return proc.stdout.strip()


def _build_repo(root: Path) -> tuple[Path, str]:
    repo = root / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "tester")
    (repo / "run_check.py").write_text(_RUN_CHECK, encoding="utf-8")
    (repo / "bug.py").write_text("def check():\n    return False  # buggy\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "initial (buggy)")
    default_branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    # Fix the bug on a feature branch.
    _git(repo, "checkout", "-q", "-b", "feature")
    (repo / "bug.py").write_text("def check():\n    return True  # fixed\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "fix the bug")
    _git(repo, "checkout", "-q", default_branch)
    return repo, default_branch


def test_try_on_this_branch_reports_fixed(tmp_path: Path) -> None:
    repo, default_branch = _build_repo(tmp_path)
    runner = CrossBranchRunner(GitSourceProvider(repo), LocalExecutor())
    command = f"{shlex.quote(sys.executable)} run_check.py"

    baseline = runner.run_ref(default_branch, command, artifacts=["pytracer-out.json"])
    candidate = runner.run_ref("feature", command, artifacts=["pytracer-out.json"])

    base_json = json.loads(baseline.artifacts["pytracer-out.json"])
    cand_json = json.loads(candidate.artifacts["pytracer-out.json"])
    assert base_json["scenarios"][0]["passed"] is False  # fails on the buggy branch
    assert cand_json["scenarios"][0]["passed"] is True  # passes on the fixed branch

    verdict = classify_reproduction(base_json, cand_json)
    assert verdict == {"fixes-bug": Outcome.FIXED}


def test_checkouts_of_two_refs_are_independent(tmp_path: Path) -> None:
    repo, default_branch = _build_repo(tmp_path)
    provider = GitSourceProvider(repo)
    main_dir = tmp_path / "co_main"
    feat_dir = tmp_path / "co_feat"
    provider.checkout(default_branch, main_dir)
    provider.checkout("feature", feat_dir)
    assert "buggy" in (main_dir / "bug.py").read_text(encoding="utf-8")
    assert "fixed" in (feat_dir / "bug.py").read_text(encoding="utf-8")
