"""Source-checkout backends: materialize a repo at a given ref into a directory.

``GitSourceProvider`` produces an isolated working copy at an arbitrary branch
or commit via a local clone, so each cross-branch run starts from a clean tree.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Protocol


class CheckoutError(RuntimeError):
    """Raised when a ref cannot be materialized (bad ref, git failure)."""


class SourceProvider(Protocol):
    def checkout(self, ref: str, dest: Path) -> None: ...


class GitSourceProvider:
    """Clones a local repo and checks out ``ref`` into ``dest`` (isolated copy)."""

    def __init__(self, repo: str | Path, *, git: str = "git") -> None:
        self._repo = str(Path(repo).resolve())
        self._git = git

    def _run(self, *args: str) -> None:
        proc = subprocess.run(
            [self._git, *args], capture_output=True, text=True, check=False
        )
        if proc.returncode != 0:
            raise CheckoutError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")

    def checkout(self, ref: str, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        # Local clone gives an independent working tree; then pin it to the ref.
        self._run("clone", "--quiet", "--no-hardlinks", self._repo, str(dest))
        self._run("-C", str(dest), "checkout", "--quiet", ref)
