"""Execution backends: run a command in an isolated workspace.

``LocalExecutor`` runs the command in a subprocess (used for lightweight cases
and tests). ``DockerExecutor`` runs it in a fresh container per call, which is
the isolation the guide calls for so two branches with different dependency
versions never share an environment. Both read named artifact files back out of
the workspace after the run.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

TIMEOUT_EXIT_CODE = 124  # matches the coreutils `timeout` convention


def _as_text(value: object) -> str:
    """Coerce subprocess output (str, bytes, or None) to text."""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return ""


@dataclass
class ExecResult:
    """Outcome of running a command: exit code, output, and read-back artifacts."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    artifacts: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class Executor(Protocol):
    def run(
        self, workdir: Path, command: str, *, artifacts: list[str] | None = None
    ) -> ExecResult: ...


def _read_artifacts(workdir: Path, artifacts: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in artifacts:
        path = workdir / name
        if path.exists():
            out[name] = path.read_text(encoding="utf-8")
    return out


class LocalExecutor:
    """Runs the command in a subprocess in ``workdir`` (no container isolation)."""

    def __init__(self, *, timeout: float = 600.0) -> None:
        self._timeout = timeout

    def run(self, workdir: Path, command: str, *, artifacts: list[str] | None = None) -> ExecResult:
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecResult(
                TIMEOUT_EXIT_CODE, _as_text(exc.stdout), _as_text(exc.stderr), timed_out=True
            )
        return ExecResult(
            proc.returncode,
            proc.stdout,
            proc.stderr,
            artifacts=_read_artifacts(workdir, artifacts or []),
        )


class DockerExecutor:
    """Runs the command in a fresh, isolated container per call.

    The workspace is bind-mounted at ``/work``, so artifacts the command writes
    there are read back from the host after the container exits. ``--rm`` means
    every run starts from a clean container.
    """

    def __init__(
        self,
        image: str = "python:3.12-slim",
        *,
        timeout: float = 900.0,
        network: str | None = None,
        docker: str = "docker",
    ) -> None:
        self._image = image
        self._timeout = timeout
        self._network = network
        self._docker = docker

    def run(self, workdir: Path, command: str, *, artifacts: list[str] | None = None) -> ExecResult:
        argv = [self._docker, "run", "--rm", "-v", f"{workdir}:/work", "-w", "/work"]
        if self._network is not None:
            argv += ["--network", self._network]
        argv += [self._image, "sh", "-lc", command]
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=self._timeout, check=False
            )
        except subprocess.TimeoutExpired as exc:
            return ExecResult(
                TIMEOUT_EXIT_CODE, _as_text(exc.stdout), _as_text(exc.stderr), timed_out=True
            )
        return ExecResult(
            proc.returncode,
            proc.stdout,
            proc.stderr,
            artifacts=_read_artifacts(workdir, artifacts or []),
        )
