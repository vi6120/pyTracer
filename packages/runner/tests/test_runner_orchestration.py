"""Orchestration: run in the checked-out tree, isolate each run, inject files."""

from __future__ import annotations

from pathlib import Path

from pytracer_runner import CrossBranchRunner, ExecResult


class FakeSource:
    """Records checkouts and writes a REF marker into each destination."""

    def __init__(self) -> None:
        self.checkouts: list[tuple[str, Path]] = []

    def checkout(self, ref: str, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        self.checkouts.append((ref, dest))
        (dest / "REF").write_text(ref, encoding="utf-8")


class FakeExecutor:
    """Records calls and emits a result artifact derived from the checked-out tree."""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, str, list[str]]] = []

    def run(self, workdir: Path, command: str, *, artifacts: list[str] | None = None) -> ExecResult:
        arts = artifacts or []
        self.calls.append((workdir, command, list(arts)))
        ref = (workdir / "REF").read_text(encoding="utf-8")  # proves we ran in the checkout
        (workdir / "pytracer-out.json").write_text(f'{{"ref":"{ref}"}}', encoding="utf-8")
        read = {a: (workdir / a).read_text(encoding="utf-8") for a in arts if (workdir / a).exists()}
        return ExecResult(0, f"ran for {ref}", "", artifacts=read)


def test_runs_inside_checked_out_tree() -> None:
    source, executor = FakeSource(), FakeExecutor()
    runner = CrossBranchRunner(source, executor)
    result = runner.run_ref("main", "cmd", artifacts=["pytracer-out.json"])
    assert result.exit_code == 0
    assert result.artifacts["pytracer-out.json"] == '{"ref":"main"}'
    assert source.checkouts[0][0] == "main"


def test_each_run_uses_a_fresh_isolated_workspace() -> None:
    source, executor = FakeSource(), FakeExecutor()
    runner = CrossBranchRunner(source, executor)
    runner.run_ref("a", "cmd")
    runner.run_ref("b", "cmd")
    dir_a, dir_b = executor.calls[0][0], executor.calls[1][0]
    assert dir_a != dir_b  # different workspace per run
    assert not dir_a.exists() and not dir_b.exists()  # cleaned up afterward


def test_inject_freezes_files_into_the_checkout(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen.jsonl"
    frozen.write_text("FROZEN-TRACE", encoding="utf-8")
    captured: dict[str, str] = {}

    class Source:
        def checkout(self, ref: str, dest: Path) -> None:
            dest.mkdir(parents=True, exist_ok=True)

    class Executor:
        def run(
            self, workdir: Path, command: str, *, artifacts: list[str] | None = None
        ) -> ExecResult:
            captured["trace"] = (workdir / "tests/traces/t.jsonl").read_text(encoding="utf-8")
            return ExecResult(0, "", "")

    runner = CrossBranchRunner(Source(), Executor())
    runner.run_ref("main", "cmd", inject={"tests/traces/t.jsonl": frozen})
    assert captured["trace"] == "FROZEN-TRACE"


def test_reproduce_parses_results_and_injects_trace(tmp_path: Path) -> None:
    frozen = tmp_path / "baseline.jsonl"
    frozen.write_text("[]", encoding="utf-8")
    injected: dict[str, str] = {}

    class Source:
        def checkout(self, ref: str, dest: Path) -> None:
            dest.mkdir(parents=True, exist_ok=True)

    class Executor:
        def run(
            self, workdir: Path, command: str, *, artifacts: list[str] | None = None
        ) -> ExecResult:
            injected["present"] = (workdir / "tests/t.jsonl").read_text(encoding="utf-8")
            payload = '{"collection":"c","scenarios":[{"name":"s","passed":true}]}'
            return ExecResult(0, "", "", artifacts={"pytracer-out.json": payload})

    runner = CrossBranchRunner(Source(), Executor())
    result = runner.reproduce(
        "feature", "tests/collection.yaml", frozen_trace=frozen, trace_dest="tests/t.jsonl"
    )
    assert result.ref == "feature"
    assert result.exit_code == 0
    assert result.results is not None
    assert result.results["scenarios"][0]["passed"] is True
    assert injected["present"] == "[]"


def test_reproduce_invokes_python_m_and_does_not_silence_install() -> None:
    captured: dict[str, str] = {}

    class Source:
        def checkout(self, ref: str, dest: Path) -> None:
            dest.mkdir(parents=True, exist_ok=True)

    class Executor:
        def run(
            self, workdir: Path, command: str, *, artifacts: list[str] | None = None
        ) -> ExecResult:
            captured["cmd"] = command
            return ExecResult(0, "", "", artifacts={"pytracer-out.json": '{"scenarios":[]}'})

    runner = CrossBranchRunner(Source(), Executor())
    runner.reproduce("main", "tests/collection.yaml")
    assert "python -m pytracer_cli run" in captured["cmd"]
    assert ">/dev/null" not in captured["cmd"]  # install errors stay visible


def test_reproduce_provisions_the_cli_by_default() -> None:
    # The CLI is installed into the ref's env so `python -m pytracer_cli` works even
    # when the project under test does not depend on it (X1); opt out with the flag.
    captured: dict[str, str] = {}

    class Source:
        def checkout(self, ref: str, dest: Path) -> None:
            dest.mkdir(parents=True, exist_ok=True)

    class Executor:
        def run(
            self, workdir: Path, command: str, *, artifacts: list[str] | None = None
        ) -> ExecResult:
            captured["cmd"] = command
            return ExecResult(0, "", "", artifacts={"pytracer-out.json": '{"scenarios":[]}'})

    runner = CrossBranchRunner(Source(), Executor())
    runner.reproduce("main", "tests/collection.yaml")
    assert "pytracer-cli" in captured["cmd"]  # hyphen = the pip provisioning step

    runner.reproduce("main", "tests/collection.yaml", provision_cli=False)
    assert "pytracer-cli" not in captured["cmd"]  # only `python -m pytracer_cli` remains


def test_reproduce_injects_env_file(tmp_path: Path) -> None:
    env = tmp_path / "prod.env"
    env.write_text("API_KEY=xyz\n", encoding="utf-8")
    seen: dict[str, str] = {}

    class Source:
        def checkout(self, ref: str, dest: Path) -> None:
            dest.mkdir(parents=True, exist_ok=True)

    class Executor:
        def run(
            self, workdir: Path, command: str, *, artifacts: list[str] | None = None
        ) -> ExecResult:
            seen["env"] = (workdir / ".env").read_text(encoding="utf-8")
            return ExecResult(0, "", "", artifacts={"pytracer-out.json": '{"scenarios":[]}'})

    runner = CrossBranchRunner(Source(), Executor())
    runner.reproduce("main", "tests/collection.yaml", env_file=env)
    assert seen["env"] == "API_KEY=xyz\n"  # the run's settings reached the clean checkout


def _command_for(isolated: bool, **kwargs: object) -> str:
    """The command reproduce builds for an executor that is (or is not) isolated."""
    captured: dict[str, str] = {}

    class Source:
        def checkout(self, ref: str, dest: Path) -> None:
            dest.mkdir(parents=True, exist_ok=True)

    class Executor:
        def run(
            self, workdir: Path, command: str, *, artifacts: list[str] | None = None
        ) -> ExecResult:
            captured["cmd"] = command
            return ExecResult(0, "", "", artifacts={"pytracer-out.json": '{"scenarios":[]}'})

    executor = Executor()
    executor.isolated = isolated  # type: ignore[attr-defined]
    CrossBranchRunner(Source(), executor).reproduce("main", "c.yaml", **kwargs)  # type: ignore[arg-type]
    return captured["cmd"]


def test_a_local_reproduction_installs_nothing_and_uses_this_interpreter() -> None:
    # Local runs happen in the caller's own environment: installing there would
    # re-point an editable install or downgrade pyTracer to a pin.
    import shlex
    import sys

    from pytracer_runner import LocalExecutor

    assert LocalExecutor.isolated is False
    command = _command_for(False)
    assert "pip install" not in command
    assert command.startswith(f"{shlex.quote(sys.executable)} -m pytracer_cli run c.yaml")


def test_a_docker_reproduction_installs_in_its_container() -> None:
    from pytracer_runner import DockerExecutor

    assert DockerExecutor.isolated is True
    command = _command_for(True)
    assert command.startswith("python -m pip install -e . -q && python -m pip install -q 'pytracer-cli")
    assert command.endswith("python -m pytracer_cli run c.yaml --json pytracer-out.json")


def test_install_can_be_forced_or_skipped_either_way() -> None:
    import shlex
    import sys

    assert "pip install" not in _command_for(True, install=False, provision_cli=False)
    local = _command_for(False, install=True)
    assert local.startswith(f"{shlex.quote(sys.executable)} -m pip install -e . -q && ")
    assert "pytracer-cli" not in local  # provisioning follows isolation unless asked
    assert _command_for(True, install="uv pip install -e .").startswith("uv pip install -e . && ")
