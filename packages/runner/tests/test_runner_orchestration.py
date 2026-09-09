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
