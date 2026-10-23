"""`pytracer registry export/import`: a shared collection stays runnable (C3)."""

from __future__ import annotations

from pathlib import Path

import cli_sample_agent
from cli_testkit import record_trace, write_collection
from typer.testing import CliRunner

from pytracer_cli.main import app
from pytracer_cli.registry import bundle_collection, materialize_collection

runner = CliRunner()


def _make_collection(tmp_path: Path) -> Path:
    record_trace(cli_sample_agent.agent, {"task": "find cats"}, tmp_path / "trace.jsonl")
    return write_collection(
        tmp_path,
        [
            {
                "name": "s1",
                "agent": "cli_sample_agent:agent",
                "entry": {"task": "find cats"},
                "trace": "trace.jsonl",
                "mode": "full",
                "assertions": [{"type": "contains", "needle": "summary"}],
            }
        ],
    )


def test_bundle_carries_the_scenarios_and_their_traces(tmp_path: Path) -> None:
    bundle = bundle_collection(_make_collection(tmp_path))
    assert "s1" in bundle["scenarios"]
    assert "trace.jsonl" in bundle["traces"]
    assert bundle["traces"]["trace.jsonl"].strip()  # the recorded spans travel with it


def test_materialize_writes_a_runnable_directory(tmp_path: Path) -> None:
    bundle = bundle_collection(_make_collection(tmp_path))
    out = tmp_path / "materialized"
    collection_path = materialize_collection(bundle, out)
    assert collection_path == out / "collection.yaml"
    assert (out / "trace.jsonl").is_file()


def test_export_then_import_produces_a_collection_that_runs(tmp_path: Path) -> None:
    collection = _make_collection(tmp_path)
    bundle_file = tmp_path / "bundle.json"
    exported = runner.invoke(app, ["registry", "export", str(collection), str(bundle_file)])
    assert exported.exit_code == 0, exported.stdout
    assert bundle_file.is_file()

    out = tmp_path / "pulled"
    imported = runner.invoke(app, ["registry", "import", str(bundle_file), str(out)])
    assert imported.exit_code == 0, imported.stdout

    # The materialized collection runs with no reference to files outside the bundle.
    ran = runner.invoke(app, ["run", str(out / "collection.yaml")])
    assert ran.exit_code == 0, ran.stdout
    assert "1/1 passed" in ran.stdout


def test_export_missing_file_exits_two(tmp_path: Path) -> None:
    res = runner.invoke(app, ["registry", "export", str(tmp_path / "nope.yaml"), str(tmp_path / "b.json")])
    assert res.exit_code == 2
