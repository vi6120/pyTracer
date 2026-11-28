"""Trace files compressed with zstd (.jsonl.zst): read and written transparently."""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

import pytracer_sdk as pytracer
from pytracer_replay import MockSet, ReplayEngine, ReplayMode
from pytracer_sdk import SpanKind

from pytracer_cli.main import app
from pytracer_cli.runner import ScenarioError, load_trace
from pytracer_cli.trace_io import read_trace_text, write_trace_text

runner = CliRunner()


def _have_zstd() -> bool:
    for name in ("compression.zstd", "zstandard"):
        try:
            __import__(name)
            return True
        except ImportError:
            continue
    return False


needs_zstd = pytest.mark.skipif(not _have_zstd(), reason='needs zstd: pip install "pytracer-cli[zstd]"')
POLICY = "Policy: authenticate the customer, confirm every write, never invent ids. " * 80


def _trace_text() -> str:
    """A recorded run that resends a long policy on every model call, as agents do."""

    @pytracer.op(kind=SpanKind.LLM, name="agent_llm")
    def model(messages: list[str]) -> str:
        return f"reply {len(messages)}"

    @pytracer.op(kind=SpanKind.AGENT, agent_name="support")
    def agent() -> str:
        history = [POLICY]
        for turn in range(8):
            history.append(f"customer turn {turn}")
            history.append(model(list(history)))
        return history[-1]

    spans = ReplayEngine(ReplayMode.LIVE).replay(agent, MockSet()).spans
    return "".join(s.to_json() + "\n" for s in spans)


@needs_zstd
def test_a_zst_trace_round_trips_and_is_much_smaller(tmp_path: Path) -> None:
    text = _trace_text()
    plain, packed = tmp_path / "run.jsonl", tmp_path / "run.jsonl.zst"
    write_trace_text(plain, text)
    write_trace_text(packed, text)
    assert read_trace_text(plain) == read_trace_text(packed) == text
    assert plain.stat().st_size > 10 * packed.stat().st_size


@needs_zstd
def test_replay_loads_a_zst_trace_like_a_plain_one(tmp_path: Path) -> None:
    packed = tmp_path / "run.jsonl.zst"
    write_trace_text(packed, _trace_text())
    plain = tmp_path / "run.jsonl"
    write_trace_text(plain, _trace_text())
    assert [s.name for s in load_trace(packed)] == [s.name for s in load_trace(plain)]
    assert len(load_trace(packed)) == 9


@needs_zstd
def test_record_writes_a_zst_trace_with_zstd(tmp_path: Path) -> None:
    trace = tmp_path / "captured.jsonl"
    trace.write_text(_trace_text(), encoding="utf-8")
    out = tmp_path / "suite"
    result = runner.invoke(
        app,
        ["record", "--from-file", str(trace), "--agent", "myapp:run", "--out", str(out), "--zstd"],
    )
    assert result.exit_code == 0, result.output
    scenario = yaml.safe_load((out / "collection.yaml").read_text())["scenarios"][0]
    assert scenario["trace"].endswith(".jsonl.zst")
    assert len(load_trace(out / scenario["trace"])) == 9


def test_a_zst_trace_without_zstd_says_how_to_get_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packed = tmp_path / "run.jsonl.zst"
    packed.write_bytes(b"any bytes: zstd is looked for before they are read")
    real_import = builtins.__import__

    def no_zstd(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in ("zstandard", "compression"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_zstd)
    with pytest.raises(ScenarioError, match=r"pytracer-cli\[zstd\]"):
        load_trace(packed)
