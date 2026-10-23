"""A shared collection carries its recorded traces, so it stays runnable (C3)."""

from __future__ import annotations

from pytracer_registry import CollectionArtifact, Registry


def test_create_and_get_round_trips_embedded_traces() -> None:
    reg = Registry()
    art = CollectionArtifact(
        name="c",
        scenarios={"s1": {"agent": "m:a", "trace": "traces/t.jsonl", "mode": "full"}},
        traces={"traces/t.jsonl": '{"span": 1}\n'},
    )
    created = reg.create_collection("alice", art)
    got = reg.get("alice", created.id)
    assert got.head.artifact.traces == {"traces/t.jsonl": '{"span": 1}\n'}
