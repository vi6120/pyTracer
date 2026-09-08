"""Versioning: history, lineage, content hashing."""

from __future__ import annotations

from reg_helpers import artifact, scenario

from navik_registry import Registry
from navik_registry.hashing import content_hash


def test_create_and_commit_build_history() -> None:
    reg = Registry()
    col = reg.create_collection("alice", artifact("s", scenarios={"a": scenario()}))
    assert len(col.versions) == 1
    assert col.head.parent_id is None

    v2 = reg.commit(
        "alice", col.id, artifact("s", scenarios={"a": scenario(), "b": scenario("y")}), "add b"
    )
    reloaded = reg.get("alice", col.id)
    assert len(reloaded.versions) == 2
    assert reloaded.head.id == v2.id
    assert reloaded.head.parent_id == col.head.id
    assert reloaded.head.content_hash != col.head.content_hash


def test_content_hash_matches_artifact() -> None:
    reg = Registry()
    art = artifact("s", scenarios={"a": scenario()})
    col = reg.create_collection("alice", art)
    assert col.head.content_hash == content_hash(art.model_dump())
