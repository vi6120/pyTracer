"""Filesystem store: collections persist as JSON and reload intact."""

from __future__ import annotations

from pathlib import Path

from reg_helpers import artifact, scenario

from pytracer_registry import FilesystemStore, Registry, Visibility


def test_filesystem_persistence(tmp_path: Path) -> None:
    reg = Registry(FilesystemStore(tmp_path))
    col = reg.create_collection(
        "alice", artifact("f", scenarios={"a": scenario()}), visibility=Visibility.PUBLIC
    )
    reg.commit("alice", col.id, artifact("f", scenarios={"a": scenario(), "b": scenario("y")}), "add b")

    # A fresh registry over the same directory sees the persisted state.
    reg2 = Registry(FilesystemStore(tmp_path))
    reloaded = reg2.get("alice", col.id)
    assert len(reloaded.versions) == 2
    assert set(reloaded.head.artifact.scenarios) == {"a", "b"}

    # And it is stored as a diffable JSON file.
    assert (tmp_path / f"{col.id}.json").exists()


def test_in_memory_reads_are_isolated_copies(tmp_path: Path) -> None:
    reg = Registry()
    col = reg.create_collection("alice", artifact("i", scenarios={"a": scenario()}))
    snapshot = reg.get("alice", col.id)
    # Mutating a read-back object must not affect stored state.
    snapshot.versions.clear()
    assert len(reg.get("alice", col.id).versions) == 1
