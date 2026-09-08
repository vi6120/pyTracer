"""Fork integrity: a fork is fully independent of its origin, both ways."""

from __future__ import annotations

from reg_helpers import artifact, scenario

from navik_registry import Registry, Visibility


def test_fork_records_lineage_and_new_owner() -> None:
    reg = Registry()
    origin = reg.create_collection(
        "alice", artifact("o", scenarios={"a": scenario("orig")}), visibility=Visibility.PUBLIC
    )
    fork = reg.fork("bob", origin.id)
    assert fork.owner == "bob"
    assert fork.visibility is Visibility.PRIVATE  # forks start private
    assert fork.forked_from is not None
    assert fork.forked_from.collection_id == origin.id


def test_fork_is_independent_both_directions() -> None:
    reg = Registry()
    origin = reg.create_collection(
        "alice", artifact("o", scenarios={"a": scenario("orig")}), visibility=Visibility.PUBLIC
    )
    fork = reg.fork("bob", origin.id)

    # Editing the fork does not touch the origin.
    reg.commit("bob", fork.id, artifact("o", scenarios={"a": scenario("changed")}), "change")
    assert reg.get("alice", origin.id).head.artifact.scenarios["a"] == scenario("orig")

    # Editing the origin does not touch the fork.
    reg.commit("alice", origin.id, artifact("o", scenarios={"a": scenario("orig2")}), "origin edit")
    assert reg.get("bob", fork.id).head.artifact.scenarios["a"] == scenario("changed")
