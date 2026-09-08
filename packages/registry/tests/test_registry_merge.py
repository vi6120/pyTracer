"""Pull requests: diff, clean merge, and conflict on competing edits."""

from __future__ import annotations

import pytest
from reg_helpers import artifact, scenario

from navik_registry import AccessDenied, MergeConflict, Registry, Visibility, diff_artifacts


def _origin_and_fork(reg: Registry) -> tuple[str, str]:
    origin = reg.create_collection(
        "alice", artifact("m", scenarios={"a": scenario("base")}), visibility=Visibility.PUBLIC
    )
    fork = reg.fork("bob", origin.id)
    return origin.id, fork.id


def test_diff_added_removed_changed() -> None:
    a = artifact("x", scenarios={"a": scenario("1"), "b": scenario("2")})
    b = artifact("x", scenarios={"a": scenario("changed"), "c": scenario("3")})
    d = diff_artifacts(a, b)
    assert d["added"] == ["c"]
    assert d["removed"] == ["b"]
    assert d["changed"] == ["a"]


def test_clean_merge_of_nonoverlapping_changes() -> None:
    reg = Registry()
    origin_id, fork_id = _origin_and_fork(reg)
    # Fork adds b; origin adds c. No overlap -> clean merge.
    reg.commit("bob", fork_id, artifact("m", scenarios={"a": scenario("base"), "b": scenario("b")}), "add b")
    reg.commit("alice", origin_id, artifact("m", scenarios={"a": scenario("base"), "c": scenario("c")}), "add c")
    pr = reg.open_pull_request("bob", fork_id, origin_id, "add b")
    reg.merge_pull_request("alice", pr.id)
    merged = reg.get("alice", origin_id).head.artifact
    assert set(merged.scenarios) == {"a", "b", "c"}


def test_merge_conflict_on_competing_edits() -> None:
    reg = Registry()
    origin_id, fork_id = _origin_and_fork(reg)
    # Both edit scenario "a" differently -> conflict.
    reg.commit("bob", fork_id, artifact("m", scenarios={"a": scenario("bob")}), "bob edits a")
    reg.commit("alice", origin_id, artifact("m", scenarios={"a": scenario("alice")}), "alice edits a")
    pr = reg.open_pull_request("bob", fork_id, origin_id, "conflict")
    with pytest.raises(MergeConflict) as exc:
        reg.merge_pull_request("alice", pr.id)
    assert "a" in exc.value.conflicts


def test_merge_requires_edit_rights_on_target() -> None:
    reg = Registry()
    origin_id, fork_id = _origin_and_fork(reg)
    reg.commit("bob", fork_id, artifact("m", scenarios={"a": scenario("base"), "b": scenario("b")}), "add b")
    pr = reg.open_pull_request("bob", fork_id, origin_id, "add b")
    # Bob cannot merge into Alice's public collection (view yes, edit no).
    with pytest.raises(AccessDenied):
        reg.merge_pull_request("bob", pr.id)
