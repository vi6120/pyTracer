"""Access control, including indirect paths (get and discover both enforce)."""

from __future__ import annotations

import pytest
from reg_helpers import artifact, scenario

from pytracer_registry import AccessDenied, NotFoundError, Registry, Visibility


def test_private_is_invisible_to_others_everywhere() -> None:
    reg = Registry()
    col = reg.create_collection("alice", artifact("p"), visibility=Visibility.PRIVATE)
    # Direct get hides even the existence of the collection.
    with pytest.raises(NotFoundError):
        reg.get("bob", col.id)
    # Indirect path: discovery never lists it for others.
    assert reg.discover("bob") == []
    # Owner still sees it.
    assert reg.get("alice", col.id).id == col.id


def test_team_visible_and_editable_by_members_only() -> None:
    reg = Registry()
    col = reg.create_collection(
        "alice", artifact("t"), visibility=Visibility.TEAM, team=["bob"]
    )
    assert reg.get("bob", col.id).id == col.id
    with pytest.raises(NotFoundError):
        reg.get("carol", col.id)
    # Team member can edit.
    reg.commit("bob", col.id, artifact("t", scenarios={"a": scenario()}), "edit")
    assert len(reg.get("alice", col.id).versions) == 2


def test_public_viewable_by_anyone_but_editable_only_by_owner() -> None:
    reg = Registry()
    col = reg.create_collection("alice", artifact("pub"), visibility=Visibility.PUBLIC)
    assert reg.get(None, col.id).id == col.id  # anonymous viewer
    assert reg.get("bob", col.id).id == col.id
    with pytest.raises(AccessDenied):
        reg.commit("bob", col.id, artifact("pub", scenarios={"a": scenario()}), "not allowed")


def test_cannot_fork_what_you_cannot_see() -> None:
    reg = Registry()
    col = reg.create_collection("alice", artifact("p"), visibility=Visibility.PRIVATE)
    with pytest.raises(NotFoundError):
        reg.fork("bob", col.id)
