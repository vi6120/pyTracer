"""Access control at the collection level.

Shared collections can leak internal tool schemas, prompt structures, and
business logic, so every registry operation is gated by these checks (guide §9).
A ``user`` of ``None`` is an anonymous viewer, who may only see public
collections.
"""

from __future__ import annotations

from .models import Collection, Visibility


def can_view(user: str | None, collection: Collection) -> bool:
    if collection.visibility is Visibility.PUBLIC:
        return True
    if user is None:
        return False
    if user == collection.owner:
        return True
    return collection.visibility is Visibility.TEAM and user in collection.team


def can_edit(user: str | None, collection: Collection) -> bool:
    if user is None:
        return False
    if user == collection.owner:
        return True
    # Team members may edit a team-shared collection; public/private are owner-only.
    return collection.visibility is Visibility.TEAM and user in collection.team


def can_fork(user: str | None, collection: Collection) -> bool:
    # You can fork anything you can see.
    return can_view(user, collection)
