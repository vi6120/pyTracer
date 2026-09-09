"""Access control at the collection level.

Shared collections can leak internal tool schemas, prompt structures, and
business logic, so every registry operation is gated by these checks (guide section 9).
A ``user`` of ``None`` is an anonymous viewer, who may only see public
collections.
"""

from __future__ import annotations

from .models import Collection, Visibility


def can_view(user: str | None, collection: Collection) -> bool:
    """True if ``user`` may view the collection (public, owner, or team member)."""
    if collection.visibility is Visibility.PUBLIC:
        return True
    if user is None:
        return False
    if user == collection.owner:
        return True
    return collection.visibility is Visibility.TEAM and user in collection.team


def can_edit(user: str | None, collection: Collection) -> bool:
    """True if ``user`` may commit to the collection (owner, or team member if team-shared)."""
    if user is None:
        return False
    if user == collection.owner:
        return True
    # Team members may edit a team-shared collection; public/private are owner-only.
    return collection.visibility is Visibility.TEAM and user in collection.team


def can_fork(user: str | None, collection: Collection) -> bool:
    """True if ``user`` may fork the collection (anything they can view)."""
    return can_view(user, collection)
