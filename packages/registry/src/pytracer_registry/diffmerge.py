"""Collection diffing and three-way merge.

Collections are diffed and merged at the scenario level (plus a few metadata
fields), which is what makes the fork / pull-request workflow behave like git:
non-overlapping changes merge cleanly, and a scenario changed differently on
both sides is a conflict.
"""

from __future__ import annotations

from typing import Any

from .models import CollectionArtifact

_META_FIELDS = ("name", "description", "framework", "use_case")


class MergeConflict(RuntimeError):
    """Raised when a three-way merge cannot be resolved automatically."""

    def __init__(self, conflicts: list[str]) -> None:
        super().__init__(f"merge conflict in: {', '.join(conflicts)}")
        self.conflicts = conflicts


def diff_artifacts(a: CollectionArtifact, b: CollectionArtifact) -> dict[str, Any]:
    """Structured diff of two artifacts: scenario add/remove/change + metadata."""
    a_keys, b_keys = set(a.scenarios), set(b.scenarios)
    changed = sorted(k for k in a_keys & b_keys if a.scenarios[k] != b.scenarios[k])
    meta_changed = [f for f in _META_FIELDS if getattr(a, f) != getattr(b, f)]
    return {
        "added": sorted(b_keys - a_keys),
        "removed": sorted(a_keys - b_keys),
        "changed": changed,
        "meta_changed": meta_changed,
    }


def _merge_map(
    base: dict[str, Any], ours: dict[str, Any], theirs: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Three-way merge of two dicts against a base. Returns (merged, conflicts)."""
    merged: dict[str, Any] = {}
    conflicts: list[str] = []
    for key in set(base) | set(ours) | set(theirs):
        b = base.get(key)
        o = ours.get(key)
        t = theirs.get(key)
        if o == t:
            if o is not None:  # both sides agree (both changed alike, or unchanged)
                merged[key] = o
            continue  # both deleted, or identical -> nothing/agree
        ours_changed = o != b
        theirs_changed = t != b
        if ours_changed and not theirs_changed:
            if o is not None:
                merged[key] = o  # take ours (theirs untouched); o is None -> our delete wins
        elif theirs_changed and not ours_changed:
            if t is not None:
                merged[key] = t  # take theirs
        else:  # both changed, and differently -> conflict
            conflicts.append(key)
    return merged, conflicts


def three_way_merge(
    base: CollectionArtifact, ours: CollectionArtifact, theirs: CollectionArtifact
) -> CollectionArtifact:
    """Merge ``theirs`` into ``ours`` using ``base`` as the common ancestor.

    Raises :class:`MergeConflict` if a scenario or metadata field was changed on
    both sides in different ways.
    """
    merged_scenarios, conflicts = _merge_map(base.scenarios, ours.scenarios, theirs.scenarios)

    meta_base = {f: getattr(base, f) for f in _META_FIELDS}
    meta_ours = {f: getattr(ours, f) for f in _META_FIELDS}
    meta_theirs = {f: getattr(theirs, f) for f in _META_FIELDS}
    merged_meta, meta_conflicts = _merge_map(meta_base, meta_ours, meta_theirs)
    conflicts.extend(f"meta:{c}" for c in meta_conflicts)

    if conflicts:
        raise MergeConflict(sorted(conflicts))

    return CollectionArtifact(
        name=merged_meta.get("name", ours.name),
        description=merged_meta.get("description", "") or "",
        framework=merged_meta.get("framework"),
        use_case=merged_meta.get("use_case"),
        scenarios=merged_scenarios,
    )
