"""Discovery: visibility-aware listing, tag filters, and scale."""

from __future__ import annotations

import time

from reg_helpers import artifact

from pytracer_registry import Registry, Visibility


def test_discovery_respects_visibility_and_filters() -> None:
    reg = Registry()
    reg.create_collection(
        "alice", artifact("langgraph-rag", framework="langgraph", use_case="rag"),
        visibility=Visibility.PUBLIC,
    )
    reg.create_collection(
        "alice", artifact("crewai-tools", framework="crewai", use_case="tool-use"),
        visibility=Visibility.PUBLIC,
    )
    reg.create_collection(
        "alice", artifact("secret", framework="langgraph"), visibility=Visibility.PRIVATE
    )

    public = reg.discover("bob")
    assert {c.name for c in public} == {"crewai-tools", "langgraph-rag"}  # private excluded

    assert [c.name for c in reg.discover("bob", framework="langgraph")] == ["langgraph-rag"]
    assert [c.name for c in reg.discover("bob", use_case="rag")] == ["langgraph-rag"]
    assert reg.discover("bob", framework="nonexistent") == []


def test_discovery_scales_to_many_collections() -> None:
    reg = Registry()
    for i in range(1000):
        vis = Visibility.PUBLIC if i % 2 == 0 else Visibility.PRIVATE
        reg.create_collection("alice", artifact(f"c{i:04d}", framework="langgraph"), visibility=vis)

    start = time.perf_counter()
    results = reg.discover("bob", framework="langgraph")
    elapsed = time.perf_counter() - start

    assert len(results) == 500  # only the public half is visible to bob
    assert results == sorted(results, key=lambda c: c.name)  # stable ordering
    assert elapsed < 5.0, f"discovery too slow at 1k collections: {elapsed:.2f}s"
