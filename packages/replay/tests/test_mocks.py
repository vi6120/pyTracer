"""MockSet matching: exact by default, opt-in identifier/fuzzy fallback."""

from __future__ import annotations

from pytracer_replay import MockSet


def test_exact_match_hits() -> None:
    mocks = MockSet()
    mocks.add("search", {"q": "cats"}, ["a"])
    assert mocks.lookup("search", {"q": "cats"}) == (True, ["a"])


def test_exact_miss_without_fuzzy() -> None:
    mocks = MockSet()
    mocks.add("search", {"q": "cats"}, ["a"])
    assert mocks.lookup("search", {"q": "dogs"}) == (False, None)


def test_fuzzy_off_by_default() -> None:
    mocks = MockSet()
    mocks.add("search", {"q": "cats", "ts": 1}, ["a"])
    # A drifted input misses without fuzzy, preserving exact-match fidelity.
    assert mocks.lookup("search", {"q": "cats", "ts": 2}) == (False, None)


def test_fuzzy_single_candidate_fallback() -> None:
    mocks = MockSet()
    mocks.add("search", {"q": "cats", "ts": 1}, ["a"])
    # Only one recording for this identifier: use it despite the drifted input.
    assert mocks.lookup("search", {"q": "cats", "ts": 999}, fuzzy=True) == (True, ["a"])


def test_fuzzy_picks_nearest_among_many() -> None:
    mocks = MockSet()
    mocks.add("search", {"q": "cats"}, "CATS")
    mocks.add("search", {"q": "dogs"}, "DOGS")
    hit, output = mocks.lookup("search", {"q": "catss"}, fuzzy=True)
    assert hit and output == "CATS"


def test_fuzzy_below_threshold_misses() -> None:
    mocks = MockSet()
    mocks.add("search", {"q": "cats"}, "CATS")
    mocks.add("search", {"q": "dogs"}, "DOGS")
    # Nothing close among the candidates, and a strict threshold: no injection.
    hit, output = mocks.lookup(
        "search", {"unrelated": "a very different structure"}, fuzzy=True, threshold=0.95
    )
    assert (hit, output) == (False, None)


def test_fuzzy_unknown_identifier_misses() -> None:
    mocks = MockSet()
    mocks.add("search", {"q": "x"}, 1)
    assert mocks.lookup("other", {"q": "x"}, fuzzy=True) == (False, None)
