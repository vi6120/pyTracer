"""MockSet matching: exact by default, opt-in positional and fuzzy fallback."""

from __future__ import annotations

from pytracer_replay import MockSet


def test_exact_match_hits() -> None:
    mocks = MockSet()
    mocks.add("search", {"q": "cats"}, ["a"])
    assert mocks.lookup("search", {"q": "cats"}) == (True, ["a"], "exact")


def test_exact_miss_without_fallback() -> None:
    mocks = MockSet()
    mocks.add("search", {"q": "cats"}, ["a"])
    assert mocks.lookup("search", {"q": "dogs"}) == (False, None, "miss")


def test_fuzzy_off_by_default() -> None:
    mocks = MockSet()
    mocks.add("search", {"q": "cats", "ts": 1}, ["a"])
    # A drifted input misses without fuzzy, preserving exact-match fidelity.
    assert mocks.lookup("search", {"q": "cats", "ts": 2}) == (False, None, "miss")


def test_fuzzy_single_candidate_similar_matches() -> None:
    mocks = MockSet()
    mocks.add("search", {"q": "cats", "ts": 1}, ["a"])
    # A similar drifted input (only the timestamp changed) clears the threshold.
    assert mocks.lookup("search", {"q": "cats", "ts": 999}, fuzzy=True) == (True, ["a"], "fuzzy")


def test_fuzzy_single_candidate_dissimilar_misses() -> None:
    mocks = MockSet()
    mocks.add("search", {"q": "cats"}, ["a"])
    # A wildly different input must NOT be served the sole recording (R9).
    assert mocks.lookup(
        "search", {"unrelated": "an entirely different question"}, fuzzy=True
    ) == (False, None, "miss")


def test_fuzzy_picks_nearest_among_many() -> None:
    mocks = MockSet()
    mocks.add("search", {"q": "cats"}, "CATS")
    mocks.add("search", {"q": "dogs"}, "DOGS")
    hit, output, match = mocks.lookup("search", {"q": "catss"}, fuzzy=True)
    assert hit and output == "CATS" and match == "fuzzy"


def test_fuzzy_below_threshold_misses() -> None:
    mocks = MockSet()
    mocks.add("search", {"q": "cats"}, "CATS")
    mocks.add("search", {"q": "dogs"}, "DOGS")
    hit, output, _ = mocks.lookup(
        "search", {"unrelated": "a very different structure"}, fuzzy=True, threshold=0.95
    )
    assert (hit, output) == (False, None)


def test_fuzzy_unknown_identifier_misses() -> None:
    mocks = MockSet()
    mocks.add("search", {"q": "x"}, 1)
    assert mocks.lookup("other", {"q": "x"}, fuzzy=True) == (False, None, "miss")


def test_positional_matches_by_call_order_regardless_of_input() -> None:
    # A live model words a tool's input differently each run; positional matching
    # serves the Nth recording for the Nth call, marked as a substitution.
    mocks = MockSet()
    mocks.add("search", {"q": "worded one way"}, "first")
    mocks.add("search", {"q": "worded another way"}, "second")
    assert mocks.lookup("search", {"q": "worded a third way"}, occurrence=0, positional=True) == (
        True,
        "first",
        "positional",
    )
    assert mocks.lookup("search", {"q": "yet another wording"}, occurrence=1, positional=True) == (
        True,
        "second",
        "positional",
    )


def test_sequence_returns_recordings_in_order() -> None:
    mocks = MockSet()
    mocks.add("counter", {"k": 1}, "a")
    mocks.add("counter", {"k": 1}, "b")
    mocks.add("counter", {"k": 1}, "c")
    assert mocks.lookup("counter", {"k": 1}, occurrence=0) == (True, "a", "exact")
    assert mocks.lookup("counter", {"k": 1}, occurrence=1) == (True, "b", "exact")
    assert mocks.lookup("counter", {"k": 1}, occurrence=2) == (True, "c", "exact")


def test_over_run_past_recorded_sequence_is_a_miss() -> None:
    mocks = MockSet()
    mocks.add("t", {"k": 1}, "a")
    mocks.add("t", {"k": 1}, "b")
    # Calling the op more times than it was recorded is a real divergence, not a
    # cue to replay the last recording again (which would silently hide it).
    assert mocks.lookup("t", {"k": 1}, occurrence=2) == (False, None, "miss")


def test_positional_over_run_is_a_miss() -> None:
    mocks = MockSet()
    mocks.add("search", {"q": "one"}, "first")
    mocks.add("search", {"q": "two"}, "second")
    # A third positional call has no recording to stand in for it.
    assert mocks.lookup(
        "search", {"q": "three"}, positional_occurrence=2, positional=True
    ) == (False, None, "miss")


def test_positional_rejects_a_different_id_but_allows_reworded_text() -> None:
    # Positional tolerates re-wording of the text, but a different non-text value
    # (a different book id) means a different call: a miss, not the recording there
    # served silently. Matching compares values, not the JSON skeleton (R12/V4).
    mocks = MockSet()
    mocks.add("lookup", {"book_id": 215, "query": "why did buck adapt to life in the wild"}, "ans")
    mocks.add("lookup", {"book_id": 84, "query": "who wrote this"}, "Mary Shelley")
    # Same id, re-worded question: still matches.
    assert mocks.lookup(
        "lookup",
        {"book_id": 215, "query": "what made buck adjust to living in the wild"},
        positional_occurrence=0,
        positional=True,
    ) == (True, "ans", "positional")
    # A different book id is a different question: a miss, not position 0 served.
    assert mocks.lookup(
        "lookup",
        {"book_id": 105, "query": "why did buck adapt to life in the wild"},
        positional_occurrence=0,
        positional=True,
    ) == (False, None, "miss")
    # A genuinely different call (different shape) is also a miss.
    assert mocks.lookup(
        "lookup", {"weather": "sunny", "city": "Paris"}, positional_occurrence=0, positional=True
    ) == (False, None, "miss")


def test_positional_matches_reworded_question_but_not_a_different_book() -> None:
    # The three inputs from the pilot re-test (R12/V4), in the captured input shape
    # {"args": [[book_ids], "question"], "kwargs": {...}}. Recording: Buck, book 215.
    mocks = MockSet()
    mocks.add(
        "search",
        {"args": [[215], "why did buck adapt to life in the wild"], "kwargs": {"k": 5}},
        "buck-results",
    )
    # 1. A re-worded Buck question, same book: matches.
    assert mocks.lookup(
        "search",
        {"args": [[215], "what made buck adjust to living in the wild"], "kwargs": {"k": 5}},
        positional=True,
    ) == (True, "buck-results", "positional")
    # 2. The Persuasion question, different book (105 vs 215): a miss.
    assert mocks.lookup(
        "search",
        {"args": [[105], "why did anne elliot break off her engagement"], "kwargs": {"k": 5}},
        positional=True,
    ) == (False, None, "miss")
    # 3. A one-character query in book 1: a miss.
    assert mocks.lookup(
        "search", {"args": [[1], "x"], "kwargs": {"k": 5}}, positional=True
    ) == (False, None, "miss")
