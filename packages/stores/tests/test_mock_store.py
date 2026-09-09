"""Mock store integration tests (guide section 4): versioning, keyed lookup,
success/failure flag, hash determinism, and concurrent version assignment.
"""

from __future__ import annotations

import threading

from pytracer_stores import MockStore, input_hash
from pytracer_stores.config import PostgresConfig


def test_hash_is_order_independent() -> None:
    assert input_hash({"a": 1, "b": 2}) == input_hash({"b": 2, "a": 1})
    assert input_hash({"a": 1}) != input_hash({"a": 2})


def test_record_and_latest_lookup(mock_store: MockStore) -> None:
    rec = mock_store.record("search", {"q": "hi"}, ["result"])
    assert rec.version == 1
    assert rec.is_failure is False
    got = mock_store.lookup("search", {"q": "hi"})
    assert got is not None
    assert got.output == ["result"]


def test_versioning_same_key_multiple_outcomes(mock_store: MockStore) -> None:
    mock_store.record("search", {"q": "hi"}, ["v1"], scenario="happy")
    mock_store.record("search", {"q": "hi"}, {"error": "timeout"}, is_failure=True, scenario="sad")
    versions = mock_store.list_versions("search", {"q": "hi"})
    assert [v.version for v in versions] == [1, 2]
    # Latest wins by default.
    assert mock_store.lookup("search", {"q": "hi"}).version == 2  # type: ignore[union-attr]
    # Specific version and scenario selection.
    assert mock_store.lookup("search", {"q": "hi"}, version=1).output == ["v1"]  # type: ignore[union-attr]
    assert mock_store.lookup("search", {"q": "hi"}, scenario="happy").version == 1  # type: ignore[union-attr]


def test_failure_flag_filtering(mock_store: MockStore) -> None:
    mock_store.record("call", {"x": 1}, "ok", is_failure=False)
    mock_store.record("call", {"x": 1}, "boom", is_failure=True)
    failure = mock_store.lookup("call", {"x": 1}, is_failure=True)
    success = mock_store.lookup("call", {"x": 1}, is_failure=False)
    assert failure is not None and failure.output == "boom"
    assert success is not None and success.output == "ok"


def test_lookup_miss_returns_none(mock_store: MockStore) -> None:
    assert mock_store.lookup("nope", {"q": "absent"}) is None


def test_different_inputs_are_separate_keys(mock_store: MockStore) -> None:
    mock_store.record("t", {"q": "a"}, 1)
    mock_store.record("t", {"q": "b"}, 2)
    assert mock_store.lookup("t", {"q": "a"}).output == 1  # type: ignore[union-attr]
    assert mock_store.lookup("t", {"q": "b"}).output == 2  # type: ignore[union-attr]
    assert len(mock_store.list_versions("t", {"q": "a"})) == 1


def test_concurrent_records_assign_contiguous_versions(mock_store: MockStore) -> None:
    cfg = PostgresConfig()
    n = 20
    errors: list[Exception] = []

    def worker(i: int) -> None:
        store = MockStore(cfg, table=mock_store._table)  # type: ignore[attr-defined]
        try:
            store.record("busy", {"k": "same"}, i)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            store.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    versions = mock_store.list_versions("busy", {"k": "same"})
    assert sorted(v.version for v in versions) == list(range(1, n + 1))  # contiguous, no gaps/dupes
