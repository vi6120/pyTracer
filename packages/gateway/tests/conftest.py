"""Pytest fixtures for gateway tests. Shared helpers live in ``helpers.py``."""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest


@pytest.fixture()
def wait_until() -> Callable[..., bool]:
    def _wait(pred: Callable[[], bool], timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if pred():
                return True
            time.sleep(0.02)
        return pred()

    return _wait
