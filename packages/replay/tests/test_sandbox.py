"""Network sandbox: block outbound connections, allow loopback, restore cleanly.

These tests deliberately try to make replay code reach the network, mirroring
the guide's sandbox-escape testing, and confirm every path is blocked.
"""

from __future__ import annotations

import socket
import urllib.error
import urllib.request

import pytest
import pytracer_sdk as pytracer
from pytracer_sdk import SpanKind

from pytracer_replay import (
    BlockedNetworkError,
    MockSet,
    NetworkSandbox,
    ReplayEngine,
    ReplayMode,
)

# A non-routable TEST-NET address, so nothing ever leaves the machine even if a
# guard were missing (RFC 5737).
BLOCKED_HOST = "203.0.113.7"


def test_blocks_connection_by_hostname() -> None:
    with NetworkSandbox(), pytest.raises(BlockedNetworkError):
        socket.getaddrinfo("example.com", 80)


def test_blocks_raw_socket_connect_by_ip() -> None:
    with NetworkSandbox():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(BlockedNetworkError):
                s.connect((BLOCKED_HOST, 9))
        finally:
            s.close()


def test_blocks_urllib() -> None:
    with NetworkSandbox():
        with pytest.raises((BlockedNetworkError, urllib.error.URLError)) as exc:
            urllib.request.urlopen(f"http://{BLOCKED_HOST}/", timeout=2)
        # The block is the root cause even when urllib wraps it.
        assert isinstance(exc.value, BlockedNetworkError) or isinstance(
            exc.value.__cause__, BlockedNetworkError
        )


def test_loopback_is_allowed_by_default() -> None:
    # Allowed destinations pass the guard; connecting to a closed loopback port
    # fails with a normal connection error, not a BlockedNetworkError.
    with NetworkSandbox():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        try:
            with pytest.raises(OSError) as exc:
                s.connect(("127.0.0.1", 1))  # almost certainly closed
            assert not isinstance(exc.value, BlockedNetworkError)
        finally:
            s.close()


def test_explicit_allowlist_permits_host() -> None:
    with NetworkSandbox(allow=[BLOCKED_HOST]):
        # getaddrinfo for an IP literal does no DNS and is now allowed.
        socket.getaddrinfo(BLOCKED_HOST, 80)  # must not raise


def test_socket_is_restored_after_exit() -> None:
    original = socket.socket
    with NetworkSandbox():
        assert socket.socket is not original
    assert socket.socket is original


def test_engine_sandbox_blocks_unmocked_live_call() -> None:
    # In PARTIAL mode the model runs live; with the sandbox on, a model that
    # reaches the network is blocked and the replay is recorded as failed.
    @pytracer.op(kind=SpanKind.LLM)
    def model(x: str) -> str:
        urllib.request.urlopen(f"http://{BLOCKED_HOST}/", timeout=2)
        return "unreachable"

    @pytracer.op(kind=SpanKind.AGENT)
    def agent() -> str:
        return model("hi")

    result = ReplayEngine(ReplayMode.PARTIAL, sandbox=True).replay(agent, MockSet())
    assert result.failed


def test_engine_sandbox_allows_fully_mocked_replay() -> None:
    # In FULL mode the tool is mocked, so no network is attempted even with the
    # sandbox on.
    @pytracer.op(kind=SpanKind.TOOL, tool_name="fetch")
    def fetch() -> str:
        urllib.request.urlopen(f"http://{BLOCKED_HOST}/", timeout=2)  # would be blocked
        return "live"

    @pytracer.op(kind=SpanKind.AGENT)
    def agent() -> str:
        return fetch()

    # The recorded mock stands in for the tool, so its network call never runs.
    mocks = MockSet()
    mocks.add("fetch", {"args": [], "kwargs": {}}, "recorded")
    result = ReplayEngine(ReplayMode.FULL, sandbox=True).replay(agent, mocks)
    assert not result.failed
    assert result.return_value == "recorded"
