"""Network sandbox for replay execution.

During a replay, tool and model calls are supposed to come from recorded mocks,
never the live network. This sandbox enforces that: while active, any outbound
socket connection to a host that is not on the allowlist raises
:class:`BlockedNetworkError`, so a replay can never accidentally trigger a real
external side effect (a real payment, a real email) unless explicitly permitted
(guide section 5).

It works by guarding the two chokepoints every stdlib and third-party client
goes through: name resolution (``socket.getaddrinfo``) and connection
(``socket.socket.connect`` / ``connect_ex``, via a guarded socket subclass).
Loopback is allowed by default so a local mock server or proxy still works.
"""

from __future__ import annotations

import socket
from collections.abc import Iterable
from types import TracebackType
from typing import Any

# Hosts that are always local; allowed by default so in-process test servers work.
_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})


class BlockedNetworkError(RuntimeError):
    """Raised when replay code attempts a non-allowlisted outbound connection."""


class NetworkSandbox:
    """Context manager that blocks outbound connections outside an allowlist.

    Parameters
    ----------
    allow:
        Hostnames or IP addresses that outbound connections may reach.
    allow_loopback:
        When True (default), loopback destinations are always permitted.
    """

    def __init__(self, allow: Iterable[str] = (), *, allow_loopback: bool = True) -> None:
        self._allow = set(allow)
        self._allow_loopback = allow_loopback
        self._orig_socket: type[socket.socket] | None = None
        self._orig_getaddrinfo: Any = None

    def _is_allowed(self, host: object) -> bool:
        if host is None or host == "":
            return True  # local bind / wildcard, not an outbound connection
        name = str(host)
        if name in self._allow:
            return True
        return self._allow_loopback and name in _LOOPBACK

    def _check(self, host: object) -> None:
        if not self._is_allowed(host):
            raise BlockedNetworkError(
                f"blocked outbound connection to {host!r} during replay "
                f"(add it to the sandbox allowlist to permit it)"
            )

    def __enter__(self) -> NetworkSandbox:  # noqa: PYI034
        sandbox = self
        self._orig_socket = socket.socket
        self._orig_getaddrinfo = socket.getaddrinfo
        base = self._orig_socket

        class _GuardedSocket(base):  # type: ignore[valid-type, misc]
            def connect(self, address: Any) -> None:
                sandbox._check(address[0] if isinstance(address, tuple) else address)
                super().connect(address)

            def connect_ex(self, address: Any) -> int:
                sandbox._check(address[0] if isinstance(address, tuple) else address)
                return super().connect_ex(address)  # type: ignore[no-any-return]

        def _guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
            sandbox._check(host)
            return sandbox._orig_getaddrinfo(host, *args, **kwargs)

        socket.socket = _GuardedSocket  # type: ignore[misc]
        socket.getaddrinfo = _guarded_getaddrinfo  # type: ignore[assignment]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._orig_socket is not None:
            socket.socket = self._orig_socket  # type: ignore[misc]
        if self._orig_getaddrinfo is not None:
            socket.getaddrinfo = self._orig_getaddrinfo
