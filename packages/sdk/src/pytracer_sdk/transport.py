"""Transports carry batches of spans off the agent's machine.

The buffer is decoupled from *where* spans go via this small protocol, so the
same buffering logic works against an in-memory sink (tests), a local file, or
the ingestion gateway over HTTP.
"""

from __future__ import annotations

import json
import ssl
import threading
import urllib.error
import urllib.request
from typing import Protocol, runtime_checkable

from .schema import Span


@runtime_checkable
class Transport(Protocol):
    """Anything that can accept a batch of spans."""

    def send(self, spans: list[Span]) -> None: ...


class InMemoryTransport:
    """Collects spans in a list. Used by tests and local debugging.

    Thread-safe: the buffer flushes from a background thread while assertions
    read from the main thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._spans: list[Span] = []

    def send(self, spans: list[Span]) -> None:
        with self._lock:
            self._spans.extend(spans)

    @property
    def spans(self) -> list[Span]:
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()


class TransportError(RuntimeError):
    """Raised when a batch could not be delivered to the gateway."""


class HTTPTransport:
    """Ships spans to the ingestion gateway's ``POST /v1/spans`` endpoint.

    Uses only the standard library (``urllib``) so the SDK stays dependency
    light. ``send`` runs on the buffer's background thread, so a synchronous
    POST here does not block the agent; the buffer also swallows a raised
    :class:`TransportError` so a gateway outage never kills the flush worker.

    TLS: point ``url`` at an ``https://`` gateway (terminate TLS at a reverse
    proxy in front of it) and the connection is encrypted with certificate and
    hostname verification via the system trust store. ``ca_bundle`` trusts a
    private/internal CA instead; ``verify=False`` disables verification and
    should only ever be used against a local test server.
    """

    def __init__(
        self,
        url: str,
        api_key: str,
        *,
        timeout: float = 5.0,
        verify: bool = True,
        ca_bundle: str | None = None,
    ) -> None:
        # Accept either the base URL or the full endpoint.
        self.url = url if url.rstrip("/").endswith("/v1/spans") else url.rstrip("/") + "/v1/spans"
        self.api_key = api_key
        self.timeout = timeout
        self._ssl_context = _build_ssl_context(verify, ca_bundle)

    def send(self, spans: list[Span]) -> None:
        if not spans:
            return
        payload = json.dumps({"spans": [s.model_dump(mode="json") for s in spans]}).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json", "X-API-Key": self.api_key},
        )
        try:
            # context is applied only to https requests; ignored for http.
            with urllib.request.urlopen(
                request, timeout=self.timeout, context=self._ssl_context
            ) as resp:
                if resp.status >= 300:
                    raise TransportError(f"gateway returned HTTP {resp.status}")
        except urllib.error.URLError as exc:
            raise TransportError(f"failed to reach gateway at {self.url}: {exc}") from exc


def _build_ssl_context(verify: bool, ca_bundle: str | None) -> ssl.SSLContext:
    """Build the TLS context: verified by default, a custom CA, or unverified."""
    if not verify:
        # Explicitly opt out of verification (test servers only).
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context
    # cafile=None uses the system trust store; a path trusts a private CA.
    return ssl.create_default_context(cafile=ca_bundle)
