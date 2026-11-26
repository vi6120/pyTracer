"""HTTP requests made inside a tool or model call, recorded on its span.

A retry often happens below the call pyTracer sees: a model SDK's own retry loop,
or a tool that retries its POST after a timeout. The call then looks like one
clean success while the request went out twice, and a write that went through
before the timeout is repeated. Every HTTP request sent while a tool or model
call is running is recorded on that call's span as ``pytracer.http``: the method,
the URL without its query string, and how it ended (status, or the error). A
request identical to an earlier one in the same call (same method, URL and body)
is numbered (``attempt``), and marked a ``retry`` when the one before failed (an
error, or status 408, 429 or 5xx).

Headers and bodies are never recorded (the body is only hashed, to tell identical
requests apart), and the query string is left out of the recorded URL, since both
can carry credentials. Requests are seen at the client: ``httpx`` and ``httpx2``
(each ``send``, which the OpenAI and Anthropic SDKs call once per attempt) and
``urllib3`` (each ``urlopen`` attempt, which covers ``requests`` and its retry
adapter). Set ``PYTRACER_HTTP_CAPTURE=0`` to turn it off.
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import importlib
import importlib.util
import os
import threading
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

from .tracer import open_call

ATTRIBUTE = "pytracer.http"
MAX_REQUESTS = 50  # per call; further requests are counted, not listed
_FAILED_STATUS = frozenset({408, 429})

_lock = threading.Lock()
_patched: list[tuple[Any, str, Any]] = []
# The urllib3 urlopen call (one attempt) running in this context; see below.
_in_urlopen: contextvars.ContextVar[list[bool] | None] = contextvars.ContextVar(
    "pytracer_in_urlopen", default=None
)


def _display_url(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}{parts.path}"


def _digest(method: str, url: str, body: Any) -> str:
    sha = hashlib.sha1(usedforsecurity=False)
    sha.update(f"{method.upper()} {url}".encode())
    if isinstance(body, str):
        body = body.encode()
    if isinstance(body, (bytes, bytearray)):
        sha.update(b"\0" + bytes(body))
    return sha.hexdigest()[:16]


def _failed(status: int | None, error: str | None) -> bool:
    return error is not None or (status is not None and (status >= 500 or status in _FAILED_STATUS))


def record(method: str, url: str, body: Any, status: int | None, error: str | None) -> None:
    """Record one request on the tool or model call running in this context."""
    active = open_call()
    if active is None:
        return
    key = _digest(method, url, body)
    entry: dict[str, Any] = {
        "method": method.upper(),
        "url": _display_url(url),
        "status": status,
        "error": error,
        "key": key,
    }
    with _lock:
        attributes = active.span.attributes
        requests = attributes.setdefault(ATTRIBUTE, [])
        same = [r for r in requests if r.get("key") == key]
        if same:
            entry["attempt"] = len(same) + 1
            entry["retry"] = _failed(same[-1].get("status"), same[-1].get("error"))
        if len(requests) < MAX_REQUESTS:
            requests.append(entry)
        else:
            attributes["pytracer.http_dropped"] = int(attributes.get("pytracer.http_dropped", 0)) + 1


# --- httpx and httpx2 ---------------------------------------------------------


def _request_body(request: Any) -> Any:
    try:
        return request.content
    except Exception:  # noqa: BLE001 - a streamed body is not read here
        return None


def _wrap_send(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def send(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        method, url, body = str(request.method), str(request.url), _request_body(request)
        try:
            response = original(self, request, *args, **kwargs)
        except Exception as exc:
            record(method, url, body, None, type(exc).__name__)
            raise
        record(method, url, body, int(response.status_code), None)
        return response

    return send


def _wrap_async_send(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    async def send(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        method, url, body = str(request.method), str(request.url), _request_body(request)
        try:
            response = await original(self, request, *args, **kwargs)
        except Exception as exc:
            record(method, url, body, None, type(exc).__name__)
            raise
        record(method, url, body, int(response.status_code), None)
        return response

    return send


# --- urllib3 (and requests) -----------------------------------------------------
# One urlopen call sends one request. A failed attempt (an error, or a status the
# Retry policy retries) goes through Retry.increment, which sees its outcome, before
# urlopen calls itself for the next attempt; so failures are recorded there, and
# urlopen records only an attempt that ends the chain without increment seeing it.
# The body is not hashed for urllib3 (increment does not see it), so its requests
# are identified by method and URL.


def _pool_url(pool: Any, url: str | None) -> str:
    url = url or "/"
    if "://" in url:
        return url
    return f"{pool.scheme}://{pool.host}:{pool.port}{url}"


def _wrap_urlopen(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def urlopen(self: Any, method: str, url: str, *args: Any, **kwargs: Any) -> Any:
        outer = _in_urlopen.get()
        if outer is not None:
            outer[0] = True  # the outer call hands over to this attempt
        frame = [False, False]  # [another attempt followed, increment recorded this one]
        token = _in_urlopen.set(frame)
        try:
            response = original(self, method, url, *args, **kwargs)
        except Exception as exc:
            if not frame[0] and not frame[1]:
                record(method, _pool_url(self, url), None, None, type(exc).__name__)
            raise
        finally:
            _in_urlopen.reset(token)
        if not frame[0] and not frame[1]:
            record(method, _pool_url(self, url), None, int(getattr(response, "status", 0) or 0) or None, None)
        return response

    return urlopen


def _wrap_increment(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def increment(self: Any, method: str | None = None, url: str | None = None, *args: Any, **kwargs: Any) -> Any:
        frame = _in_urlopen.get()
        pool = kwargs.get("_pool")
        if frame is not None and not frame[1] and method and pool is not None:
            frame[1] = True
            response, error = kwargs.get("response"), kwargs.get("error")
            status = int(getattr(response, "status", 0) or 0) or None if response is not None else None
            record(method, _pool_url(pool, url), None, status, type(error).__name__ if error else None)
        return original(self, method, url, *args, **kwargs)

    return increment


def _targets() -> list[tuple[str, str, str, Callable[[Callable[..., Any]], Callable[..., Any]]]]:
    return [
        ("httpx", "Client", "send", _wrap_send),
        ("httpx", "AsyncClient", "send", _wrap_async_send),
        ("httpx2", "Client", "send", _wrap_send),
        ("httpx2", "AsyncClient", "send", _wrap_async_send),
        ("urllib3.connectionpool", "HTTPConnectionPool", "urlopen", _wrap_urlopen),
        ("urllib3.util.retry", "Retry", "increment", _wrap_increment),
    ]


def install() -> None:
    """Record HTTP requests inside tool and model calls (idempotent)."""
    if os.environ.get("PYTRACER_HTTP_CAPTURE", "1").strip().lower() in ("0", "false", "no", "off"):
        return
    with _lock:
        if _patched:
            return
        for module_name, class_name, method, wrap in _targets():
            if importlib.util.find_spec(module_name.partition(".")[0]) is None:
                continue
            try:
                cls = getattr(importlib.import_module(module_name), class_name)
            except (ImportError, AttributeError):
                continue
            original = cls.__dict__.get(method)
            if original is None or getattr(original, "__pytracer_http__", False):
                continue
            wrapper = wrap(original)
            wrapper.__pytracer_http__ = True  # type: ignore[attr-defined]
            setattr(cls, method, wrapper)
            _patched.append((cls, method, original))


def uninstall() -> None:
    """Restore the HTTP clients patched by :func:`install`."""
    with _lock:
        while _patched:
            cls, method, original = _patched.pop()
            setattr(cls, method, original)
