"""Seams for Microsoft Agent Framework, so a run is captured and replays without ``@op``.

Agent Framework routes every model call through a chat client's
``_inner_get_response`` (one call per turn of the tool loop, streamed or not) and
every function tool through ``FunctionTool.invoke``. Both are wrapped with
pyTracer's op path (``pytracer_sdk.seam``), so capture and replay share one path:

- **Model calls**: ``_inner_get_response`` on every chat client class (each
  connector's client, and any client defined later), recorded as LLM spans. A
  streamed call is recorded update by update and replays as a stream.
- **Function tools**: ``FunctionTool.invoke``, recorded as tool spans. Hosted tools
  (web search, code interpreter) run inside the model call, so recording the model
  response covers them.
- **Agent runs**: ``run`` on every agent class opens an agent span that the model
  and tool calls nest under. It is a plain span, not an op: a replay still runs the
  agent, and only its model and tool calls come from the recording.

Agent Framework's responses are not pydantic models, so a call records their
``to_dict()`` form and a replay rebuilds them with ``from_dict()``. A live call
still returns the framework's own object, untouched.

Each model call's recorded input is normalized: the ids the framework generates
fresh every run (``af-call-...`` function call ids, message and response ids) are
left out, so a replay of the same conversation matches the recording.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from typing import Any

from agent_framework import (
    BaseAgent,
    BaseChatClient,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    FunctionTool,
    ResponseStream,
)
from pytracer_sdk import SpanKind, get_tracer
from pytracer_sdk.seam import ClassSeams, acall_through, astream_through, key_data, tool_names
from pytracer_sdk.tracer import ActiveSpan, _current

# Keys the framework fills with a fresh value on every run.
_VOLATILE_KEYS = frozenset({"id", "message_id", "response_id", "created_at"})

# The agents whose run is open in this context, so a run that calls its parent
# class's run (Agent.run -> RawAgent.run) opens one span, not two.
_open_agents: contextvars.ContextVar[frozenset[int]] = contextvars.ContextVar(
    "pytracer_agent_framework_open_agents", default=frozenset()
)


def _strip(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k not in _VOLATILE_KEYS}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def _format_name(response_format: Any) -> str | None:
    if response_format is None:
        return None
    if isinstance(response_format, type):
        return response_format.__name__
    return str(key_data(response_format))


def _model_key(messages: Any, options: Mapping[str, Any]) -> dict[str, Any]:
    """The model call's input as recorded and matched: the conversation and the
    request options, without the ids the framework generates per run."""
    rest = {k: v for k, v in options.items() if k not in ("tools", "response_format")}
    return {
        "messages": _strip([m.to_dict() for m in messages or ()]),
        "options": _strip(key_data(rest)),
        "tools": tool_names(options.get("tools")),
        "response_format": _format_name(options.get("response_format")),
    }


def _model_name(client: Any, options: Mapping[str, Any]) -> str:
    for name in (options.get("model"), getattr(client, "model", None)):
        if isinstance(name, str) and name:
            return name
    return type(client).__name__


def _usage_total(usage: Any) -> int | None:
    if not isinstance(usage, dict):
        return None
    if usage.get("total_token_count"):
        return int(usage["total_token_count"])
    count = int(usage.get("input_token_count") or 0) + int(usage.get("output_token_count") or 0)
    return count or None


def _update_tokens(update: Any) -> int | None:
    found = [
        _usage_total(content.get("usage_details"))
        for content in update.get("contents") or ()
        if isinstance(content, dict) and content.get("type") == "usage"
    ]
    counts = [count for count in found if count]
    return sum(counts) if counts else None


def _response_tokens(output: Any) -> int | None:
    """A model call's tokens: ``usage_details`` on the response, or for a stream the
    usage contents its updates carried."""
    if isinstance(output, list):
        counts = [_update_tokens(u) for u in output if isinstance(u, dict)]
        total = sum(c for c in counts if c)
        return total or None
    if isinstance(output, dict):
        return _usage_total(output.get("usage_details"))
    return None


def _wrap_inner_get_response(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def _inner_get_response(
        self: Any, *, messages: Any, stream: bool = False, options: Mapping[str, Any], **kwargs: Any
    ) -> Any:
        name = _model_name(self, options)
        key = _model_key(messages, options)

        def call() -> Any:
            return original(self, messages=messages, stream=stream, options=options, **kwargs)

        if stream:
            return _stream_response(call, name, key, options)

        live: deque[Any] = deque()

        async def body() -> Any:
            response = await call()
            live.append(response)
            return response.to_dict()

        async def response() -> Any:
            data = await acall_through(
                body, kind=SpanKind.LLM, name=name, tokens=_response_tokens, input=key
            )
            return live.popleft() if live else ChatResponse.from_dict(data)

        return response()

    return _inner_get_response


def _stream_response(
    call: Callable[[], Any], name: str, key: Any, options: Mapping[str, Any]
) -> ResponseStream[Any, Any]:
    """A streamed model call, recorded update by update.

    Live, the client's own stream is consumed and its updates are passed through
    unchanged, and the final response comes from that stream. Replayed, the
    recorded updates are rebuilt and combined the way the framework combines them.
    """
    live: deque[Any] = deque()
    client_stream: list[ResponseStream[Any, Any]] = []

    async def body() -> AsyncIterator[Any]:
        stream = call()
        if not isinstance(stream, ResponseStream):
            stream = await stream
        client_stream.append(stream)
        async for update in stream:
            live.append(update)
            yield update.to_dict()

    async def updates() -> AsyncIterator[Any]:
        recorded = astream_through(
            body, kind=SpanKind.LLM, name=name, tokens=_response_tokens, input=key
        )
        async for item in recorded:
            yield live.popleft() if live else ChatResponseUpdate.from_dict(item)

    async def finalize(items: Any) -> Any:
        if client_stream:
            return await client_stream[0].get_final_response()
        return ChatResponse.from_updates(items, output_format_type=options.get("response_format"))

    return ResponseStream(updates(), finalizer=finalize)


def _tool_input(arguments: Any, context: Any, kwargs: Mapping[str, Any]) -> Any:
    if arguments is None and kwargs:
        arguments = dict(kwargs)
    if arguments is None and context is not None:
        arguments = getattr(context, "arguments", None)
    return {"arguments": key_data(arguments)}


def _tool_data(result: Any) -> Any:
    if isinstance(result, list) and all(isinstance(item, Content) for item in result):
        return {"contents": [item.to_dict() for item in result]}
    return {"value": result}


def _tool_result(data: Any) -> Any:
    if isinstance(data, dict) and "contents" in data:
        return [Content.from_dict(item) for item in data["contents"]]
    return data.get("value") if isinstance(data, dict) else data


def _wrap_invoke(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    async def invoke(
        self: Any,
        *,
        arguments: Any = None,
        context: Any = None,
        tool_call_id: str | None = None,
        skip_parsing: bool = False,
        **kwargs: Any,
    ) -> Any:
        live: deque[Any] = deque()

        async def body() -> Any:
            result = await original(
                self, arguments=arguments, context=context, tool_call_id=tool_call_id,
                skip_parsing=skip_parsing, **kwargs,
            )
            live.append(result)
            return _tool_data(result)

        data = await acall_through(
            body, kind=SpanKind.TOOL, name=self.name, tool_name=self.name,
            input=_tool_input(arguments, context, kwargs),
        )
        return live.popleft() if live else _tool_result(data)

    return invoke


@contextlib.contextmanager
def _inside(span: ActiveSpan, agent: Any) -> Iterator[None]:
    """Make ``span`` the active span, and ``agent``'s run open, in this context."""
    span_token = _current.set(span.context)
    agent_token = _open_agents.set(_open_agents.get() | {id(agent)})
    try:
        yield
    finally:
        _open_agents.reset(agent_token)
        _current.reset(span_token)


def _run_input(messages: Any) -> Any:
    if messages is None or isinstance(messages, str):
        return messages
    to_dict = getattr(messages, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(messages, (list, tuple)):
        return [_run_input(m) for m in messages]
    return str(messages)


def _wrap_run(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def run(self: Any, messages: Any = None, *, stream: bool = False, **kwargs: Any) -> Any:
        if id(self) in _open_agents.get():
            return original(self, messages, stream=stream, **kwargs)
        name = getattr(self, "name", None) or type(self).__name__
        span = get_tracer().start_span(
            name, kind=SpanKind.AGENT, agent_name=name, input=_run_input(messages)
        )
        span.set_attribute("framework", "agent-framework")
        if not stream:
            with _inside(span, self):
                pending = original(self, messages, stream=False, **kwargs)

            async def response() -> Any:
                with span, _inside(span, self):
                    result = await pending
                    span.set_output(getattr(result, "text", None))
                    return result

            return response()

        with _inside(span, self):
            agent_stream = original(self, messages, stream=True, **kwargs)
        text: list[str] = []

        @contextlib.contextmanager
        def pull() -> Iterator[None]:
            try:
                with _inside(span, self):
                    yield
            except StopAsyncIteration:
                raise
            except BaseException as exc:
                span.set_error(exc)
                span.close()
                raise

        def collect(update: Any) -> Any:
            text.append(getattr(update, "text", "") or "")
            return update

        def finish() -> None:
            span.set_output("".join(text))
            span.close()

        return (
            agent_stream.with_pull_context_manager(pull)
            .with_transform_hook(collect)
            .with_cleanup_hook(finish)
        )

    return run


class Seams:
    """The installed Agent Framework seams; :meth:`uninstall` removes them."""

    def __init__(self) -> None:
        self._seams = ClassSeams({
            BaseChatClient: [("_inner_get_response", lambda _, original: _wrap_inner_get_response(original))],
            FunctionTool: [("invoke", lambda _, original: _wrap_invoke(original))],
            BaseAgent: [("run", lambda _, original: _wrap_run(original))],
        })

    def install(self) -> None:
        self._seams.install()

    def uninstall(self) -> None:
        self._seams.uninstall()


_installed: Seams | None = None


def install() -> Seams:
    """Capture every Agent Framework run, replayably, and return the seams.

    Model calls, function tools and agent runs are recorded with no change to the
    agent code, and a captured run replays with the model and tool responses
    frozen. Spans go to the tracer active when each call runs (the one
    ``pytracer.configure(...)`` set, or a replay's), so this can be called before or
    after ``configure``. Installing again replaces the previous seams.
    """
    global _installed
    uninstall()
    _installed = Seams()
    _installed.install()
    return _installed


def uninstall() -> None:
    """Remove the seams installed by :func:`install`."""
    global _installed
    if _installed is not None:
        _installed.uninstall()
        _installed = None
