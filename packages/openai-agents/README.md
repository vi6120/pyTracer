# pytracer-openai-agents

pyTracer adapter for the OpenAI Agents SDK. It auto-instruments an agent through
the SDK's built-in tracing, so you do not add `@op` by hand: agent runs, model
generations, tool (function) calls, and handoffs are captured as pyTracer spans.

```python
import pytracer_sdk as pytracer
from pytracer_openai_agents import install

pytracer.configure(
    pytracer.HTTPTransport("https://gateway.internal", api_key="YOUR_KEY"),
    service_name="my-agent", branch="main", commit="abc123",
)

install()   # now every agent run is captured, and only pyTracer receives traces
```

`install()` uses the SDK's `set_trace_processors` to make pyTracer the **only**
trace processor. Do not use `add_trace_processor`: it appends, leaving the SDK's
default processor in place, which uploads every trace to OpenAI when
`OPENAI_API_KEY` is set. If you deliberately want both destinations, add the
processor yourself with `add_trace_processor(PyTracerTracingProcessor())`.

Agent spans map to agent spans, generations and responses to LLM spans, function
calls to tool spans, and handoffs to handoff spans, all linked into one trace by
the SDK's own span tree.

## Alongside other frameworks

CrewAI requires `openai<3`, so with CrewAI in the same environment pip and uv
install OpenAI Agents 0.20.0, the last release that accepts `openai` 2.x; this
adapter supports it (0.20.0 is also its minimum). Asking for a newer OpenAI Agents
SDK beside pyTracer's CrewAI adapter is reported as a conflict, not settled by
installing an old CrewAI. For a newer OpenAI Agents SDK, run it in an environment
without CrewAI. See
[using several frameworks in one environment](https://pytracer.com/docs#framework-combinations).

## One trace per request

Each run starts its own trace. To keep a request that makes several runs, or
also uses another framework, in one trace, wrap it in a pyTracer span; the run
nests under the span that is open when it starts:

```python
with pytracer.start_span("answer-question"):
    await Runner.run(agent, question)
    ...  # further runs, other adapters, or @op calls join the same trace
```

## Replay

Install with `replayable=True` and a captured run replays without adding `@op`
to anything:

```python
install(replayable=True)   # also routes model calls and function tools through pyTracer
```

The SDK's model calls (`get_response` / `stream_response` on every `Model`,
including custom ones) and function tools (`FunctionTool.on_invoke_tool`,
including tools created before `install`) then run through pyTracer, so the
replay engine can return their recorded results: in full mode neither the model
nor the tools run again, in partial mode the tools stay frozen while the model
runs live. Hosted tools (web search, file search) run inside the model call, so
the recorded model response covers them. Each model call is matched on what was
asked (instructions, conversation so far, tools, settings), not on per-run ids.
`uninstall_seams()` removes the routing again.

Without `replayable=True` the adapter only captures, and replaying an
adapter-only trace is flagged (the run has recordings but hits no `@op` seam)
rather than silently running live. See the main README for the replay workflow.
