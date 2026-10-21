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

## Replay

This adapter **captures** runs. Replaying a captured run currently requires `@op`
instrumentation, because the replay engine intercepts `@op` calls; replay through
the adapter itself is on the roadmap. Attempting to replay an adapter-only trace
is flagged (the run has recordings but hits no `@op` seam) rather than silently
running live. See the main README for the replay workflow.
