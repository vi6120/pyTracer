# pytracer-openai-agents

pyTracer adapter for the OpenAI Agents SDK. It auto-instruments an agent through
the SDK's built-in tracing, so you do not add `@op` by hand: agent runs, model
generations, tool (function) calls, and handoffs are captured as pyTracer spans.

```python
import pytracer_sdk as pytracer
from agents.tracing import add_trace_processor
from pytracer_openai_agents import PyTracerTracingProcessor

pytracer.configure(
    pytracer.HTTPTransport("https://gateway.internal", api_key="YOUR_KEY"),
    service_name="my-agent", branch="main", commit="abc123",
)

add_trace_processor(PyTracerTracingProcessor())   # now every agent run is captured
```

Agent spans map to agent spans, generations and responses to LLM spans, function
calls to tool spans, and handoffs to handoff spans, all linked into one trace by
the SDK's own span tree.
