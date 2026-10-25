# pytracer-autogen

pyTracer adapter for AutoGen (0.4+). AutoGen instruments agents, tool calls, and
model calls with OpenTelemetry using the GenAI semantic conventions. This adapter
plugs a pyTracer exporter into an OpenTelemetry tracer provider, so those spans are
captured as pyTracer spans with no `@op` in the agent code.

AutoGen emits its GenAI model and tool spans through the **global** OpenTelemetry
provider, so register pyTracer globally with `install()` before building the
runtime. Configure pyTracer first, so the exporter records into your transport:

```python
import pytracer_sdk as pytracer
from pytracer_autogen import install
from autogen_core import SingleThreadedAgentRuntime

pytracer.configure(
    pytracer.HTTPTransport("https://gateway.internal", api_key="YOUR_KEY"),
    service_name="my-agent", branch="main", commit="abc123",
)

install()  # register pyTracer as the global OpenTelemetry provider
runtime = SingleThreadedAgentRuntime()
```

Passing a provider only to the runtime
(`SingleThreadedAgentRuntime(tracer_provider=pytracer_tracer_provider())`) sees
just the runtime's internal messaging spans, which the exporter drops as noise, so
`install()` is the setup that captures the actual model and tool calls.

Because it keys off standard GenAI attributes (`gen_ai.operation.name`,
`gen_ai.tool.name`, `gen_ai.agent.name`, `gen_ai.usage.*`), the same exporter
captures spans from any GenAI-instrumented library, not only AutoGen.

**Versions:** what is captured depends on the AutoGen version emitting GenAI
spans. Some versions emit no model-call span or tool result; pyTracer records what
the framework reports, so upgrade AutoGen (or add the relevant GenAI
instrumentation) if model calls or tool outputs are missing.

## One trace per run

AutoGen delivers each message in a new OpenTelemetry trace, linked to the trace
that sent it. The provider built by `install()` follows those links, so a team
run is recorded as **one** pyTracer trace, with each agent turn at the top of the
run and its tool calls beneath it. A run started inside an open pyTracer span
nests under it, like the other adapters:

```python
with pytracer.start_span("answer-question"):
    await team.run(task="...")   # one trace, rooted at "answer-question"
```

## Replay

This adapter **captures** runs. Replaying a captured run currently requires `@op`
instrumentation, because the replay engine intercepts `@op` calls; replay through
the adapter itself is on the roadmap. Attempting to replay an adapter-only trace
is flagged (the run has recordings but hits no `@op` seam) rather than silently
running live. See the main README for the replay workflow.
