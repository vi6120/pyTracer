# pytracer-autogen

pyTracer adapter for AutoGen (0.4+). AutoGen emits OpenTelemetry spans using the
GenAI semantic conventions; this adapter plugs a pyTracer exporter into an
OpenTelemetry tracer provider, so the spans AutoGen emits are captured as pyTracer
spans with no `@op` in the agent code. pyTracer records what AutoGen reports, and
AutoGen reports less than the other frameworks:

| | Captured |
|---|---|
| Agent spans, with the agent's name | Yes |
| Tool calls | Yes, as tool spans. AutoGen 0.7.x does not report the tool's result, so the output is empty |
| Model calls | Not from AutoGen 0.7.x, which emits no model-call span. With a GenAI-instrumented model client (for example `opentelemetry-instrumentation-openai-v2`), each call is captured with its model and token counts, but not its output, because that library sends content to the OpenTelemetry logs API |
| One trace per team run | No. AutoGen gives each agent its own OpenTelemetry trace, so a team run arrives as several traces |
| Nesting under an enclosing `pytracer.start_span(...)` | No. Spans keep AutoGen's own trace ids |
| Replay | Not through the adapter (see [Replay](#replay)) |

If you need full model and tool payloads or one trace per run today, add `@op` to
the model client wrapper and the tools; those spans are complete and replayable.

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

## Replay

This adapter **captures** runs. Replaying a captured run currently requires `@op`
instrumentation, because the replay engine intercepts `@op` calls; replay through
the adapter itself is on the roadmap. Attempting to replay an adapter-only trace
is flagged (the run has recordings but hits no `@op` seam) rather than silently
running live. See the main README for the replay workflow.
