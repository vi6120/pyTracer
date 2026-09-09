# pytracer-autogen

pyTracer adapter for AutoGen (0.4+). AutoGen instruments agents, tool calls, and
model calls with OpenTelemetry using the GenAI semantic conventions. This adapter
plugs a pyTracer exporter into an OpenTelemetry tracer provider, so those spans are
captured as pyTracer spans with no `@op` in the agent code.

```python
import pytracer_sdk as pytracer
from pytracer_autogen import pytracer_tracer_provider
from autogen_core import SingleThreadedAgentRuntime

pytracer.configure(
    pytracer.HTTPTransport("https://gateway.internal", api_key="YOUR_KEY"),
    service_name="my-agent", branch="main", commit="abc123",
)

runtime = SingleThreadedAgentRuntime(tracer_provider=pytracer_tracer_provider())
```

Because it keys off standard GenAI attributes (`gen_ai.operation.name`,
`gen_ai.tool.name`, `gen_ai.agent.name`, `gen_ai.usage.*`), the same exporter
captures spans from any GenAI-instrumented library, not only AutoGen.
