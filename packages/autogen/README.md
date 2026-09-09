# navik-autogen

Navik adapter for AutoGen (0.4+). AutoGen instruments agents, tool calls, and
model calls with OpenTelemetry using the GenAI semantic conventions. This adapter
plugs a Navik exporter into an OpenTelemetry tracer provider, so those spans are
captured as Navik spans with no `@op` in the agent code.

```python
import navik_sdk as navik
from navik_autogen import navik_tracer_provider
from autogen_core import SingleThreadedAgentRuntime

navik.configure(
    navik.HTTPTransport("https://gateway.internal", api_key="YOUR_KEY"),
    service_name="my-agent", branch="main", commit="abc123",
)

runtime = SingleThreadedAgentRuntime(tracer_provider=navik_tracer_provider())
```

Because it keys off standard GenAI attributes (`gen_ai.operation.name`,
`gen_ai.tool.name`, `gen_ai.agent.name`, `gen_ai.usage.*`), the same exporter
captures spans from any GenAI-instrumented library, not only AutoGen.
