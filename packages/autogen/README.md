# pytracer-autogen

pyTracer adapter for AutoGen (0.4+; the separate AutoGen 0.2 API is not
supported). AutoGen emits OpenTelemetry spans using the
GenAI semantic conventions; this adapter plugs a pyTracer exporter into an
OpenTelemetry tracer provider, so the spans AutoGen emits are captured as pyTracer
spans with no `@op` in the agent code. AutoGen's own spans carry less than the
other frameworks report, so the adapter fills in the rest:

| | Captured |
|---|---|
| Agent spans, with the agent's name | Yes |
| Tool calls with their output | Yes. AutoGen's tool span has only the call id; the result is joined from the message AutoGen sends with it, and an error result marks the span failed |
| Model calls | Yes. AutoGen emits no model-call span, so each call is recorded from the call event the model client logs (the `autogen_ext` clients for OpenAI, Azure, Anthropic, Ollama and others do), with model, request, reply and tokens; a client that logs nothing has its calls rebuilt from the replies AutoGen delivers (reply and tokens, marked `pytracer.derived`). A GenAI-instrumented client (for example `opentelemetry-instrumentation-openai-v2`) keeps its prompt and reply too: `install()` reads the content that library sends as log records |
| One trace per team run | Yes. AutoGen delivers each message in a new OpenTelemetry trace linked to the sender's; the adapter follows those links (see [One trace per run](#one-trace-per-run)) |
| Nesting under an enclosing `pytracer.start_span(...)` | Yes |
| Replay without `@op` | Yes, with `install(replayable=True)` (see [Replay](#replay)) |

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

**What `install()` changes in your process**, each logged at INFO when it happens:

- It registers pyTracer as the global OpenTelemetry tracer provider.
- It adds a handler to the `autogen_core.events` logger to read model-call events.
  If that logger was dropping INFO records (the default), it is set to INFO with
  propagation off, so your other handlers see exactly what they saw before; set
  that logger's level yourself to manage it.
- It adds a processor to the global OpenTelemetry LoggerProvider to read GenAI
  model content, registering a LoggerProvider if none is set. To use your own,
  register it before `install()` and pyTracer adds to it.

**Versions:** what is captured depends on the AutoGen version emitting GenAI
spans. Some versions emit no model-call span or tool result; pyTracer records what
the framework reports, so upgrade AutoGen (or add the relevant GenAI
instrumentation) if model calls or tool outputs are missing.

## Alongside other frameworks

AutoGen 0.7.5, its latest release (September 2025; AutoGen is in maintenance mode), pins
`protobuf~=5.29.3`. AutoGen does not use `googleapis-common-protos` itself, but its
protobuf pin rules out 1.75.1 and later, which need `protobuf>=6.33.5`. Installed
in one command with other frameworks (CrewAI pulls it in through both
`opentelemetry-exporter-otlp-proto-http` and `chromadb`), pip and uv pick a
compatible 1.75.0. Added to an existing
environment that has a newer one, pip downgrades protobuf and only warns, and the
next import fails with `VersionError: Detected incompatible Protobuf Gencode/Runtime
versions`; fix it with `pip install "googleapis-common-protos<1.75.1"`. See
[using several frameworks in one environment](https://pytracer.com/docs#framework-combinations).

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

Install with `replayable=True` and a captured run replays without adding `@op`
to anything:

```python
install(replayable=True)   # also routes model clients and tools through pyTracer
```

AutoGen's model calls (`create` / `create_stream` on every `ChatCompletionClient`,
including the `autogen_ext` clients and custom ones) and tool calls
(`BaseTool.run_json`, which every tool call goes through) then run through
pyTracer, so the replay engine can return their recorded results: in full mode
neither the models nor the tools run again, in partial mode the tools stay frozen
while the models run live, and streamed replies replay chunk by chunk. Each model
call is matched on what was asked (the messages, the tools offered, the output
format). `uninstall_seams()` removes the routing again.

Without `replayable=True` the adapter only captures, and replaying an
adapter-only trace is flagged (the run has recordings but hits no `@op` seam)
rather than silently running live. See the main README for the replay workflow.

## AutoGen versions

Every AutoGen release from 0.4.0 is captured in full (the table above), but older
releases report less about a run, so `install()` fills the gap differently:

| AutoGen | What it reports | How the adapter captures the rest |
|---|---|---|
| 0.7.2 and later | GenAI agent, tool and model-call data, and each tool's result | From AutoGen's own spans and events |
| 0.6.2 to 0.7.1 | The GenAI spans, but no tool results | Model and tool calls go through pyTracer (the routing `replayable=True` uses), which records their output |
| 0.4.0 to 0.6.1 | No agent or tool spans (before 0.4.9, no spans at all) | As above, plus each agent turn and each team run go through pyTracer, so a run is one trace with its turns, calls and tool output |

On releases before 0.7.2 that routing is always on, since it is how the calls
are captured; `uninstall_seams()` would stop it. Streamed replies replay chunk by
chunk wherever AgentChat streams them (0.4.5 and later). CI runs the adapter's
tests on 0.4.0, 0.4.9, 0.6.2, 0.7.2 and the latest release.
