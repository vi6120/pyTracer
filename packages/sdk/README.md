# pytracer-sdk

Layer 1 of pyTracer: captures agent actions (LLM calls, tool calls, reasoning steps, handoffs)
as structured, OpenTelemetry-compliant spans, with local redaction and non-blocking flush.

```python
import pytracer_sdk as pytracer

@pytracer.op(kind="llm")
def call_model(prompt: str) -> str:
    ...
```

## Capturing to a gateway

`HTTPTransport` ships spans to the ingestion gateway. Capture is fire-and-forget:
a gateway outage never blocks or crashes the agent. The gateway can accept a batch
(HTTP 202) yet still reject individual spans (for example a span missing
`branch`/`commit`). Those rejections are surfaced rather than lost:

- `HTTPTransport.rejected_spans` is a running count of spans the gateway rejected.
- Each rejection is logged as a warning (on the `pytracer_sdk` logger) with the
  gateway's reason.
- `on_rejection`, an optional `Callable[[int, list], None]`, is called with the
  count and the error list when a batch had rejections, so a sender can alert or
  fail its CI on silently dropped spans.

```python
from pytracer_sdk.transport import HTTPTransport

def alert(count, errors):
    raise SystemExit(f"gateway rejected {count} span(s): {errors[0]}")

transport = HTTPTransport("https://gateway.example", api_key="...", on_rejection=alert)
pytracer.configure(transport, service_name="my-agent")
# ... run the agent ...
print(transport.rejected_spans)  # 0 when every span was accepted
```

A bad `on_rejection` callback can never kill the flush worker: it is called
defensively and a raise from it is logged, not propagated.
