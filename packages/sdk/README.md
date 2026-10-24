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

## Spans lost before they reach the gateway

Spans can also be lost on the client side, before the gateway sees them: the
in-process queue is full (the agent produces spans faster than they can be
shipped), or a delivery fails (the gateway is unreachable, or a 429/503 persists
past the retries). These are surfaced the same way:

- `tracer.buffer.dropped` counts spans dropped because the queue was full, and
  `tracer.buffer.send_errors` counts spans whose delivery failed.
- Each loss is logged as a warning on the `pytracer_sdk` logger. The warning is
  rate limited to one per reason every 30 seconds, carrying the number of spans
  lost since the previous one, so an outage does not flood the agent's log.
- `on_drop`, an optional `Callable[[str, int, BaseException | None], None]`
  passed to `configure()`, is called on every loss with the reason
  (`"queue_full"` or `"send_failed"`), the number of spans, and the delivery error
  (`None` for a full queue). Like `on_rejection`, a raise from it is logged, not
  propagated.

```python
def on_drop(reason, count, error):
    metrics.increment("pytracer.spans_lost", count, tags={"reason": reason})

tracer = pytracer.configure(transport, service_name="my-agent", on_drop=on_drop)
# ... run the agent ...
print(tracer.buffer.dropped, tracer.buffer.send_errors)  # 0 0 when nothing was lost
```

If `dropped` grows, raise the queue bound with
`SpanBuffer(transport, max_queue_size=...)` passed as `Tracer(..., buffer=...)`,
or ship to a closer or larger gateway.
