# pytracer-sdk

Layer 1 of pyTracer: captures agent actions (LLM calls, tool calls, reasoning steps, handoffs)
as structured, OpenTelemetry-compliant spans, with local redaction and non-blocking flush.

```python
import pytracer_sdk as pytracer

@pytracer.op(kind="llm")
def call_model(prompt: str) -> str:
    ...
```
