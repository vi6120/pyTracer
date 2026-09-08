# navik-sdk

Layer 1 of Navik: captures agent actions (LLM calls, tool calls, reasoning steps, handoffs)
as structured, OpenTelemetry-compliant spans, with local redaction and non-blocking flush.

```python
import navik_sdk as navik

@navik.op(kind="llm")
def call_model(prompt: str) -> str:
    ...
```
