# pytracer

The `pytracer` meta-package installs the entire pyTracer platform in one command:

```bash
pip install pytracer
```

That pulls in every component: `pytracer-sdk`, `pytracer-stores`, `pytracer-gateway`,
`pytracer-replay`, `pytracer-runner`, `pytracer-registry`, and `pytracer-cli`.

For convenience the SDK's core entrypoints are re-exported here, so `import
pytracer` is enough to instrument an agent:

```python
import pytracer

@pytracer.op(kind=pytracer.SpanKind.TOOL, tool_name="search")
def search(query: str) -> dict:
    ...
```

If you only need one component (say, just the SDK in your agent), install that
component directly instead: `pip install pytracer-sdk`.
