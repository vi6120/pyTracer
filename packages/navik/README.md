# navik

The `navik` meta-package installs the entire Navik platform in one command:

```bash
pip install navik
```

That pulls in every component: `navik-sdk`, `navik-stores`, `navik-gateway`,
`navik-replay`, `navik-runner`, `navik-registry`, and `navik-cli`.

For convenience the SDK's core entrypoints are re-exported here, so `import
navik` is enough to instrument an agent:

```python
import navik

@navik.op(kind=navik.SpanKind.TOOL, tool_name="search")
def search(query: str) -> dict:
    ...
```

If you only need one component (say, just the SDK in your agent), install that
component directly instead: `pip install navik-sdk`.
