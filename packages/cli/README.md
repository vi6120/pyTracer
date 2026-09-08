# navik-cli

Layer 6 of Navik: a CLI that runs agent **test collections** locally and in CI,
matching exactly what runs in a GitHub Action so there are no environment
surprises. The GitHub Action and GitLab CI templates are thin wrappers around
this command.

```bash
navik run tests/collection.yaml            # human summary; exit 1 if any scenario fails
navik run tests/collection.yaml --json out.json --junit out.xml --comment pr.md
```

A collection (`collection.yaml`):

```yaml
name: my-agent-tests
scenarios:
  - name: summarize-happy-path
    agent: myapp.agents:run        # import path "module:callable"
    entry: { task: "summarize the doc" }
    trace: traces/summarize.jsonl  # recorded spans (JSONL), relative to this file
    mode: full                     # full | partial | live
    assertions:
      - { type: no_errors }
      - { type: contains, needle: "summary" }
```
