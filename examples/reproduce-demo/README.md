# Cross-branch reproduction demo

This is pyTracer's headline feature in one command: take a **captured failure**
and re-run it on another branch with the model and tool responses **held
frozen**, so you learn whether that branch actually fixed the bug, not whether
the agent happened to behave differently this time.

`run_demo.sh` builds a throwaway git repo with a buggy `main` branch and a `fix`
branch, then reproduces the captured failure on `fix`. The two branches differ by
exactly one line, and the mocks are identical, so the verdict is a clean signal
about the code change alone.

## The agent

A tiny "analyst" that searches for sources and writes a one-line answer
([`variants/analyst_fixed.py`](variants/analyst_fixed.py)). `search` and
`summarize` carry an `@op` decorator, so in replay their recorded outputs are
injected from [`traces/run.jsonl`](traces/run.jsonl) and the run is
deterministic. The bug lives in `_headline`, which is plain glue code (not an
`@op`), so it runs **live** during replay. That is the whole point: the fix
changes live code, while the mocked model and tool calls stay frozen.

The one-line bug (`analyst_buggy.py` vs `analyst_fixed.py`):

```diff
-    return f"Reviewed {n_sources - 1} sources. {summary}"   # undercounts
+    return f"Reviewed {n_sources} sources. {summary}"       # correct
```

The collection ([`collection.yaml`](collection.yaml)) asserts the answer says
`Reviewed 3 sources`. On `main` it says "2" and fails; on `fix` it says "3" and
passes.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "pytracer-cli[runner]"
./run_demo.sh
```

Expected output:

```
==> The only difference between main and fix:
-    return f"Reviewed {n_sources - 1} sources. {summary}"
+    return f"Reviewed {n_sources} sources. {summary}"

==> Reproducing the captured failure on the fix branch (mocks frozen)...

SCENARIO                     OUTCOME
cites-every-source           fixed

==> Every scenario is FIXED on the fix branch. ...
```

By default the demo uses `--isolation local` (no Docker, runs in the current
virtualenv). For full container isolation - each ref in its own
`python:3.12-slim` - run it with Docker instead:

```bash
ISOLATION=docker ./run_demo.sh
```

## The three verdicts

`pytracer reproduce` classifies each scenario on the candidate branch against the
baseline:

- **fixed** - the scenario now passes (this demo).
- **still failing** - it fails the same way (same failure fingerprint): the
  branch did not fix it.
- **diverged** - it fails, but differently: a new, different failure.

The command exits non-zero unless every scenario is `fixed`, so it drops straight
into a CI gate or a "try on this branch" PR check.

## Using it on your own repo

There is nothing demo-specific about the command. Point it at your repository, a
collection, and a candidate branch:

```bash
pytracer reproduce \
  --repo . \
  --baseline main \
  --branch feature/fix-the-bug \
  --collection tests/collection.yaml \
  --freeze-trace traces/failure.jsonl \
  --trace-dest traces/failure.jsonl
```

Scaffold the collection and trace from a real captured failure with
`pytracer traces list --failed` and `pytracer record` (see the
[docs](https://pytracer.com/docs)).

## Recording it as a GIF

The run is short and deterministic, which makes it easy to record for a demo.
With [vhs](https://github.com/charmbracelet/vhs):

```
# reproduce.tape
Output reproduce.gif
Set FontSize 20
Set Width 1200
Set Height 700
Type "./run_demo.sh" Enter
Sleep 12s
```

```bash
vhs reproduce.tape
```

Or record a terminal session with [asciinema](https://asciinema.org) and convert
it with [agg](https://github.com/asciinema/agg).

## Regenerating the frozen trace

The committed [`traces/run.jsonl`](traces/run.jsonl) was produced by
[`capture.py`](capture.py). To regenerate it:

```bash
python capture.py
```
