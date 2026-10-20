# Three-verdict reproduction demo

Most agent test tools answer one question on a fix branch: did the test pass or
fail? That is not enough. "Still broken" and "broken a new way" both show up as
a red X, but a team needs to act on them differently.

pyTracer replays a captured failure on another branch with the model and tool
responses **held frozen**, so only the code changes, and classifies each
scenario three ways:

- **fixed** - the scenario now passes.
- **still-failing** - it fails the same way (same failure fingerprint): the
  change did not address the bug.
- **diverged** - it fails, but differently (a new fingerprint): the branch
  swapped the old bug for a new one.

`run_demo.sh` builds a throwaway git repo with a buggy `main` branch and three
candidate branches, then reproduces the same captured failure on each:

```
BRANCH     VERDICT        MEANING
fix        fixed          the branch actually fixed it
still      still_failing  same failure: the change was a no-op
diverged   diverged       fails a new, different way
```

## The agent

A tiny "analyst" that searches for sources and writes a one-line answer.
`search` and `summarize` carry an `@op` decorator, so in replay their recorded
outputs are injected from [`traces/run.jsonl`](traces/run.jsonl) and the run is
deterministic. The behavior under test lives in `_headline`, which is plain glue
code (not an `@op`), so it runs **live** during replay. Each branch changes only
`_headline`:

| Branch | `_headline` does | Replay outcome |
| --- | --- | --- |
| `main` (baseline) | indexes a 4th source that is not there | fails (`IndexError`) |
| `fix` | counts the sources it has | **fixed** |
| `still` | reworded, still indexes the 4th source | **still-failing** |
| `diverged` | reads the prose summary as a number | **diverged** (`ValueError`) |

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "pytracer-cli[runner]"
./run_demo.sh
```

By default the demo uses `--isolation local` (no Docker, runs in the current
virtualenv). For full container isolation, each branch in its own
`python:3.12-slim`, run it with Docker instead:

```bash
ISOLATION=docker ./run_demo.sh
```

## How the verdict is decided

`still-failing` and `diverged` are told apart by a **failure fingerprint**
derived from the call sequence and the error type (see
`pytracer_replay.fingerprint`), so the same bug on two branches fingerprints the
same, while a genuinely different failure fingerprints differently.

The fingerprint is defined for failures that surface as an error (an errored span
or a raised exception), which is why this demo's bugs raise. A failure that only
trips an assertion on otherwise-clean output is reported as failing, but is not
yet fingerprinted finely enough to separate still-failing from diverged; widening
that is on the roadmap.

## Regenerating the frozen trace

The committed [`traces/run.jsonl`](traces/run.jsonl) was produced by
[`capture.py`](capture.py). To regenerate it:

```bash
python capture.py
```

## Using it on your own repo

There is nothing demo-specific about the command. Point `pytracer reproduce` at
your repository, a collection, and a candidate branch:

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
