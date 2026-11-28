# From AutoGen to Microsoft Agent Framework, keeping your tests

Microsoft has archived AutoGen (no bug or security fixes) and points its users to
Microsoft Agent Framework. This recipe ports a small agent across and shows what
pyTracer can and cannot judge along the way. It runs with no API key: both versions
of the agent use a scripted model client.

## The agent

A support agent that refunds an order with one tool, `refund_order`, and returns a
receipt. The tool and the receipt live in `refunds.py`, which does not import either
framework, so the port leaves it unchanged. The receipt has a bug: it drops the
cents, so a $49.99 refund is reported as 4900 cents.

| File | What it is |
|---|---|
| [`variants/agent_autogen.py`](variants/agent_autogen.py) | The agent on AutoGen 0.7 (`AssistantAgent`) |
| [`variants/agent_maf.py`](variants/agent_maf.py) | The same agent on Agent Framework (`Agent`) |
| [`variants/refunds_buggy.py`](variants/refunds_buggy.py), [`refunds_fixed.py`](variants/refunds_fixed.py) | The tool and the receipt, before and after the fix |
| [`collection.yaml`](collection.yaml) | The test: one refund of the right order and amount, and the right receipt |
| [`capture.py`](capture.py) | Records one run of the agent: the recording a replay freezes |

The test checks what the agent does (the `writes` assertion: `refund_order` called
once with `#W1001` and `49.99`) and what it returns (the receipt). Neither depends on
the framework, which is why the same file tests both versions.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "pytracer-cli[runner]" pytracer-autogen pytracer-agent-framework \
  "autogen-agentchat==0.7.5" "autogen-ext==0.7.5" "agent-framework-core>=1.26.0"
./run_demo.sh
```

It builds a throwaway git repository with three branches: `main` (the agent on
AutoGen, with the bug), `port-maf` (ported to Agent Framework; the bug comes along)
and `fix-on-maf` (the bug fixed on the port). Then, in four steps:

**1. Record the failing run on AutoGen.** The receipt says 4900 cents.

**2. Replay the AutoGen recording on the port:**

```
SCENARIO                     OUTCOME
refund-one-order             cannot_judge

refund-one-order: The branch calls model 'ScriptedClient', which the recording never called. A frozen replay cannot judge a change to the calls themselves; re-record the scenario on this branch to judge it.
```

A port changes every model call (another client, another message format), so an
AutoGen recording has no answer for the ported agent's calls. pyTracer says so
instead of guessing. Recordings do not carry over a port.

**3. Record the port once, and run the unchanged test on it:**

```
  [FAIL] refund-one-order
      receipt (returned {'status': 'refunded', 'order_id': '#W1001', 'amount_cents': 4900})
0/1 passed
```

The test does carry over. The port makes the same refund (the `writes` and
`no_errors` checks pass) and fails only the receipt, exactly as the AutoGen agent
did: the port kept the behaviour, bug included.

**4. Fix the bug on the port, judged against the port's own recording:**

```
SCENARIO                     OUTCOME
refund-one-order             fixed
```

From here on, `pytracer reproduce` works as before, on the new framework.

## Porting your own agent

1. Before the port, make sure each important scenario has assertions on what the
   agent does and returns (`writes`, `returns`, `no_errors`), not only a recording.
   Those are what carry over.
2. Port the agent on a branch, and swap `pytracer_autogen.install(...)` for
   `pytracer_agent_framework.install()`.
3. Record each scenario once on the ported branch (`pytracer record`, or your own
   capture) and run the collection: the same checks should pass, or fail, as before.
4. Judge later fixes on the new framework with `pytracer reproduce` against those
   recordings.
