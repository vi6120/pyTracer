# Changelog

## Unreleased

### Added

- **Replay through every framework adapter, without `@op`.** With replay switched
  on (`install_seams()` for LangGraph and CrewAI, `install(replayable=True)` for
  OpenAI Agents and AutoGen), the adapter routes the framework's own model and
  tool calls through pyTracer, so a captured run replays in full and partial
  modes like an `@op`-instrumented one, and streamed model replies replay chunk
  by chunk for LangGraph, OpenAI Agents and AutoGen (CrewAI replays the full reply
  its model call returns). Model calls are matched on what
  was asked, not on ids the framework generates per run. Built on new seams in
  the SDK (`pytracer_sdk.seam`).
- **AutoGen captures model calls and tool output.** Tool spans get their result
  from AutoGen's result message; model calls are recorded from the call events
  the model clients log (or rebuilt from AutoGen's replies for a client that logs
  none); a GenAI-instrumented client keeps its prompt and reply, read from the
  OpenTelemetry log records it sends them as (P7).

### Fixed

- **The gateway keeps writing after a Redis outage.** One error from the ingest
  queue (for example Redis restarting) ended the gateway's background writer for
  good: ingest kept answering 202 while nothing reached storage until the gateway
  was restarted (the spans waited in Redis). The writer now logs the error, backs
  off and retries, keeps a batch it already took until it is written and
  acknowledged, and recreates the Redis consumer group if Redis came back without
  its data; Redis commands retry once over a fresh connection. Shutdown no longer
  hangs on an unreachable queue (`PYTRACER_GATEWAY_SHUTDOWN_TIMEOUT`), retried
  errors are counted (`writer_errors`, `pytracer_ingest_writer_errors_total`), and
  the compose health check probes `/readyz` instead of `/healthz`.
- **Recorded values keep their type.** `Decimal`, `datetime`, `date`, `time`,
  `timedelta`, `UUID`, `bytes`, enums, tuples, sets, dataclasses and pydantic
  models in span inputs and outputs are tagged when serialized and restored on
  load, on the JSONL, gateway and store paths, so a replay computes with the same
  types the agent saw. A pydantic model or dataclass returned by a call is kept as
  that object (it was flattened to a dict before), and a rebuilt pydantic model
  keeps which fields were set. Recordings made before this release must be
  re-recorded to benefit. Only serialized data is decoded (a recording, a store
  row, the gateway queue): a live value that looks like a tag, such as an end
  user's dict, is recorded as that dict. Classes, and recorded exception types,
  are only rebuilt from packages the process has already imported, so a recording
  never imports a module; and the gateway stores payloads exactly as received,
  including when it requeues dead letters.
- **Client-side span loss is visible (L1).** A full SDK queue or a failed delivery
  is logged as a rate-limited warning and passed to an optional
  `on_drop(reason, count, error)` callback set through `configure()`.
- **Concurrent same-input calls replay by caller (R7).** Same-input calls racing
  on different threads each get the response recorded under their own calling op;
  only calls with an identical path are still reported as ambiguous.
- **AutoGen: one trace per team run (P2).** The adapter follows the links AutoGen
  puts between its per-message traces, records a team run as one trace, and nests
  it under an open pyTracer span like the other adapters.
- **The gateway reports its real version** in `/openapi.json`, `/docs`, the
  startup log and `/healthz` (instead of `0.0.1`).

### Documentation

- Using several frameworks in one environment: all four install together, with
  the constraints their own dependencies impose (CrewAI caps the OpenAI Agents SDK
  at 0.20.0; AutoGen needs `googleapis-common-protos<1.75.1`) and how to handle
  them. CI now runs every adapter with all four frameworks installed together.
- A per-adapter capability table on the docs page and in the README, and an
  accurate description of what the AutoGen adapter captures.
- One trace per request with `pytracer.start_span(...)`, for every adapter.
- Queue sizing guidance in the deploy guide.

## 0.1.3

First release after the pilot re-tests. All 12 packages move to 0.1.3 together.

**Upgrade order: upgrade the gateway before the SDKs that send to it (the hosted
gateway first).** A gateway on 0.1.3 ignores span fields it does not recognise, so
a 0.1.3 SDK stays compatible with it. A gateway still on 0.1.2 rejects a failed
span that carries `error_module`; a 0.1.3 SDK resends such a span without the
field so it still lands, but the field is stored only once the gateway is on
0.1.3.

### Fixed

- **SDK / gateway wire compatibility (V1).** A failed span is no longer dropped
  against an older gateway: the SDK resends the batch without a field the gateway
  rejects as unknown, and a 0.1.3 gateway ignores unknown fields.
- **Trace store keeps `error_module` (V2 / R4).** A failure captured through the
  gateway replays with its real exception type instead of a generic error; an
  idempotent column backfill upgrades an existing table.
- **Positional replay matching (R12 / V4).** Matching compares argument values,
  not the JSON skeleton, so a different id or a genuinely different question is a
  miss, never the wrong recording served silently.
- **Run verdict by the agent's outcome (V3).** A tool error the agent caught and
  recovered from no longer fails the replay; `no_errors` gains a `scope` (`root`
  default, `any` to also flag a handled child error).
- **Failure reason in `--json` (R3).** The report carries `error_message`, so
  `issues sync` and CI can see which call failed.
- **Honest `reproduce` verdicts (X4, X2 / X5).** `cannot_judge` (a model input
  changed) and `cannot_reproduce` (recordings did not match) exit with code `2`;
  `reproduce` gains `--evaluate-changed-prompts`.
- **Backpressure is not lossy (L1).** The queue-full `429` carries `Retry-After`
  and the SDK retries the backpressured spans instead of dropping the batch.
- **Deploy settings take effect (D1).** The compose file forwards the managed-key,
  rate-limit and queue-size settings, and the gateway logs the active auth backend
  at startup.
- **Full-mode replay blocks loopback (P6).** A framework model call cannot reach a
  local model server during replay; add a loopback host to the allowlist to permit
  it.
- **Framework adapters.** AutoGen captures via global registration with `install()`
  and resolves its tracer lazily (P1 / P2 / P8); CrewAI nests model and tool calls
  under the open agent (P3); the OpenAI Agents adapter folds structural task/turn
  spans (P4); the LangGraph, CrewAI and OpenAI Agents adapters nest a framework run
  under an active pyTracer span, so wrapping a multi-framework request in
  `pytracer.start_span(...)` (or an `@op`) makes it one trace; without that wrapper
  each framework run is still its own trace, and AutoGen does not nest (P5).

### Documentation

- `HTTPTransport.rejected_spans` and the `on_rejection` callback are documented
  (G1 / G5).
- The docs page covers positional matching, the `cannot_judge` / `cannot_reproduce`
  verdicts and exit code 2, `reproduce --evaluate-changed-prompts`, `no_errors`
  `scope`, and the gateway-first upgrade order.
- The deploy guide documents reaching the published stack's stores from the CLI,
  which publish no ports (D2).

### Known limitations

- A model call captured only through a framework adapter cannot be replayed through
  `@op` on tools; replaying an adapter-only run is flagged, not run live (A1,
  roadmap).
- Some GenAI-instrumented libraries send model output to the OpenTelemetry logs
  API, which the AutoGen exporter does not read, so those outputs are not captured
  (P7).

## 0.1.2

Initial public release (12 packages on PyPI).
