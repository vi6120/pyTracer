# pyTracer benchmarks

Load-test the ingestion gateway with [k6](https://k6.io) while
[Prometheus](https://prometheus.io) scrapes the gateway's own metrics, so you
can line up client-side throughput and latency (k6) against server-side counters
(accepted / written / backpressured) and the gateway's CPU and memory.

Everything runs locally in Docker. Nothing here touches the live
`gateway.pytracer.com` box.

## What is in here

| File | Purpose |
| --- | --- |
| `docker-compose.bench.yml` | Gateway (built with the `[metrics]` extra) + Redis + ClickHouse + Prometheus, plus optional Grafana and an in-network k6 runner. |
| `k6/ingest.js` | The load script: posts schema-valid span batches to `POST /v1/spans`, either ramping virtual users (default) or holding a constant rate for a long soak (`MODE=soak`). |
| `prometheus.yml` | Scrape config pointed at the gateway's `/metrics`. |
| `grafana/datasource.yml` | Pre-wires Grafana to the benchmark Prometheus (optional profile). |

## Prerequisites

- Docker Desktop running.
- k6, to drive load from the host: `brew install k6`. (Or skip it and use the
  in-network k6 runner shown below.)

## Run it

From the repository root:

```bash
# 1. Bring up the gateway, its stores, and Prometheus (first run builds the image).
docker compose -f benchmarks/docker-compose.bench.yml up -d --build

# 2. Drive load. Either from the host:
k6 run benchmarks/k6/ingest.js
#    ...or inside the compose network, no host k6 needed:
docker compose -f benchmarks/docker-compose.bench.yml --profile load run --rm k6

# 3. Read the server-side metrics while it runs:
open http://localhost:9090          # Prometheus UI (PromQL below)
```

Optional Grafana (anonymous viewer, datasource already wired):

```bash
docker compose -f benchmarks/docker-compose.bench.yml --profile grafana up -d
open http://localhost:3000
```

Tear down (including volumes):

```bash
docker compose -f benchmarks/docker-compose.bench.yml down -v
```

## Tuning the load

Pass these as environment variables to k6 (`-e NAME=value`, or export them):

| Variable | Default | Meaning |
| --- | --- | --- |
| `MAX_VUS` | `200` | Peak virtual users at the top of the ramp. |
| `BATCH` | `20` | Spans per request (so accepted spans/sec is `requests/sec x BATCH`). |
| `PYTRACER_URL` | `http://localhost:8080` | Gateway base URL. |
| `PYTRACER_API_KEY` | `devkey` | Value sent as `X-API-Key`. |

Example, a heavier run:

```bash
k6 run -e MAX_VUS=500 -e BATCH=50 benchmarks/k6/ingest.js
```

### Soak: a constant rate for a long time

The ramp finds where latency bends; a soak shows the gateway sustains a rate:
latency stays flat, every accepted span is written, nothing is dead-lettered,
and memory does not grow. Pick a rate below the knee the ramp found:

```bash
k6 run -e MODE=soak -e RATE=180 -e DURATION=30m benchmarks/k6/ingest.js
#    ...or in the compose network:
docker compose -f benchmarks/docker-compose.bench.yml --profile load run --rm \
  -e MODE=soak -e RATE=180 -e DURATION=30m k6
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODE` | `ramp` | `ramp` or `soak`. |
| `RATE` | `50` | Soak: requests per second (spans/sec is `RATE x BATCH`). |
| `DURATION` | `30m` | Soak: how long to hold the rate. |

Afterwards, compare the gateway's own counters (`curl localhost:8080/v1/stats`):
`accepted` and `written` must be equal once the queue drains, and
`dead_letters` must be 0.

Gateway-side knobs worth sweeping (set them in `docker-compose.bench.yml` under
the `gateway` service, then `up -d` again):

- `PYTRACER_GATEWAY_QUEUE_MAXSIZE` (default 10000): raise it if you see
  backpressure (429s) before the writer is saturated.
- `PYTRACER_GATEWAY_BATCH_SIZE` (default 256) and `PYTRACER_GATEWAY_FLUSH_INTERVAL`
  (default 0.5s): the write-through batch size and flush cadence.
- `PYTRACER_GATEWAY_QUEUE_BACKEND=memory`: compare the in-process queue against
  Redis to isolate the queue's overhead.

## Reading the results

**From k6 (client side):** watch `http_reqs` (requests/sec), `http_req_duration`
(p95/p99 end-to-end latency), `http_req_failed` (error rate), and the custom
`pytracer_spans_accepted` / `pytracer_spans_rejected` / `pytracer_spans_backpressured`
counters, which are read from the gateway's answer (a `202` can still carry spans
it rejected). The run fails its thresholds if p95 exceeds 500 ms, the error rate
exceeds 1%, or any span is rejected.

**From Prometheus (server side):** paste these into the query bar at
`http://localhost:9090`.

Accept throughput, spans/sec:
```promql
rate(pytracer_spans_accepted_total[30s])
```

Write-through throughput to the trace store, spans/sec (this is where a storage
bottleneck shows up as a gap below the accept rate):
```promql
rate(pytracer_spans_written_total[30s])
```

Server-side request latency, p95 seconds:
```promql
histogram_quantile(0.95, sum(rate(pytracer_ingest_request_seconds_bucket[30s])) by (le))
```

Backpressure and dead-letters (should stay flat; a rising line means the queue
filled or writes failed):
```promql
rate(pytracer_spans_backpressured_total[30s])
rate(pytracer_spans_storage_dlq_total[30s])
```

Gateway CPU seconds/sec and resident memory:
```promql
rate(process_cpu_seconds_total[30s])
process_resident_memory_bytes
```

## Recorded results

Measured on 2026-10-26 on an Apple M1 Pro (10 cores, 16 GB) running Docker
Desktop with 10 CPUs and 8 GB, the bench stack as committed (one gateway, Redis
queue, single-node ClickHouse), batches of 20 spans:

| Run | Result |
| --- | --- |
| Ramp to 200 VUs (2 min) | The writer's ceiling: about 3,600 spans/s written. Past it the queue filled and the gateway answered 429 (backpressure, as designed) for 75% of requests at the peak; every accepted span was written, none dead-lettered. |
| Constant 150 and 180 req/s (1 min each) | p95 4.7 ms and 4.9 ms, no errors, no backpressure. |
| **Soak: 180 req/s (3,600 spans/s) for 30 min** | 324,001 requests, 6,480,020 spans, 0 failed, 0 rejected, 0 backpressured; p95 5.6 ms, p90 4.7 ms, median 3.0 ms, max 329 ms. Gateway totals after the run: accepted = written = 7,424,180 (the soak plus the earlier runs), 0 dead letters, 0 writer errors; ClickHouse holds 7,424,180 rows with 7,424,180 distinct span ids (none lost, none twice). Gateway memory 99 MB at the start, 75 MB at the end. |

These are one laptop's numbers, not a capacity promise: size a deployment with
the ramp and a soak on its own hardware.

## Interpreting the shape

- **Accept rate high, write rate tracking it, latency flat:** healthy. Push
  `MAX_VUS` / `BATCH` up until something bends.
- **Accept rate flat while backpressure climbs:** the queue is full. Raise
  `PYTRACER_GATEWAY_QUEUE_MAXSIZE`, or the writer cannot keep up (see next).
- **Write rate plateaus below accept rate:** ClickHouse write-through is the
  bottleneck. Tune `BATCH_SIZE` / `FLUSH_INTERVAL`, or it is disk/CPU on the box.
- **p95 latency rises with VUs but accept rate is flat:** you are past the knee;
  back off to find the sustainable throughput.
