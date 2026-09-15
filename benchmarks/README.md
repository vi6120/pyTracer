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
| `k6/ingest.js` | The load script: ramps virtual users posting schema-valid span batches to `POST /v1/spans`. |
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
`pytracer_spans_accepted` / `pytracer_spans_backpressured` counters. The run
fails its thresholds if p95 exceeds 500 ms or the error rate exceeds 1%.

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

## Interpreting the shape

- **Accept rate high, write rate tracking it, latency flat:** healthy. Push
  `MAX_VUS` / `BATCH` up until something bends.
- **Accept rate flat while backpressure climbs:** the queue is full. Raise
  `PYTRACER_GATEWAY_QUEUE_MAXSIZE`, or the writer cannot keep up (see next).
- **Write rate plateaus below accept rate:** ClickHouse write-through is the
  bottleneck. Tune `BATCH_SIZE` / `FLUSH_INTERVAL`, or it is disk/CPU on the box.
- **p95 latency rises with VUs but accept rate is flat:** you are past the knee;
  back off to find the sustainable throughput.
