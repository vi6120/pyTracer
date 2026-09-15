// k6 load test for the pyTracer ingestion gateway.
//
// Ramps virtual users against POST /v1/spans with schema-valid span batches and
// checks that the gateway accepts them (202). While this runs, Prometheus
// scrapes the gateway's /metrics so you can line up server-side counters
// (accepted / written / backpressured) against k6's client-side throughput and
// latency.
//
// Run against the local bench stack (see benchmarks/README.md):
//   k6 run benchmarks/k6/ingest.js
//
// Tunable via environment variables:
//   PYTRACER_URL      gateway base URL   (default http://localhost:8080)
//   PYTRACER_API_KEY  X-API-Key value    (default devkey)
//   BATCH             spans per request  (default 20)
//   MAX_VUS           peak virtual users (default 200)

import http from 'k6/http';
import { check } from 'k6';
import { Counter } from 'k6/metrics';

const BASE = __ENV.PYTRACER_URL || 'http://localhost:8080';
const API_KEY = __ENV.PYTRACER_API_KEY || 'devkey';
const BATCH = parseInt(__ENV.BATCH || '20', 10);
const MAX_VUS = parseInt(__ENV.MAX_VUS || '200', 10);

const spansAccepted = new Counter('pytracer_spans_accepted');
const spansBackpressured = new Counter('pytracer_spans_backpressured');

export const options = {
  scenarios: {
    ramp: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '30s', target: Math.max(1, Math.ceil(MAX_VUS / 4)) },
        { duration: '1m', target: MAX_VUS },
        { duration: '30s', target: 0 },
      ],
      gracefulRampDown: '10s',
    },
  },
  thresholds: {
    // A well-sized gateway should accept nearly everything and stay fast.
    http_req_failed: ['rate<0.01'],
    http_req_duration: ['p(95)<500'],
    checks: ['rate>0.99'],
  },
};

const HEX = '0123456789abcdef';

function randHex(len) {
  let out = '';
  for (let i = 0; i < len; i++) {
    out += HEX[Math.floor(Math.random() * 16)];
  }
  return out;
}

function makeSpan() {
  const start = Date.now() * 1e6; // epoch nanoseconds
  const end = start + Math.floor(Math.random() * 50 + 1) * 1e6;
  return {
    trace_id: randHex(32),
    span_id: randHex(16),
    name: 'llm.chat',
    kind: 'llm',
    start_time_ns: start,
    end_time_ns: end,
    status: 'ok',
    attributes: { model: 'bench-demo', batch: BATCH },
  };
}

function makeBatch(n) {
  const batch = new Array(n);
  for (let i = 0; i < n; i++) {
    batch[i] = makeSpan();
  }
  return batch;
}

export default function () {
  const res = http.post(`${BASE}/v1/spans`, JSON.stringify(makeBatch(BATCH)), {
    headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY },
  });

  const ok = check(res, {
    'status is 202 accepted': (r) => r.status === 202,
    'not rejected (schema)': (r) => r.status !== 400,
    'authorized': (r) => r.status !== 401,
  });

  if (res.status === 202) {
    spansAccepted.add(BATCH);
  } else if (res.status === 429) {
    // Backpressure or rate limit: the gateway asked us to slow down.
    spansBackpressured.add(BATCH);
  }

  return ok;
}
