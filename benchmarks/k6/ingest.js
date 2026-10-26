// k6 load test for the pyTracer ingestion gateway.
//
// Posts schema-valid span batches to POST /v1/spans and checks that the gateway
// accepts every span (202 with the whole batch accepted, none rejected). While
// this runs, Prometheus scrapes the gateway's /metrics so you can line up
// server-side counters (accepted / written / backpressured) against k6's
// client-side throughput and latency.
//
// Two modes:
//   ramp (default)  ramps virtual users up and down over 2 minutes, to find
//                   where latency bends
//   soak            a constant request rate for a long time (MODE=soak), to show
//                   the gateway sustains it: no growing latency, no loss, no leak
//
// Run against the local bench stack (see benchmarks/README.md):
//   k6 run benchmarks/k6/ingest.js
//   k6 run -e MODE=soak -e RATE=100 -e DURATION=30m benchmarks/k6/ingest.js
//
// Tunable via environment variables:
//   PYTRACER_URL      gateway base URL                  (default http://localhost:8080)
//   PYTRACER_API_KEY  X-API-Key value                   (default devkey)
//   BATCH             spans per request                 (default 20)
//   MAX_VUS           peak virtual users                (default 200)
//   MODE              ramp or soak                      (default ramp)
//   RATE              soak: requests per second         (default 50)
//   DURATION          soak: how long to hold the rate   (default 30m)

import http from 'k6/http';
import { check } from 'k6';
import { Counter } from 'k6/metrics';

const BASE = __ENV.PYTRACER_URL || 'http://localhost:8080';
const API_KEY = __ENV.PYTRACER_API_KEY || 'devkey';
const BATCH = parseInt(__ENV.BATCH || '20', 10);
const MAX_VUS = parseInt(__ENV.MAX_VUS || '200', 10);
const MODE = __ENV.MODE || 'ramp';
const RATE = parseInt(__ENV.RATE || '50', 10);
const DURATION = __ENV.DURATION || '30m';

// Counted from the gateway's answer, not assumed from the status code: a 202 can
// still carry spans the gateway rejected (and dead-lettered).
const spansAccepted = new Counter('pytracer_spans_accepted');
const spansRejected = new Counter('pytracer_spans_rejected');
const spansBackpressured = new Counter('pytracer_spans_backpressured');

const SCENARIOS = {
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
  soak: {
    executor: 'constant-arrival-rate',
    rate: RATE,
    timeUnit: '1s',
    duration: DURATION,
    preAllocatedVUs: Math.min(MAX_VUS, Math.max(10, RATE)),
    maxVUs: MAX_VUS,
  },
};

if (!(MODE in SCENARIOS)) {
  throw new Error(`MODE must be ramp or soak, got ${MODE}`);
}

export const options = {
  scenarios: { [MODE]: SCENARIOS[MODE] },
  thresholds: {
    // A well-sized gateway should accept nearly everything and stay fast.
    http_req_failed: ['rate<0.01'],
    http_req_duration: ['p(95)<500'],
    checks: ['rate>0.99'],
    // Every span sent must be valid; a rejection means the script is wrong.
    pytracer_spans_rejected: ['count==0'],
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
    // The trace store requires the git context; spans without it are rejected.
    resource: { service_name: 'bench', branch: 'main', commit: 'bench' },
  };
}

function makeBatch(n) {
  const batch = new Array(n);
  for (let i = 0; i < n; i++) {
    batch[i] = makeSpan();
  }
  return batch;
}

function body(res) {
  try {
    return res.json();
  } catch (e) {
    return {};
  }
}

export default function () {
  const res = http.post(`${BASE}/v1/spans`, JSON.stringify(makeBatch(BATCH)), {
    headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY },
  });
  const answer = body(res);
  const accepted = answer.accepted || 0;
  const rejected = answer.rejected || 0;

  const ok = check(res, {
    'status is 202 accepted': (r) => r.status === 202,
    'every span accepted': () => accepted === BATCH,
    'no span rejected (schema)': () => rejected === 0,
    'authorized': (r) => r.status !== 401,
  });

  spansAccepted.add(accepted);
  spansRejected.add(rejected);
  if (res.status === 429) {
    // Backpressure or rate limit: the gateway asked us to slow down.
    spansBackpressured.add(answer.backpressured || BATCH);
  }

  return ok;
}
