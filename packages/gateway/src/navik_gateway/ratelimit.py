"""Per-key rate limiting for the ingest path.

A token bucket per API key: each key refills at ``rate`` tokens per second up to
a ceiling of ``burst`` tokens, and a request costs one token. This smooths
bursts while capping sustained throughput per client, so one noisy key cannot
crowd out the rest.

The bucket state is in-process, so limits are enforced per gateway instance.
That is the right scope for a single-instance beta; a multi-instance deployment
would move the counters into Redis so the limit is shared. Limiting is disabled
(every request allowed) when ``rate`` is zero or negative.
"""

from __future__ import annotations

import time
from threading import Lock


class TokenBucketLimiter:
    """A per-key token bucket. Thread-safe; cheap when disabled."""

    def __init__(self, rate: float, burst: float) -> None:
        self._rate = rate
        # A burst below one token could never admit a request; floor it at the rate.
        self._burst = burst if burst > 0 else rate
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_ts)
        self._lock = Lock()

    @property
    def enabled(self) -> bool:
        return self._rate > 0

    def allow(self, key: str, cost: float = 1.0) -> bool:
        """Charge ``cost`` tokens to ``key``. Returns False when the bucket is dry."""
        if self._rate <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (self._burst, now))
            # Refill for the elapsed time, capped at the burst ceiling.
            tokens = min(self._burst, tokens + (now - last) * self._rate)
            if tokens >= cost:
                self._buckets[key] = (tokens - cost, now)
                return True
            self._buckets[key] = (tokens, now)
            return False
