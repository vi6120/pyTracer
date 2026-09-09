"""Canonical hashing of op inputs.

Uses the same algorithm as the mock store (sorted-key JSON, SHA-256) so mocks
recorded there and mocks built from a trace here resolve to identical keys.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def input_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
