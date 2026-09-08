"""Deterministic hashing of tool inputs for mock-store keys.

Two calls with the same logical input must produce the same hash regardless of
dict key ordering, so we canonicalize to sorted-key JSON before hashing.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Stable JSON encoding: sorted keys, no incidental whitespace."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def input_hash(value: Any) -> str:
    """SHA-256 hex digest of a tool input's canonical form."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
