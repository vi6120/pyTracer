"""Per-project API-key authentication.

Keys are resolved to a project before any span is processed, so an invalid or
missing key is rejected up front (guide section 3: reject before any processing).
"""

from __future__ import annotations

API_KEY_HEADER = "X-API-Key"


class APIKeyAuth:
    """Resolves an API key to its project, or ``None`` if unknown."""

    def __init__(self, keys: dict[str, str]) -> None:
        self._keys = dict(keys)

    def project_for(self, api_key: str | None) -> str | None:
        """Return the project an API key maps to, or None if the key is unknown."""
        if not api_key:
            return None
        return self._keys.get(api_key)

    def __len__(self) -> int:
        return len(self._keys)
