"""
Secret Sanitization Utilities

Provides helpers to mask sensitive values (tokens, passwords, secrets) so they
never appear in logs, status payloads, or API responses.

Usage:
    from src.core.sanitize import sanitize_value, sanitize_dict

    # Mask a single value
    sanitize_value("eyJhbGciOiJIUz...", "token")  # -> "eyJh***z..."

    # Mask all sensitive keys in a dict
    sanitize_dict({"url": "http://ha:8123", "token": "secret123"})
    # -> {"url": "http://ha:8123", "token": "secr***"}
"""

import re
from typing import Any

# Keys whose values should be masked
SENSITIVE_KEYS: frozenset[str] = frozenset({
    "token",
    "access_token",
    "refresh_token",
    "password",
    "secret",
    "client_secret",
    "api_key",
    "authorization",
})

# Patterns that indicate a value is sensitive regardless of key name
_BEARER_RE = re.compile(r"Bearer\s+\S+", re.IGNORECASE)


def mask_string(value: str, visible_chars: int = 4) -> str:
    """Mask a string, keeping only the first `visible_chars` characters visible.

    Args:
        value: The string to mask.
        visible_chars: Number of leading characters to keep visible.

    Returns:
        Masked string, e.g. ``"eyJh***"``.
    """
    if len(value) <= visible_chars:
        return "***"
    return value[:visible_chars] + "***"


def sanitize_value(value: str, key: str = "") -> str:
    """Sanitize a single value based on its key name.

    If the key matches a known sensitive key the value is masked.
    Bearer tokens embedded in header-style strings are always masked.

    Args:
        value: The value to inspect/mask.
        key: The associated key name (lowered internally).

    Returns:
        Original value or masked version.
    """
    if key.lower() in SENSITIVE_KEYS:
        return mask_string(value)
    # Mask inline Bearer tokens
    if _BEARER_RE.search(value):
        return _BEARER_RE.sub(lambda m: f"Bearer {mask_string(m.group().split()[-1])}", value)
    return value


def sanitize_dict(data: dict[str, Any], *, _depth: int = 0) -> dict[str, Any]:
    """Return a shallow copy of *data* with sensitive values masked.

    Recurses into nested dicts (up to depth 10 to prevent infinite loops).

    Args:
        data: The dictionary to sanitize.

    Returns:
        New dict with sensitive values masked.
    """
    if _depth > 10:
        return data

    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = sanitize_value(value, key)
        elif isinstance(value, dict):
            result[key] = sanitize_dict(value, _depth=_depth + 1)
        elif isinstance(value, list):
            result[key] = [
                sanitize_dict(item, _depth=_depth + 1) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value
    return result
