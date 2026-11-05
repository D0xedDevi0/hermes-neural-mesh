"""Reusable server hardening helpers for the optional Flask API.

Pure stdlib and framework-agnostic so the core test suite can validate security
behavior without importing Flask.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from hmac import compare_digest
from typing import Mapping


def safe_path(base_dir: str, user_path: str, default_name: str = "mesh_export.mesh") -> str:
    """Resolve ``user_path`` under ``base_dir`` or raise ValueError.

    Absolute paths and ``..`` traversal are rejected. Parent directories are
    created for accepted paths. This keeps import/export endpoints inside a
    purpose-built scratch directory instead of arbitrary filesystem locations.
    """
    base = os.path.abspath(base_dir)
    raw = str(user_path or default_name)
    if os.path.isabs(raw):
        raise ValueError("absolute paths are not allowed")
    candidate = os.path.abspath(os.path.join(base, raw))
    if candidate != base and not candidate.startswith(base + os.sep):
        raise ValueError("path escapes allowed directory")
    os.makedirs(os.path.dirname(candidate), exist_ok=True)
    return candidate


def auth_ok(headers: Mapping[str, str], token: str) -> bool:
    """Return True when auth is disabled or request headers carry the token."""
    expected = str(token or "")
    if not expected:
        return True
    auth = str(headers.get("Authorization", ""))
    api_key = str(headers.get("X-API-Key", ""))
    bearer = auth[7:] if auth.lower().startswith("bearer ") else ""
    return compare_digest(bearer, expected) or compare_digest(api_key, expected)


class RateLimiter:
    """Tiny sliding-window rate limiter keyed by client id."""

    def __init__(self, limit: int = 120, window_seconds: int = 60):
        self.limit = int(limit)
        self.window_seconds = int(window_seconds)
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.time() if now is None else float(now)
        q = self._events[str(key or "unknown")]
        cutoff = now - self.window_seconds
        while q and q[0] <= cutoff:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        return True


def origin_allowed(origin: str, allowed_origins: set[str]) -> bool:
    """CORS helper: empty origin is same-origin/non-browser and allowed."""
    if not origin:
        return True
    return origin in allowed_origins


__all__ = ["RateLimiter", "auth_ok", "origin_allowed", "safe_path"]
