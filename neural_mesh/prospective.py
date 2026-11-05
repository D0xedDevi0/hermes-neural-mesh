"""Prospective memory retrieval — the "memory of the future" lane.

Real cognitive type (episodic future thinking / intentions), and a documented
whitespace lane in the agentic-memory world: systems store the PAST; almost
none store INTENTIONS and surface them *before* they're due.

A prospective memory is a MemoryNode written with ``prospective_at=<unix ts>``
(accepted by ``Mesh.add(...)`` today) representing a future commitment:

    mesh.add("Follow up with Maya about the Acme deployment on Tuesday",
             type=MemoryType.PROSPECTIVE, prospective_at=<ts>)

This module adds the retrieval half that never existed:
  - ``due(now, horizon_sec)``   -> intentions due within a lookahead window
  - ``due_rank(now, k)``        -> top-k upcoming intentions by proximity+trust
  - ``snooze(node, to, mesh)``  -> push a due intention out (re-future it)

Pure stdlib, no deps. Same house rules as the rest of the core: honest,
no fabrication, works against the real ``Mesh`` API.
"""

from __future__ import annotations

import time
from typing import Iterable, Optional

from .core import MemoryType

# Link key under which Mesh.add stores the due timestamp.
PROSPECTIVE_LINK = "__prospective_at__"

# A prospective memory older than this is assumed forgotten/expired and is
# not surfaced as "due" — it becomes a historical record instead.
DEFAULT_EXPIRE_SEC = 7 * 24 * 3600  # 7 days past due before it's stale


def _prospective_at(node) -> Optional[float]:
    """Return the due timestamp for a node, or None if it isn't prospective."""
    links = getattr(node, "links", {}) or {}
    raw = links.get(PROSPECTIVE_LINK)
    if raw is None:
        return None
    return float(raw)


def upcoming(mesh, now: Optional[float] = None,
             horizon_sec: float = 24 * 3600,
             expire_sec: float = DEFAULT_EXPIRE_SEC) -> list:
    """Intentions due within ``horizon_sec`` of ``now`` (default 24h).

    Returns nodes sorted by proximity (soonest first), excluding ones long
    past-due (treated as forgotten) and quarantined/flagged nodes.
    """
    now = now if now is not None else time.time()
    out = []
    for node in mesh._load().values():
        at = _prospective_at(node)
        if at is None:
            continue
        delta = at - now
        # only surface things not yet due OR recently due (within expire window)
        if delta > horizon_sec:
            continue
        if delta < -expire_sec:
            continue
        # skip quarantined / poisoned content
        if getattr(node, "lane", "") == "quarantine":
            continue
        out.append((node, delta))
    out.sort(key=lambda x: x[1])  # soonest first
    return [n for n, _ in out]


def due_rank(mesh, now: Optional[float] = None, k: int = 5,
             horizon_sec: float = 24 * 3600,
             trust_floor: float = 0.2) -> list:
    """Top-k upcoming intentions ranked by proximity * trust (a recall signal
    that keeps high-trust commitments high even slightly farther out)."""
    now = now if now is not None else time.time()
    scored = []
    for node in upcoming(mesh, now=now, horizon_sec=horizon_sec):
        at = _prospective_at(node)
        proximity = 1.0 / (1.0 + abs(at - now))   # closer = higher
        trust = max(float(getattr(node, "trust", 1.0) or 1.0), 0.0)
        if trust < trust_floor:
            continue
        scored.append((proximity * trust, node, at))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [n for _, n, _ in scored[:k]]


def snooze(mesh, node_id: str, to_ts: float,
           now: Optional[float] = None) -> bool:
    """Re-future a prospective memory by rewriting its due timestamp.

    Returns True on success. Raises KeyError if the node isn't prospective.
    """
    nodes = mesh._load()
    node = nodes.get(node_id)
    if node is None:
        raise KeyError(node_id)
    links = dict(getattr(node, "links", {}) or {})
    if PROSPECTIVE_LINK not in links:
        raise KeyError(f"{node_id} is not a prospective memory")
    links[PROSPECTIVE_LINK] = float(to_ts)
    node.links = links
    mesh._save(node)
    mesh._invalidate_cache()
    return True


def expired(mesh, now: Optional[float] = None,
            expire_sec: float = DEFAULT_EXPIRE_SEC) -> list:
    """Past-due, assumed-forgotten intentions (kept as history, not surfaced)."""
    now = now if now is not None else time.time()
    out = []
    for node in mesh._load().values():
        at = _prospective_at(node)
        if at is not None and (now - at) > expire_sec:
            out.append(node)
    return out
