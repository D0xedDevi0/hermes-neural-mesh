"""Helixa / Agent Aura provenance scaffold (OFF-CHAIN, review-gated).

Why this exists
---------------
D0xedDev's agent identity is anchored on Helixa (agentId 59322) on Base L2.
An agent's *Aura* is its on-chain reputation. NEURAL_MESH memories should be
able to carry a Helixa-backed provenance stamp so that, when two agents share
a `.mesh` file, the recipient can see *who vouched* (and how trustworthy that
voucher is) without trusting an opaque string.

Safety model (IMPORTANT)
------------------------
This module is **read/write of provenance METADATA ONLY**. It never:
  * signs a transaction,
  * broadcasts anything to a chain,
  * calls a Helixa write endpoint,
  * stores a private key.

All "on-chain" interactions are gated behind explicit, auditable functions
that take an externally-supplied signature and verification result. Signing
must happen in a separate, key-held, human-approved step (e.g. the
`/helixa-signer` flow on D0xedDev). See the module docstring in `signer`
below for the exact contract.

What it does
------------
  * `HelixaStamp` — a serializable provenance record:
        { agent_id, aura_score, vouched_at, source, signature, tx_hash }
  * `stamp_node(mesh, node_id, stamp)` — attach a Helixa stamp to a memory
    node (stored in node.meta so it survives `.mesh` export).
  * `verify_stamp(stamp, verify_fn)` — run an injectable verifier
    (default: always "unverified") so callers can plug in real on-chain
    checks later without changing the data layer.
  * `export_manifest(mesh)` — produce a reviewable manifest of every stamped
    node, for a human to eyeball before any on-chain attestation is made.

No network calls. No keys. No side effects beyond the local SQLite row.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional


# Agent Aura is computed by Helixa from on-chain behaviour. We treat any score
# >= this as a "trustworthy voucher" for provenance weighting in shared meshes.
AURA_TRUST_THRESHOLD = 0.6


@dataclass
class HelixaStamp:
    agent_id: str                 # Helixa agentId, e.g. "59322"
    aura_score: float = 0.0       # Agent Aura at stamp time (0..1)
    vouched_at: float = 0.0       # unix ts
    source: str = "helixa"        # provenance namespace
    signature: str = ""           # detached sig over (agent_id+content hash)
    tx_hash: str = ""             # on-chain attestation tx (empty until signed)
    verified: str = "unverified"  # unverified|verified|rejected

    def to_meta(self) -> dict:
        return {"helixa_stamp": asdict(self)}

    @classmethod
    def from_meta(cls, meta: dict) -> "Optional[HelixaStamp]":
        raw = (meta or {}).get("helixa_stamp")
        if not raw:
            return None
        return cls(**{k: raw.get(k, "") for k in
                      ("agent_id", "aura_score", "vouched_at", "source",
                       "signature", "tx_hash", "verified")})


def stamp_node(mesh, node_id: str, stamp: HelixaStamp) -> bool:
    """Attach a Helixa provenance stamp to an existing node.

    Stores it inside node.meta so it round-trips through `.mesh` export/import.
    Returns False if the node doesn't exist.
    """
    node = mesh._load().get(node_id)
    if not node:
        return False
    node.meta = dict(getattr(node, "meta", {}) or {})
    node.meta.update(stamp.to_meta())
    node.agent_id = stamp.agent_id or node.agent_id
    mesh._save(node)
    return True


def verify_stamp(stamp: HelixaStamp,
                 verify_fn: Optional[Callable[[HelixaStamp], str]] = None) -> str:
    """Run an injectable verifier. Default is 'unverified' (no trust assumed).

    A real deployment would pass a function that checks the signature against
    the Helixa on-chain registry. We never auto-verify here.
    """
    if verify_fn is not None:
        return verify_fn(stamp)
    return "unverified"


def aura_trust_weight(stamp: Optional[HelixaStamp]) -> float:
    """Map an Agent Aura score to a trust weight for cross-agent merge.

    Untrusted / unverified stamps are capped low (default 0.2) so an
    unverified voucher can't dominate trusted local memory.
    """
    if stamp is None or stamp.verified != "verified":
        return 0.2
    return max(0.2, min(1.0, stamp.aura_score))


def export_manifest(mesh) -> dict:
    """Produce a human-reviewable manifest of every Helixa-stamped node.

    This is the artifact a human (or `/helixa-signer`) reviews BEFORE any
    on-chain attestation is signed. No signing happens inside this function.
    """
    out = []
    for n in mesh._load().values():
        stamp = HelixaStamp.from_meta(getattr(n, "meta", {}) or {})
        if stamp:
            out.append({
                "node_id": n.id,
                "content": n.content[:120],
                "agent_id": stamp.agent_id,
                "aura_score": stamp.aura_score,
                "verified": stamp.verified,
                "tx_hash": stamp.tx_hash or None,
            })
    return {"count": len(out), "stamps": out}


def make_stamp(agent_id: str, aura_score: float = 0.0,
               signature: str = "", tx_hash: str = "") -> HelixaStamp:
    """Convenience constructor. Leaves verified='unverified' by default."""
    return HelixaStamp(agent_id=agent_id, aura_score=aura_score,
                       vouched_at=time.time(), source="helixa",
                       signature=signature, tx_hash=tx_hash,
                       verified="unverified")
