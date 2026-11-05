"""Helixa on-chain attestation gateway — sign locally, record in mesh.

SAFETY CONTRACT (NON-NEGOTIABLE)
---------------------------------
This module NEVER stores, logs, prints, or exposes a private key. Signing is
performed via an **injectable ``sign_fn``** supplied by the caller. The caller
is responsible for loading the key from a secure path and passing only the
signing function — this module never touches the key file.

All on-chain contract interaction is gated behind an injectable
``broadcast_fn``. Without it, attestations are signed and recorded locally
but NOT broadcast. This keeps the module safe to import and test without any
network or key access.

Architecture
------------
::

    caller loads key → sign_fn(message) → signature
    attest_node(mesh, node_id, agent_id, sign_fn, broadcast_fn?)
        ├─ build_attestation_message(node, agent_id)
        ├─ sign_attestation(message, sign_fn)
        ├─ (optional) broadcast_attestation(tx, broadcast_fn)
        └─ record_onchain_attestation(mesh, node_id, sig, tx_hash)
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable

from .helixa_provenance import HelixaStamp, stamp_node


# ── message construction ──────────────────────────────────────────────

@dataclass
class AttestationMessage:
    """Canonical message that an agent signs to attest a memory node.

    Following EIP-712-ish structured signing: we hash (agent_id ‖ node_id ‖
    content_hash ‖ timestamp) so the signature commits to a specific
    agent→node pair at a point in time.
    """
    agent_id: str
    node_id: str
    content_hash: str       # keccak256 or sha256 of the node content
    timestamp: int          # unix seconds
    domain: str = "NEURAL_MESH.helixa.attest.v1"

    def signing_hash(self) -> str:
        """Deterministic 256-bit hash of all fields for signing."""
        payload = "‖".join([
            self.domain,
            self.agent_id,
            self.node_id,
            self.content_hash,
            str(self.timestamp),
        ])
        return hashlib.sha256(payload.encode()).hexdigest()


def build_attestation_message(node, agent_id: str) -> AttestationMessage:
    """Construct the canonical attestation message for a memory node.

    The caller signs this message; the signature is then recorded in the
    node's Helixa stamp.
    """
    content = getattr(node, "content", "")
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    return AttestationMessage(
        agent_id=str(agent_id),
        node_id=str(node.id),
        content_hash=content_hash,
        timestamp=int(time.time()),
    )


# ── signing ───────────────────────────────────────────────────────────

def sign_attestation(message: AttestationMessage,
                     sign_fn: Callable[[str], str]) -> str:
    """Sign the attestation message hash.

    ``sign_fn`` receives the hex signing hash (``message.signing_hash()``)
    as a string and MUST return the signature as a hex string.

    This module NEVER calls ``sign_fn`` with the private key — the caller
    injects a closure that holds the key.
    """
    return sign_fn(message.signing_hash())


# ── recording ─────────────────────────────────────────────────────────

def record_onchain_attestation(mesh,
                                node_id: str,
                                signature: str,
                                tx_hash: str = "",
                                agent_id: str = "",
                                aura_score: float = 0.0) -> dict[str, Any]:
    """Record a signed attestation in the node's Helixa stamp.

    After calling this, the node carries:
    - ``helixa_stamp.signature`` — the detached signature
    - ``helixa_stamp.tx_hash`` — the on-chain transaction (if broadcast)
    - ``helixa_stamp.verified`` — set to ``"onchain_attested"``

    Returns a status dict suitable for API responses.
    """
    stamp = HelixaStamp(
        agent_id=str(agent_id),
        aura_score=float(aura_score),
        vouched_at=time.time(),
        source="helixa-onchain",
        signature=str(signature),
        tx_hash=str(tx_hash),
        verified="onchain_attested",
    )
    ok = stamp_node(mesh, node_id, stamp)
    return {
        "ok": ok,
        "node_id": node_id,
        "agent_id": agent_id,
        "verified": stamp.verified,
        "tx_hash": tx_hash or None,
        "vouched_at": stamp.vouched_at,
    }


# ── full flow ─────────────────────────────────────────────────────────

def attest_node(mesh,
                node_id: str,
                agent_id: str,
                sign_fn: Callable[[str], str],
                broadcast_fn: Callable[[str, str], str] | None = None,
                aura_score: float = 0.0) -> dict[str, Any]:
    """Full attestation flow: build → sign → (broadcast) → record.

    Parameters
    ----------
    mesh : Mesh
        The active mesh instance.
    node_id : str
        Node to attest.
    agent_id : str
        Helixa agent ID performing the attestation.
    sign_fn : callable
        ``sign_fn(signing_hash: str) -> signature: str``.
        The caller injects a closure that loads the key from a secure path.
    broadcast_fn : callable or None
        ``broadcast_fn(signature: str, message_json: str) -> tx_hash: str``.
        If None, the attestation is signed + recorded locally but NOT
        broadcast to chain. This is the safe default for testing.
    aura_score : float
        Agent Aura score at attestation time (0..1).

    Returns
    -------
    dict with keys: ok, node_id, agent_id, verified, tx_hash, message_hash,
    vouched_at.

    Raises
    ------
    ValueError
        If the node doesn't exist.
    """
    nodes = mesh._load()
    node = nodes.get(node_id)
    if node is None:
        raise ValueError(f"node not found: {node_id}")

    message = build_attestation_message(node, agent_id)
    signature = sign_attestation(message, sign_fn)

    tx_hash = ""
    if broadcast_fn is not None:
        import json
        tx_hash = broadcast_fn(signature, json.dumps(asdict(message)))

    result = record_onchain_attestation(
        mesh, node_id, signature, tx_hash=tx_hash,
        agent_id=agent_id, aura_score=aura_score,
    )
    result["message_hash"] = message.signing_hash()
    return result


__all__ = [
    "AttestationMessage",
    "build_attestation_message",
    "sign_attestation",
    "record_onchain_attestation",
    "attest_node",
]
