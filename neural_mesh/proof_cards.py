"""Proof-aware recall helpers.

A proof card is a compact, UI/API-friendly evidence envelope attached to a
retrieved memory. It lets downstream agents answer with both the claim and the
public receipt that backs it.

Pure stdlib. Read-only. No network, no signing.
"""
from __future__ import annotations

from typing import Any


PROOF_SOURCE_KINDS = {"intuition_receipt", "helixa_stamp"}


def short_hex(value: str, head: int = 10, tail: int = 8) -> str:
    """Human-friendly shortening for tx hashes / term IDs."""
    value = str(value or "")
    if len(value) <= head + tail + 1:
        return value
    return f"{value[:head]}…{value[-tail:]}"


def proof_card(node) -> dict[str, Any] | None:
    """Build a compact proof card for a proof-backed MemoryNode.

    Returns None for ordinary memories without supported proof metadata.
    """
    meta = dict(getattr(node, "meta", {}) or {})
    source_kind = meta.get("source_kind")

    # Intuition receipt nodes were created by onchain_provenance.py.
    if source_kind == "intuition_receipt":
        role = meta.get("receipt_role", "")
        tx = (meta.get("atom_tx") if role == "atom" else meta.get("triple_tx")) or meta.get("triple_tx") or meta.get("atom_tx") or ""
        claim = meta.get("statement") or meta.get("label") or getattr(node, "content", "")
        term_id = meta.get("term_id", "")
        url = meta.get("explorer_url") or (f"https://explorer.intuition.systems/tx/{tx}" if tx else "")
        return {
            "proof_type": "intuition_receipt",
            "claim": claim,
            "trust": round(float(getattr(node, "trust", 1.0)), 3),
            "by": getattr(node, "by", "") or getattr(node, "provenance", ""),
            "provenance": getattr(node, "provenance", ""),
            "network": meta.get("network", ""),
            "chain_id": meta.get("chain_id", ""),
            "tx": tx,
            "tx_short": short_hex(tx),
            "term_id": term_id,
            "term_id_short": short_hex(term_id),
            "block": meta.get("block", ""),
            "url": url,
        }

    # Helixa stamps are off-chain review/verification metadata; include only public fields.
    stamp = meta.get("helixa_stamp")
    if stamp:
        return {
            "proof_type": "helixa_stamp",
            "claim": getattr(node, "content", ""),
            "trust": round(float(getattr(node, "trust", 1.0)), 3),
            "by": getattr(node, "by", "") or getattr(node, "provenance", ""),
            "provenance": getattr(node, "provenance", ""),
            "agent_id": stamp.get("agent_id", ""),
            "aura_score": stamp.get("aura_score", ""),
            "verified": stamp.get("verified", ""),
            "source": stamp.get("source", ""),
            "tx": stamp.get("tx_hash", ""),
            "tx_short": short_hex(stamp.get("tx_hash", "")),
            "url": stamp.get("url", ""),
        }

    return None


def node_card(node, include_proof: bool = True) -> dict[str, Any]:
    """Serialize a MemoryNode for proof-aware API responses."""
    card = {
        "id": node.id,
        "content": node.content,
        "type": getattr(node.type, "value", str(node.type)),
        "trust": round(float(node.trust), 3),
        "by": node.by,
        "provenance": node.provenance,
        "lane": node.lane,
        "resonance": round(float(node.resonance), 3),
    }
    if include_proof:
        card["proof"] = proof_card(node)
    return card


def recall_with_proofs(mesh, query: str, top_k: int = 5, mode: str = "hybrid", alpha: float = 0.5) -> dict[str, Any]:
    """Run recall and return hits with adjacent proof cards.

    mode: hybrid | dense | lexical | resonance
    """
    if mode == "dense":
        hits = mesh.dense_recall(query, top_k=top_k)
    elif mode == "lexical":
        hits = mesh.lexical_recall(query, top_k=top_k)
    elif mode == "resonance":
        hits = mesh.recall(query, top_k=top_k)
    else:
        hits = mesh.hybrid_recall(query, top_k=top_k, alpha=alpha)
    return {
        "query": query,
        "mode": mode,
        "total": len(hits),
        "results": [node_card(n, include_proof=True) for n in hits],
    }


def citation_for_proof(proof: dict[str, Any], index: int) -> str:
    """Compact human citation for a proof card."""
    if not proof:
        return ""
    if proof.get("proof_type") == "intuition_receipt":
        return f"[{index}] {proof.get('network', '')} block {proof.get('block', '')} tx {proof.get('tx_short', '')}".strip()
    if proof.get("proof_type") == "helixa_stamp":
        agent = proof.get("agent_id", "")
        verified = proof.get("verified", "")
        return f"[{index}] Helixa agent {agent} {verified}".strip()
    return f"[{index}] {proof.get('proof_type', 'proof')}"


def answer_with_proofs(mesh, query: str, top_k: int = 5, mode: str = "hybrid", alpha: float = 0.5, reader=None) -> dict[str, Any]:
    """Answer a query from recalled context and attach supporting proof cards.

    Default reader is extractive: it returns the top retrieved passage. A custom
    reader can be supplied as any object with ``answer(query, passages)``.
    """
    recalled = recall_with_proofs(mesh, query, top_k=top_k, mode=mode, alpha=alpha)
    passages = [r["content"] for r in recalled["results"]]
    if reader is None:
        from .reader import ExtractiveReader
        reader = ExtractiveReader()
    answer = reader.answer(query, passages) if passages else ""
    # Detect reader type for honest method labelling
    try:
        from .reader_llm import LLMReader
        method = "llm_with_proofs" if isinstance(reader, LLMReader) else "extractive_with_proofs"
    except Exception:
        method = "extractive_with_proofs"
    proofs = [r["proof"] for r in recalled["results"] if r.get("proof")]
    citations = [citation_for_proof(p, i + 1) for i, p in enumerate(proofs)]
    return {
        "query": query,
        "answer": answer,
        "method": method,
        "proof_count": len(proofs),
        "proofs": proofs,
        "citations": citations,
        "support": recalled["results"],
    }


__all__ = ["proof_card", "node_card", "recall_with_proofs", "answer_with_proofs", "citation_for_proof", "short_hex"]
