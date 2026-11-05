"""On-chain provenance ingestion for NEURAL_MESH.

Turns public deployment receipts (Intuition Mainnet txs, atom/triple term IDs,
creator address, block numbers) into high-trust semantic memories. This makes
external proofs recallable by the mesh instead of living only in docs/social posts.

Pure stdlib. No signing, no RPC writes, no secrets.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .core import Mesh, MemoryType


@dataclass
class IntuitionTripleReceipt:
    """One Intuition triple emitted by a createTriples transaction."""

    statement: str
    term_id: str

    @property
    def parts(self) -> tuple[str, str, str]:
        """Return (subject, predicate, object) if the statement uses arrows."""
        bits = [p.strip() for p in self.statement.split("→")]
        if len(bits) == 3:
            return bits[0], bits[1], bits[2]
        bits = [p.strip() for p in self.statement.split("->")]
        if len(bits) == 3:
            return bits[0], bits[1], bits[2]
        return self.statement, "", ""


@dataclass
class IntuitionDeploymentReceipt:
    """Parsed public receipt bundle for an Intuition deployment."""

    network: str = "Intuition Mainnet"
    chain_id: str = "1155"
    signer: str = ""
    multivault: str = ""
    triple_tx: str = ""
    block: str = ""
    status: str = ""
    batch_value: str = ""
    before_balance: str = ""
    after_balance: str = ""
    entity_atoms: dict[str, dict[str, str]] = field(default_factory=dict)
    predicate_atoms: dict[str, dict[str, str]] = field(default_factory=dict)
    triples: list[IntuitionTripleReceipt] = field(default_factory=list)

    @property
    def explorer_url(self) -> str:
        return f"https://explorer.intuition.systems/tx/{self.triple_tx}" if self.triple_tx else ""

    def digest(self) -> str:
        return (
            f"Intuition deployment verified on {self.network} chain {self.chain_id}: "
            f"{len(self.entity_atoms)} entity atoms, {len(self.predicate_atoms)} predicate atoms, "
            f"{len(self.triples)} triples in tx {self.triple_tx} at block {self.block}."
        )


def _field(markdown: str, label: str) -> str:
    m = re.search(rf"^{re.escape(label)}:\s*`?([^`\n]+)`?", markdown, re.MULTILINE)
    return m.group(1).strip() if m else ""


def _parse_atom_table(markdown: str, heading: str) -> dict[str, dict[str, str]]:
    start = markdown.find(heading)
    if start < 0:
        return {}
    next_heading = markdown.find("\n## ", start + len(heading))
    block = markdown[start: next_heading if next_heading > 0 else len(markdown)]
    atoms: dict[str, dict[str, str]] = {}
    for line in block.splitlines():
        # | Label | `tx` | `term_id` |
        m = re.match(r"\|\s*([^|`][^|]*?)\s*\|\s*`(0x[a-fA-F0-9]+)`\s*\|\s*`(0x[a-fA-F0-9]{64})`\s*\|", line)
        if not m:
            continue
        label, tx, term_id = [x.strip() for x in m.groups()]
        if label.lower() in {"label", "---"}:
            continue
        atoms[label] = {"tx": tx, "term_id": term_id}
    return atoms


def _parse_triple_table(markdown: str) -> list[IntuitionTripleReceipt]:
    start = markdown.find("## Triple Batch")
    if start < 0:
        return []
    next_heading = markdown.find("\n## Balance", start)
    block = markdown[start: next_heading if next_heading > 0 else len(markdown)]
    triples: list[IntuitionTripleReceipt] = []
    for line in block.splitlines():
        m = re.match(r"\|\s*([^|`][^|]*?→[^|]*?)\s*\|\s*`(0x[a-fA-F0-9]{64})`\s*\|", line)
        if not m:
            continue
        triples.append(IntuitionTripleReceipt(statement=m.group(1).strip(), term_id=m.group(2).strip()))
    return triples


def parse_intuition_receipts(markdown: str) -> IntuitionDeploymentReceipt:
    """Parse the Markdown receipt file produced after Intuition deployment."""
    receipt = IntuitionDeploymentReceipt(
        network=_field(markdown, "Network") or "Intuition Mainnet",
        chain_id=_field(markdown, "Chain ID") or "1155",
        signer=_field(markdown, "Signer / creator"),
        multivault=_field(markdown, "MultiVault"),
        triple_tx=_field(markdown, "Tx"),
        block=_field(markdown, "Block"),
        status=_field(markdown, "Status"),
        batch_value=_field(markdown, "Batch value"),
        before_balance=_field(markdown, "Before triple batch"),
        after_balance=_field(markdown, "After triple batch"),
        entity_atoms=_parse_atom_table(markdown, "## Entity Atoms"),
        predicate_atoms=_parse_atom_table(markdown, "## Predicate Atoms"),
        triples=_parse_triple_table(markdown),
    )
    return receipt


def load_intuition_receipts(path: str | Path) -> IntuitionDeploymentReceipt:
    """Read and parse an Intuition deployment receipt markdown file."""
    return parse_intuition_receipts(Path(path).read_text(encoding="utf-8"))


def receipt_memory_payloads(receipt: IntuitionDeploymentReceipt) -> list[dict]:
    """Build mesh.add kwargs for receipt summary + atom/triple proof nodes."""
    base_meta = {
        "network": receipt.network,
        "chain_id": receipt.chain_id,
        "creator": receipt.signer,
        "multivault": receipt.multivault,
        "triple_tx": receipt.triple_tx,
        "block": receipt.block,
        "status": receipt.status,
        "explorer_url": receipt.explorer_url,
        "source_kind": "intuition_receipt",
    }
    payloads: list[dict] = [
        {
            "content": receipt.digest(),
            "type": MemoryType.SEMANTIC,
            "provenance": "intuition-mainnet",
            "by": "intuition-mainnet",
            "trust": 0.98,
            "conflict_group": f"intuition-deploy:{receipt.triple_tx}",
            "meta": {**base_meta, "receipt_role": "deployment_digest"},
        }
    ]

    for label, atom in {**receipt.entity_atoms, **receipt.predicate_atoms}.items():
        payloads.append({
            "content": f"Intuition atom '{label}' is deployed with term ID {atom['term_id']} in tx {atom['tx']}.",
            "type": MemoryType.SEMANTIC,
            "provenance": "intuition-mainnet",
            "by": "intuition-mainnet",
            "trust": 0.97,
            "conflict_group": f"intuition-atom:{atom['term_id']}",
            "meta": {**base_meta, "receipt_role": "atom", "label": label, "term_id": atom["term_id"], "atom_tx": atom["tx"]},
        })

    for triple in receipt.triples:
        subj, pred, obj = triple.parts
        payloads.append({
            "content": f"Intuition triple verified: {triple.statement} (term ID {triple.term_id}) in tx {receipt.triple_tx}.",
            "type": MemoryType.SEMANTIC,
            "provenance": "intuition-mainnet",
            "by": "intuition-mainnet",
            "trust": 0.99,
            "conflict_group": f"intuition-triple:{triple.term_id}",
            "meta": {
                **base_meta,
                "receipt_role": "triple",
                "term_id": triple.term_id,
                "statement": triple.statement,
                "subject": subj,
                "predicate": pred,
                "object": obj,
            },
        })
    return payloads


def _existing_conflict_groups(mesh: Mesh) -> set[str]:
    groups = set()
    for node in mesh._load().values():
        if node.conflict_group:
            groups.add(node.conflict_group)
    return groups


def ingest_intuition_receipts(mesh: Mesh, path: str | Path) -> dict:
    """Ingest receipt markdown into a Mesh idempotently.

    Uses conflict_group keys derived from tx/term IDs, so repeated ingestion skips
    already-known proofs instead of duplicating them.
    """
    receipt = load_intuition_receipts(path)
    existing = _existing_conflict_groups(mesh)
    added = []
    skipped = []
    for payload in receipt_memory_payloads(receipt):
        group = payload.get("conflict_group", "")
        if group and group in existing:
            skipped.append(group)
            continue
        node = mesh.add(**payload)
        added.append(node.id)
        if group:
            existing.add(group)
    return {
        "ok": True,
        "added": len(added),
        "skipped": len(skipped),
        "node_ids": added,
        "digest": receipt.digest(),
        "triple_tx": receipt.triple_tx,
        "triple_count": len(receipt.triples),
        "atom_count": len(receipt.entity_atoms) + len(receipt.predicate_atoms),
        "explorer_url": receipt.explorer_url,
    }


__all__ = [
    "IntuitionTripleReceipt",
    "IntuitionDeploymentReceipt",
    "parse_intuition_receipts",
    "load_intuition_receipts",
    "receipt_memory_payloads",
    "ingest_intuition_receipts",
]
