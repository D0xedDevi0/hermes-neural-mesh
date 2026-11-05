"""`.mesh` — a portable, agent-agnostic interchange format for NEURAL_MESH graphs.

Design goals
------------
* **Portable across agents/models.** Embeddings are embedder-specific, so a
  `.mesh` file stores *content + type + topology (links) + metadata*, NOT raw
  vectors. On import the receiving agent re-derives embeddings with its own
  embedder (optionally reusing stored vectors when the `embedder` tag matches).
* **Self-describing.** The file header carries a schema version, an optional
  embedder fingerprint, and node/edge counts, so a loader can reject
  incompatible files loudly instead of silently mis-importing.
* **One JSONL file.** Each line is a compact node record (JSON). Edges live
  inside each node's `links` map, so the file is append-friendly and diff-able
  under git.

Format
------
    {"mesh": {"version": 1, "embedder": "<fingerprint or null>",
              "generated_at": <epoch>, "node_count": N}}
    {"id": "...", "type": "semantic", "content": "...", "links": {"<id>": 0.83},
     "meta": {"created_at": ..., "trust": 1.0, "lane": "hot", "version": 1,
              "superseded_by": "", "provenance": "..."}}
    ...

Pure stdlib. No external deps.
"""
from __future__ import annotations

import json
import time
import uuid

SCHEMA_VERSION = 1


def _fingerprint(embedder) -> str | None:
    """Best-effort stable label for the embedder that produced the vectors."""
    if embedder is None:
        return None
    name = getattr(embedder, "__name__", None)
    if name:
        return name
    mod = getattr(embedder, "__module__", "")
    qual = getattr(embedder, "__qualname__", "")
    return (mod + "." + qual).strip(".") or repr(embedder)


def export_mesh(mesh, path: str) -> dict:
    """Write the live mesh to a `.mesh` file. Returns a summary dict."""
    nodes = mesh._load()
    live = [n for n in nodes.values() if not n.superseded_by
            or n.superseded_by in ("", "__pruned__")]
    # Include only non-archived nodes (skip __pruned__), but keep superseded_by
    # so version history survives the round-trip.
    keep = [n for n in nodes.values() if n.superseded_by != "__pruned__"]
    header = {
        "mesh": {
            "version": SCHEMA_VERSION,
            "embedder": _fingerprint(getattr(mesh, "embedder", None)),
            "generated_at": time.time(),
            "node_count": len(keep),
        }
    }
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(header) + "\n")
        for n in keep:
            rec = {
                "id": n.id,
                "type": n.type.value,
                "content": n.content,
                "links": {k: round(float(v), 4)
                          for k, v in n.links.items()
                          if not k.startswith("__")},
                "meta": {
                    "created_at": n.created_at,
                    "last_accessed": n.last_accessed,
                    "access_count": n.access_count,
                    "resonance": n.resonance,
                    "trust": n.trust,
                    "provenance": n.provenance,
                    "lane": n.lane,
                    "version": n.version,
                    "superseded_by": n.superseded_by,
                    "agent_id": n.agent_id,
                    "conflict_group": n.conflict_group,
                },
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"path": path, "nodes": len(keep), "embedder": header["mesh"]["embedder"]}


def import_mesh(path: str, mesh, *, reembed: bool = True) -> dict:
    """Load a `.mesh` file into `mesh` (a Mesh instance).

    If `reembed` is True (default) each node's embedding is recomputed with the
    target mesh's embedder, guaranteeing vectors are compatible even if the file
    came from a different agent/model. Set `reembed=False` to trust stored
    vectors (faster, but only safe when embedders match).
    """
    loaded = 0
    with open(path, encoding="utf-8") as fh:
        header_line = fh.readline()
        header = json.loads(header_line).get("mesh", {})
        if header.get("version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported .mesh schema v{header.get('version')} "
                f"(this loader speaks v{SCHEMA_VERSION})")
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            from .node import MemoryNode, MemoryType
            emb = mesh.embedder(rec["content"]) if reembed else tuple()
            node = MemoryNode(
                id=rec["id"],
                type=MemoryType(rec["type"]),
                content=rec["content"],
                embedding=emb,
                links={k: float(v) for k, v in rec.get("links", {}).items()},
            )
            m = rec.get("meta", {})
            node.created_at = m.get("created_at", 0.0)
            node.last_accessed = m.get("last_accessed", node.created_at)
            node.access_count = m.get("access_count", 0)
            node.resonance = m.get("resonance", 0.0)
            node.trust = m.get("trust", 1.0)
            node.provenance = m.get("provenance", "")
            node.lane = m.get("lane", "hot")
            node.version = m.get("version", 1)
            node.superseded_by = m.get("superseded_by", "")
            node.agent_id = m.get("agent_id", "")
            node.conflict_group = m.get("conflict_group", "")
            mesh._save(node)
            loaded += 1
    return {"loaded": loaded,
            "source_embedder": header.get("embedder"),
            "target_embedder": _fingerprint(getattr(mesh, "embedder", None))}
