"""YantrikDB bridge for NEURAL_MESH.

Connects our existing mesh.db (SQLite + graph + proof cards) to yantrikdb's
embedded engine for contradiction tracking, explainable recall, self-tuning,
skill extraction, and the self-directing task loop.

Architecture:
  mesh.db (NEURAL_MESH source of truth)
    ↕ sync
  YantrikDB (contradiction/KG/skill layer)
    ↕
  Hermes memory provider (optional — for Hermes session hooks)

The mesh stays canonical. YantrikDB enriches it.

Usage:
  from neural_mesh.integrations.yantrikdb_bridge import YantrikDBBridge
  br = YantrikDBBridge(mesh, db_path="/opt/data/yantrikdb/memory.db")
  br.ingest_mesh()               # copy existing mesh nodes → yantrikdb
  br.contradictions()            # first-class conflict detection
  br.gaps()                      # auto-discovered knowledge gaps
  br.enhanced_recall("ship deploy", top_k=7)  # mesh + yantrikdb merged
"""
from __future__ import annotations

import json
import time
from typing import Any

try:
    from yantrikdb_hermes_plugin.embedded import (
        EmbeddedYantrikDBClient, YantrikDBConfig, make_backend,
    )
    _YANTRIKDB_OK = True
except ImportError:  # pragma: no cover - graceful when not installed
    _YANTRIKDB_OK = False
    EmbeddedYantrikDBClient = object  # type: ignore[misc, assignment]


class YantrikDBBridge:
    """Dual-write / dual-read bridge between NEURAL_MESH and YantrikDB."""

    def __init__(self, mesh, *, db_path: str = "/opt/data/yantrikdb/memory.db",
                 namespace: str = "d0xeddev", top_k: int = 10):
        self.mesh = mesh
        self.top_k = top_k
        self.namespace = namespace
        if not _YANTRIKDB_OK:
            self.client = None
            return
        cfg = YantrikDBConfig(
            mode="embedded",
            db_path=db_path,
            namespace=namespace,
            top_k=top_k,
            owner_scoping=True,
            include_base_namespace_recall=True,
            include_legacy_actor_namespace_recall=False,
        )
        self.client = make_backend(cfg)

    @property
    def available(self) -> bool:
        return self.client is not None

    # ---- Bidirectional sync ------------------------------------------------

    def ingest_mesh(self, limit: int = 1000) -> dict[str, Any]:
        """Copy existing mesh nodes → yantrikDB store.

        Maps MemoryType → memory_type, preserves provenance/trust/meta,
        uses node.id as idempotency_key for upsert-safety.
        """
        if not self.available:
            return {"ok": False, "error": "yantrikdb not available"}
        cur = self.mesh.db.execute(
            "SELECT id, type, content, meta FROM nodes ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = cur.fetchall()
        written, skipped = 0, 0
        for r in rows:
            meta = json.loads(r["meta"]) if r["meta"] else {}
            if meta.get("superseded_by"):
                skipped += 1
                continue
            try:
                self.client.remember(
                    text=r["content"],
                    namespace=self.namespace,
                    importance=float(meta.get("trust", 0.6)),
                    memory_type=r["type"],
                    metadata={
                        "mesh_id": r["id"],
                        "provenance": meta.get("provenance", "unknown"),
                        "by": meta.get("by", "unknown"),
                    },
                    idempotency_key=f"mesh:{r['id']}",
                )
                written += 1
            except Exception as exc:
                skipped += 1
        return {
            "ok": True, "written": written, "skipped": skipped,
            "scanned": len(rows), "namespace": self.namespace,
        }

    # ---- YantrikDB enrichment queries ---------------------------------------

    def contradictions(self) -> dict[str, Any]:
        """Detect first-class contradictions between mesh memories."""
        if not self.available:
            return {"ok": False, "error": "yantrikdb not available", "conflicts": []}
        try:
            return self.client.conflicts(namespace=self.namespace)
        except Exception as exc:
            return {"ok": False, "error": str(exc), "conflicts": []}

    def gaps(self, *, limit: int = 20) -> dict[str, Any]:
        """Auto-discover knowledge gaps from repeated-but-unanswered queries."""
        if not self.available:
            return {"ok": False, "error": "yantrikdb not available", "gaps": []}
        try:
            return self.client.knowledge_gaps(limit=limit, namespace=self.namespace)
        except Exception as exc:
            return {"ok": False, "error": str(exc), "gaps": []}

    def think(self, *, consolidate: bool = True, scan_conflicts: bool = True) -> dict[str, Any]:
        """Run a self-direction pass: consolidate, scan conflicts, mine patterns."""
        if not self.available:
            return {"ok": False, "error": "yantrikdb not available"}
        try:
            return self.client.think(
                run_consolidation=consolidate,
                run_conflict_scan=scan_conflicts,
                namespace=self.namespace,
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def stats(self) -> dict[str, Any]:
        if not self.available:
            return {"ok": False, "error": "yantrikdb not available"}
        try:
            return self.client.stats(namespace=self.namespace)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def record_turn(self, role: str, content: str) -> dict[str, Any]:
        if not self.available:
            return {"ok": False, "error": "yantrikdb not available"}
        try:
            return self.client.record_turn(
                role=role, content=content, namespace=self.namespace,
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # ---- Enhanced recall ----------------------------------------------------

    def recall(self, query: str, *, top_k: int | None = None) -> dict[str, Any]:
        """YantrikDB explainable recall (per-hit scoring reasons)."""
        if not self.available:
            return {"ok": False, "error": "yantrikdb not available", "results": []}
        try:
            return self.client.recall(
                query, namespace=self.namespace, top_k=top_k or self.top_k,
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc), "results": []}

    def enhanced_recall(self, query: str, *, top_k: int = 5,
                        mode: str = "hybrid", alpha: float = 0.9) -> dict[str, Any]:
        """Merge mesh hybrid recall with yantrikDB explainable recall.

        Returns a joint top-k ranked by combined score, tagged with source.
        Proofs/citations from mesh pass through; yantrikDB adds `why_retrieved`.
        """
        mesh_results = []
        try:
            if mode == "dense":
                nodes = self.mesh.dense_recall(query, top_k=top_k)
            elif mode == "lexical":
                nodes = self.mesh.lexical_recall(query, top_k=top_k)
            elif mode == "hybrid":
                nodes = self.mesh.hybrid_recall(query, top_k=top_k, alpha=alpha)
            else:
                nodes = self.mesh.recall(query, limit=top_k)
            mesh_results = [
                {
                    "source": "mesh",
                    "id": n.id,
                    "content": n.content,
                    "type": n.type.value,
                    "trust": n.trust,
                    "meta": n.meta,
                }
                for n in nodes
            ]
        except Exception:
            pass

        yan_results = []
        if self.available:
            try:
                r = self.client.recall(query, namespace=self.namespace, top_k=top_k)
                for item in (r.get("results") or []):
                    yan_results.append({
                        "source": "yantrikdb",
                        "id": item.get("rid", ""),
                        "content": item.get("text", ""),
                        "type": item.get("memory_type", "semantic"),
                        "trust": float(item.get("importance", 0.5)),
                        "why_retrieved": item.get("why_retrieved", []),
                        "score": float(item.get("score", 0.0)),
                    })
            except Exception:
                pass

        # Interleave: alternate mesh/yantrik up to top_k total
        merged: list[dict[str, Any]] = []
        m_iter, y_iter = iter(mesh_results), iter(yan_results)
        while len(merged) < top_k:
            m = next(m_iter, None)
            y = next(y_iter, None)
            if m:
                merged.append(m)
            if y and len(merged) < top_k:
                merged.append(y)
            if not m and not y:
                break

        return {
            "ok": True,
            "query": query,
            "mode": mode,
            "count": len(merged),
            "results": merged,
            "mesh_hits": len(mesh_results),
            "yantrikdb_hits": len(yan_results),
        }

    # ---- Skill extraction ---------------------------------------------------

    def define_skill(self, skill_id: str, body: str, *,
                     skill_type: str = "procedure",
                     applies_to: list[str] | None = None) -> dict[str, Any]:
        if not self.available:
            return {"ok": False, "error": "yantrikdb not available"}
        try:
            return self.client.skill_define(
                skill_id=skill_id, body=body,
                skill_type=skill_type,
                applies_to=applies_to or ["mesh", "d0xeddev"],
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def search_skills(self, query: str, *, top_k: int = 5) -> dict[str, Any]:
        if not self.available:
            return {"ok": False, "error": "yantrikdb not available", "results": []}
        try:
            return self.client.skill_search(query, top_k=top_k)
        except Exception as exc:
            return {"ok": False, "error": str(exc), "results": []}
