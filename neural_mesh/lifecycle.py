"""Integrated memory lifecycle: ingest, retrieve, consolidate, and sleep.

This module connects NEURAL_MESH's existing primitives without hiding their
individual reports. Large payloads are externalized before a compact,
searchable pointer node is written to the mesh.
"""
from __future__ import annotations

from .core import MemoryType, Mesh
from .pointer import PointerStore


class MemoryLifecycle:
    """Coordinate pointer-safe ingestion and maintenance for one mesh."""

    def __init__(self, mesh: Mesh, pointer_root: str = ".mesh_pointers",
                 pointer_threshold: int = 8_192):
        if pointer_threshold < 1:
            raise ValueError("pointer_threshold must be positive")
        self.mesh = mesh
        self.pointers = PointerStore(pointer_root)
        self.pointer_threshold = pointer_threshold

    def ingest(self, payload: str, *, label: str = "data",
               type: MemoryType = MemoryType.SEMANTIC,
               provenance: str = "", lane: str = "hot", trust: float = 1.0,
               summary: str = "", meta: dict | None = None) -> dict:
        """Ingest text, externalizing it when it exceeds the context budget."""
        if not isinstance(payload, str):
            raise TypeError("payload must be a string")

        node_meta = dict(meta or {})
        externalized = len(payload) > self.pointer_threshold
        pointer = ""
        content = payload
        if externalized:
            pointer = self.pointers.put(payload, label)
            preview = summary.strip() or self.pointers.summarize(
                pointer, max_chars=min(400, self.pointer_threshold)
            )
            content = f"[{label}] {preview}\nPointer: {pointer}"
            node_meta.update({
                "pointer": pointer,
                "payload_chars": len(payload),
                "externalized": True,
            })

        node = self.mesh.add(
            content,
            type=type,
            lane=lane,
            provenance=provenance,
            trust=trust,
            meta=node_meta,
        )
        return {
            "node_id": node.id,
            "externalized": externalized,
            "pointer": pointer,
            "payload_chars": len(payload),
        }

    def retrieve(self, query: str, *, mode: str = "fact", top_k: int = 5,
                 alpha: float = 0.9, writeback: bool = True,
                 lane: "str | None" = None) -> dict:
        """Route direct fact lookup separately from associative exploration.

        ``fact`` uses dense-heavy hybrid retrieval, while ``associative`` uses
        topology-aware resonance. Explicit primitive mode names are accepted too.
        """
        routes = {
            "fact": "hybrid",
            "associative": "resonance",
            "hybrid": "hybrid",
            "dense": "dense",
            "lexical": "lexical",
            "resonance": "resonance",
        }
        selected = routes.get(mode)
        if selected is None:
            raise ValueError(f"unknown retrieval mode: {mode}")
        if selected == "hybrid":
            hits = self.mesh.hybrid_recall(query, top_k=top_k, alpha=alpha,
                                           writeback=writeback, lane=lane)
        elif selected == "dense":
            hits = self.mesh.dense_recall(query, top_k=top_k, writeback=writeback,
                                          lane=lane)
        elif selected == "lexical":
            hits = self.mesh.lexical_recall(query, top_k=top_k, writeback=writeback,
                                            lane=lane)
        else:
            hits = self.mesh.recall(query, top_k=top_k, writeback=writeback, lane=lane)
        return {"mode": selected, "hits": hits}

    def maintain(self, *, hot_ttl: float = 86_400.0,
                 cold_threshold: int = 3, prune_below: float = 0.05,
                 max_age_days: float = 30.0, reflect_fn=None,
                 mode: str = "sleep", muse_fn=None) -> dict:
        """Consolidate lanes, then run lightweight SLEEP or enriched DREAM."""
        if mode not in ("sleep", "dream"):
            raise ValueError("maintenance mode must be 'sleep' or 'dream'")
        before = {n.id: n.lane for n in self.mesh._load().values()
                  if not n.superseded_by}
        self.mesh.consolidate(hot_ttl=hot_ttl, cold_threshold=cold_threshold)
        after = {n.id: n.lane for n in self.mesh._load().values()
                 if not n.superseded_by}
        promoted = sum(1 for nid, lane in before.items()
                       if lane == "hot" and after.get(nid) == "cold")
        if mode == "dream":
            from .dream import dream
            cycle_report = dream(
                self.mesh,
                prune_below=prune_below,
                max_age_days=max_age_days,
                muse_fn=muse_fn,
            )
        else:
            cycle_report = self.mesh.sleep(
                prune_below=prune_below,
                max_age_days=max_age_days,
                reflect_fn=reflect_fn,
            )
        return {
            "lanes": {"promoted": promoted},
            mode: cycle_report,
            "stats": self.mesh.stats(),
        }

    def cycle(self, payload: str, *, query: str, mode: str = "fact",
              top_k: int = 5, alpha: float = 0.9,
              hot_ttl: float = 86_400.0, cold_threshold: int = 3,
              prune_below: float = 0.05, max_age_days: float = 30.0,
              reflect_fn=None, retrieval_lane: "str | None" = None,
              maintenance_mode: str = "sleep", muse_fn=None,
              **ingest_options) -> dict:
        """Execute the complete pointer → recall → lanes → sleep lifecycle."""
        ingest_report = self.ingest(payload, **ingest_options)
        retrieval = self.retrieve(query, mode=mode, top_k=top_k, alpha=alpha,
                                  lane=retrieval_lane)
        maintenance = self.maintain(
            hot_ttl=hot_ttl,
            cold_threshold=cold_threshold,
            prune_below=prune_below,
            max_age_days=max_age_days,
            reflect_fn=reflect_fn,
            mode=maintenance_mode,
            muse_fn=muse_fn,
        )
        return {
            "ingest": ingest_report,
            "retrieval": retrieval,
            "maintenance": maintenance,
        }
