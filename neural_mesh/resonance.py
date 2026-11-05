"""Resonance retrieval — the NEURAL_MESH differentiator.

Instead of flat cosine top-k, a query seeds the most similar nodes, then
activation spreads across the mesh topology to linked neighbours with decay
(HippoRAG-style hippocampal indexing). Final ranking fuses resonance x recency
x trust. This surfaces *related-but-not-obviously-similar* memories that flat
vector search misses.
"""
from __future__ import annotations

import time

from .embed import cosine


def rust_available() -> bool:
    """Return whether the optional Rust extension has the v0.21 hot-path API."""
    try:
        import rust_mesh
        return (
            callable(getattr(rust_mesh, "query_dot_similarity", None))
            and callable(getattr(rust_mesh, "spread_activation", None))
        )
    except ImportError:
        return False


def _select_backend(requested: str) -> str:
    if requested not in {"auto", "python", "rust"}:
        raise ValueError("backend must be 'auto', 'python', or 'rust'")
    if requested == "rust" and not rust_available():
        raise RuntimeError("Rust resonance backend is not installed")
    if requested == "auto":
        return "rust" if rust_available() else "python"
    return requested


def _spread_python(mesh_nodes: dict, resonance: dict[str, float], frontier,
                   spread_steps: int, decay: float) -> dict[str, float]:
    for _ in range(spread_steps):
        nxt = []
        for n in frontier:
            for nbr_id, w in n.links.items():
                nbr = mesh_nodes.get(nbr_id)
                if not nbr or nbr.superseded_by:
                    continue
                gain = resonance[n.id] * decay * w
                if gain > resonance.get(nbr_id, 0.0):
                    resonance[nbr_id] = gain
                    nxt.append(nbr)
        frontier = nxt
    return resonance


def _spread_rust(mesh_nodes: dict, resonance: dict[str, float], frontier,
                 spread_steps: int, decay: float) -> dict[str, float]:
    import rust_mesh

    node_ids = list(mesh_nodes)
    index = {node_id: i for i, node_id in enumerate(node_ids)}
    edges = []
    for source_id, node in mesh_nodes.items():
        source = index[source_id]
        for target_id, weight in node.links.items():
            target = index.get(target_id)
            if target is not None and not mesh_nodes[target_id].superseded_by:
                edges.append((source, target, float(weight)))
    spread = rust_mesh.spread_activation(
        [float(resonance.get(node_id, 0.0)) for node_id in node_ids],
        edges,
        [index[node.id] for node in frontier],
        spread_steps,
        decay,
    )
    return {node_id: float(spread[i]) for i, node_id in enumerate(node_ids)}


def retrieve(mesh_nodes: dict, query_embedding, top_k: int = 5,
             seed_k: int = 6, spread_steps: int = 2, decay: float = 0.5,
             backend: str = "auto"):
    selected = _select_backend(backend)
    # 1. Seed: nodes most similar to the query.
    live = [n for n in mesh_nodes.values() if not n.superseded_by]
    seeds = []
    if selected == "rust" and live:
        import rust_mesh

        dim = len(query_embedding)
        if dim and all(len(n.embedding) == dim for n in live):
            flat_nodes = [value for n in live for value in n.embedding]
            similarities = rust_mesh.query_dot_similarity(
                list(query_embedding), flat_nodes, dim)
            seeds = list(zip(live, similarities))
        else:
            # Preserve custom-embedder compatibility: Python's zip-based dot
            # product tolerates uneven vectors, while the SIMD-shaped Rust API
            # intentionally requires a rectangular matrix.
            seeds = [(n, cosine(query_embedding, n.embedding)) for n in live]
    else:
        seeds = [(n, cosine(query_embedding, n.embedding)) for n in live]
    seeds.sort(key=lambda x: -x[1])

    # 2. Resonance field: seed similarity is the initial activation.
    resonance = {n.id: max(0.0, sim) for n, sim in seeds}

    # 3. Spread: activation flows to linked neighbours, decaying each hop.
    # The optional Rust path implements this exact max-propagation contract;
    # `auto` selects it when installed and otherwise stays pure stdlib.
    frontier = [n for n, _ in seeds[:max(3, seed_k)]]
    # Building a Rust edge matrix costs more than a two-hop walk over six
    # seeds. Reserve it for genuinely large/deep traversals; query scoring is
    # still Rust-accelerated for ordinary recalls.
    if selected == "rust" and len(mesh_nodes) >= 5000 and spread_steps >= 4:
        resonance = _spread_rust(mesh_nodes, resonance, frontier, spread_steps, decay)
    else:
        resonance = _spread_python(mesh_nodes, resonance, frontier, spread_steps, decay)

    # 4. Rank: resonance x recency x trust.
    now = time.time()
    scored = []
    for nid, r in resonance.items():
        n = mesh_nodes.get(nid)
        if not n:
            continue
        age_days = max(0.0, (now - n.last_accessed) / 86400.0)
        recency = 1.0 / (1.0 + age_days)          # decays over ~a day
        score = r * (0.5 + 0.5 * recency) * n.trust
        scored.append((score, n))
    scored.sort(key=lambda x: -x[0])
    return [n for _, n in scored[:top_k]]
