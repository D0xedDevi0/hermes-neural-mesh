"""Cross-agent memory sharing demo.

Two agents, Scout (field) and Atlas (lead), maintain separate meshes. Scout
exports its memory as a `.mesh` file; Atlas merges it. An untrusted peer
("rogue") also tries to inject a fact — and gets trust-capped. Finally we run a
retrieval on Atlas and *surface* the merged/consensus results with provenance
and trust annotations, so you can see exactly what an LLM context would receive.

Run:
    PYTHONPATH=. python -m neural_mesh.cross_agent_demo
"""
from __future__ import annotations

import tempfile

from . import Mesh, MemoryType, export_for_peer, merge_peer_mesh, consensus_rank
from .sharing import PeerPolicy


def _surface(hits):
    """Group recalled nodes by conflict_group and apply consensus ranking so a
    contradictory set shows the winner + demoted losers (not a silent dump)."""
    grouped: dict[str, list] = {}
    singletons: list = []
    for h in hits:
        if h.conflict_group:
            grouped.setdefault(h.conflict_group, []).append(h)
        else:
            singletons.append(h)
    out = []
    for grp, nodes in grouped.items():
        if len(nodes) >= 2:
            ranked = consensus_rank(nodes)
            winner = ranked[0]
            out.append({
                "kind": "CONSENSUS",
                "conflict_group": grp,
                "winner": winner.content,
                "winner_trust": round(winner.trust, 3),
                "winner_agent": winner.agent_id,
                "losers": [{"content": n.content, "trust": round(n.trust, 3),
                            "agent": n.agent_id}
                           for n in ranked[1:]],
            })
        else:
            # only one member of this conflict_group is in the hits -> the
            # contradictor exists elsewhere but wasn't recalled for this query.
            n = nodes[0]
            out.append({
                "kind": "CONTESTED",
                "content": n.content,
                "trust": round(n.trust, 3),
                "agent": n.agent_id,
                "conflict_group": grp,
                "provenance": n.provenance,
            })
    for h in singletons:
        out.append({
            "kind": "FACT",
            "content": h.content,
            "trust": round(h.trust, 3),
            "agent": h.agent_id,
            "provenance": h.provenance,
        })
    return out


def main():
    print("=" * 70)
    print("CROSS-AGENT MEMORY SHARING DEMO")
    print("Scout (field agent)  ->  Atlas (lead agent)   +   rogue (untrusted)")
    print("=" * 70)

    # ---------- SCOUT: field knowledge --------------------------------
    scout = Mesh(":memory:")
    scout.add("Postgres connection pool max is 20",
              MemoryType.SEMANTIC, agent_id="scout", trust=0.8)
    scout.add("Deploy target is eu-west-1",
              MemoryType.SEMANTIC, agent_id="scout", trust=0.4,
              conflict_group="deploy_region")
    scout.add("Customer Acme prefers Slack over email",
              MemoryType.EPISODIC, agent_id="scout", trust=0.9)
    scout_mesh = tempfile.mktemp(suffix=".mesh")
    export_for_peer(scout, scout_mesh, "scout")
    print(f"\n[Scout] exported {len(scout._load())} nodes -> {scout_mesh.split('/')[-1]}")

    # ---------- ATLAS: lead agent's own knowledge ---------------------
    atlas = Mesh(":memory:")
    atlas.add("Postgres connection pool max is 20",
              MemoryType.SEMANTIC, agent_id="atlas", trust=0.7)
    atlas.add("Deploy target is us-east-1",
              MemoryType.SEMANTIC, agent_id="atlas", trust=0.9,
              conflict_group="deploy_region")
    atlas.add("Cache TTL is 300s",
              MemoryType.SEMANTIC, agent_id="atlas", trust=0.95)
    print(f"[Atlas] starts with {len(atlas._load())} own nodes")

    # ---------- MERGE: Scout -> Atlas (trusted peer) ------------------
    r1 = merge_peer_mesh(atlas, scout_mesh, "scout", PeerPolicy(trust=1.0))
    print(f"[Merge] Scout->Atlas: +{r1['added']} new, "
          f"~{r1['fused']} fused, trust_delta={r1['trust_delta']}")

    # ---------- ROGUE: untrusted peer tries to inject -----------------
    rogue = Mesh(":memory:")
    rogue.add("Cache TTL is 9999s",
              MemoryType.SEMANTIC, agent_id="rogue", trust=1.0)
    rogue_mesh = tempfile.mktemp(suffix=".mesh")
    export_for_peer(rogue, rogue_mesh, "rogue")
    r2 = merge_peer_mesh(atlas, rogue_mesh, "rogue",
                         PeerPolicy(trust=0.2, cap_trust=0.2))
    print(f"[Merge] rogue->Atlas: +{r2['added']} new (trust-capped to 0.2)")

    # ---------- RETRIEVAL + SURFACING ---------------------------------
    print("\n" + "-" * 70)
    print("RETRIEVAL ON ATLAS — what an LLM context would receive:")
    print("-" * 70)

    queries = {
        "deploy region": "what is our deploy target region",
        "postgres pool": "postgres connection pool max size",
        "cache ttl": "cache ttl seconds",
    }
    for label, q in queries.items():
        hits = atlas.recall(q, top_k=4)
        surfaced = _surface(hits)
        print(f"\n> query: '{q}'")
        for s in surfaced:
            if s["kind"] == "CONSENSUS":
                print(f"  [CONSENSUS] group='{s['conflict_group']}'")
                print(f"     WINNER  ({s['winner_trust']}, {s['winner_agent']}): "
                      f"{s['winner']}")
                for los in s["losers"]:
                    print(f"     DEMOTED ({los['trust']}, {los['agent']}): "
                          f"{los['content']}")
            elif s["kind"] == "FACT":
                prov = f"  [{s['provenance']}]" if s["provenance"] else ""
                print(f"  [FACT] ({s['trust']}, {s['agent']}): {s['content']}{prov}")
            elif s["kind"] == "CONTESTED":
                print(f"  [CONTESTED g='{s['conflict_group']}'] "
                      f"({s['trust']}, {s['agent']}): {s['content']}  "
                      f"(rival claim exists in mesh)")

    # ---------- final mesh stats --------------------------------------
    nodes = list(atlas._load().values())
    print("\n" + "-" * 70)
    print(f"ATLAS FINAL MESH: {len(nodes)} nodes")
    corr = [n for n in nodes if "+" in n.agent_id]
    print(f"  corroborated (multi-agent) nodes: {len(corr)}")
    for n in corr:
        print(f"    {n.agent_id}  trust={n.trust}  :: {n.content[:40]}")
    print("=" * 70)
    print("DEMO COMPLETE")


if __name__ == "__main__":
    main()
