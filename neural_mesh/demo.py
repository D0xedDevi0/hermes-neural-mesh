"""Live demo — run `python -m neural_mesh.demo`.

Shows: 5-type writes, auto-linking, resonance retrieval (finds related-but-not
obvious memories), the pointer protocol (big output -> context stays tiny),
the sleep cycle (prune + reflect), and live stats.
"""
from __future__ import annotations

import os
import tempfile

from .core import Mesh, MemoryType
from .pointer import store_big_output


def main():
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "mesh.db")
    m = Mesh(db_path=db)

    print("=" * 70)
    print("NEURAL_MESH demo — self-organizing agentic memory")
    print("=" * 70)

    # 1. Five memory types
    m.add("User Cody is based in Kuala Lumpur, timezone Asia/KL.", MemoryType.SEMANTIC, provenance="chat")
    m.add("The auth service uses service accounts, not per-user keys.", MemoryType.SEMANTIC, provenance="codebase")
    m.add("Deploy failed at 02:14 because Vercel blocked unknown git author.", MemoryType.EPISODIC, provenance="log")
    m.add("Refactor the memory module before Thursday's launch.", MemoryType.PROSPECTIVE, provenance="user")
    m.add("To deploy: run hermes-scheduled-jobs validate, then gh workflow.", MemoryType.PROCEDURAL, provenance="skill")
    m.add("Screenshot of the dashboard shows 20k star target.", MemoryType.SENSORY, provenance="image-summary")
    print("\n[+] Wrote 6 nodes across 6 memory types. Auto-linking on ingest...")

    # 2. Resonance retrieval — should surface related nodes even if not literal match
    print("\n-- recall('how do I ship the deploy?') --")
    for n in m.recall("how do I ship the deploy?"):
        print(f"   {n.type.value:11s} | {n.content[:60]}")

    # 3. Pointer protocol — big tool output stays OUT of context
    big = "X" * 200_000  # simulate a 200KB log dump
    ptr = store_big_output(big, "logs-payment", tmp)
    print(f"\n[+] 200KB tool output stored. Context only received: {ptr!r} "
          f"({len(ptr)} bytes vs 200000).")

    # 4. Sleep cycle — prune weak, reflect into insights
    def reflect(nodes):
        return ["Reflection: deployment requires a known git author to pass Vercel gate."]
    res = m.sleep(reflect_fn=reflect, max_age_days=9999)
    print(f"\n[+] SLEEP cycle: pruned={res['pruned']}, insights_generated={res['insights']}")

    # 5. Stats
    s = m.stats()
    print(f"\n[+] Mesh stats: {s}")
    print("=" * 70)
    print("Done. Pure-stdlib core. Swap embedder for real vectors to scale.")
    print("=" * 70)


if __name__ == "__main__":
    main()
