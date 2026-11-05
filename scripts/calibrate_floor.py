"""Calibrate the precision floor used for automatic memory injection.

Automatic injection must be strict: a memory that is merely nearby pollutes every turn's
context. This script measures where the separation actually sits, instead of guessing a
threshold. It produced the table in README.md.

    python3 scripts/calibrate_floor.py
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neural_mesh import Mesh  # noqa: E402

QUERIES = [
    ("relevant:   how often does the deploy key rotate?", "how often does the deploy key rotate?"),
    ("relevant:   what is the dog called?", "what is the dog called?"),
    ("relevant:   when is the hackathon deadline?", "when is the hackathon deadline?"),
    ("unrelated:  zzzz-nonexistent-topic", "zzzz-nonexistent-topic"),
    ("unrelated:  quantum chromodynamics lattice spacing", "quantum chromodynamics lattice spacing"),
]

MEMORIES = [
    "the deploy key rotates every 90 days",
    "the user's dog is called Kumo and he is a husky",
    "the hackathon deadline is November 11 at 5PM EST",
]

ABSOLUTE_FLOORS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
RELATIVE_FLOORS = [0.0, 0.3, 0.5, 0.7, 0.9]


def store():
    mesh = Mesh(os.path.join(tempfile.mkdtemp(prefix="nm-floor-"), "m.db"))
    for memory in MEMORIES:
        mesh.add(memory)
    return mesh


def table(title, floors, call):
    print(title)
    print(f"{'query':48} " + " ".join(f"{f:>5}" for f in floors))
    for label, query in QUERIES:
        counts = []
        for floor in floors:
            try:
                counts.append(len(call(query, floor)))
            except Exception as exc:  # a floor the core rejects is worth seeing
                counts.append(f"E{type(exc).__name__[:4]}")
        print(f"{label:48} " + " ".join(f"{str(c):>5}" for c in counts))
    print()


def main():
    mesh = store()

    table(
        "absolute floor (min_score) — hits remaining per query",
        ABSOLUTE_FLOORS,
        lambda q, f: mesh.recall(q, top_k=5, min_score=f) if f else mesh.recall(q, top_k=5),
    )

    table(
        "relative floor (rel_floor, fraction of the top score)",
        RELATIVE_FLOORS,
        lambda q, f: mesh.recall(q, top_k=5, rel_floor=f) if f else mesh.recall(q, top_k=5),
    )

    print("Reading: unrelated queries return every stored memory with no floor, and nothing")
    print("at any floor >= 0.05. Relevant queries keep their hit up to 0.30. The plugin's")
    print("default of 0.25 sits inside that band; a relative floor cannot trim a single")
    print("candidate, which is why an absolute one is used.")
    print()
    print("stats:", mesh.stats())


if __name__ == "__main__":
    main()
