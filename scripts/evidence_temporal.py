"""Measure what temporal structure buys, against a flat vector memory on the same data.

The claim under test: bi-temporal memory answers questions a flat vector store cannot —
"which value is current" and "which value was current on that date".

The ablation is deliberately fair. Both arms use the *same* engine, the *same* embedding
function, and the *same* facts; the only difference is whether the facts carry temporal
structure (`valid_from` plus `supersedes` chaining) or are simply appended, which is what a
flat vector memory does. So the measured difference is attributable to the temporal model and
nothing else.

    python3 scripts/evidence_temporal.py

Writes docs/evidence/temporal-accuracy.json.
"""

import datetime
import json
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from neural_mesh import Mesh  # noqa: E402

FAMILIES = 15
SUBJECT = "service-{}"
ATTRIBUTE = "retry limit"
VALUES = (3, 7, 11)
DATES = ("2025-01-01", "2025-07-01", "2026-01-01")
# mid-history: the answer here differs from the answer now
ASOF_DATE = "2025-08-01"
ASOF_EPOCH = datetime.datetime.fromisoformat("2025-08-01T00:00:00").timestamp()


def facts(family: int):
    for value, date in zip(VALUES, DATES):
        yield f"the {ATTRIBUTE} for {SUBJECT.format(family)} is {value} attempts", value, date


def question(family: int) -> str:
    return f"what is the {ATTRIBUTE} for {SUBJECT.format(family)}?"


def build(temporal: bool):
    """One store per arm. Same facts, same order, same embeddings — different structure."""
    mesh = Mesh(os.path.join(tempfile.mkdtemp(prefix="nm-evidence-"), "mesh.db"))
    for family in range(FAMILIES):
        previous = None
        for text, _value, date in facts(family):
            if temporal:
                kwargs = {"valid_from": datetime.datetime.fromisoformat(
                    f"{date}T00:00:00").timestamp()}
                if previous:
                    kwargs["supersedes"] = previous
                node = mesh.add(text, **kwargs)
                previous = node.id
            else:
                mesh.add(text)
    return mesh


def top_content(nodes) -> str:
    return nodes[0].content if nodes else ""


def score(mesh, temporal: bool):
    now_hits = asof_hits = 0
    for family in range(FAMILIES):
        latest = f"is {VALUES[-1]} attempts"
        historical = f"is {VALUES[1]} attempts"

        nodes = mesh.recall(question(family), top_k=1)
        if latest in top_content(nodes):
            now_hits += 1

        if temporal:
            nodes = mesh.recall_asof(question(family), as_of=ASOF_EPOCH, top_k=1)
        else:
            # a flat store has no point-in-time operation; its best available answer is
            # ordinary similarity search, which is what the arm is given.
            nodes = mesh.recall(question(family), top_k=1)
        if historical in top_content(nodes):
            asof_hits += 1

    return now_hits, asof_hits


def main():
    temporal_mesh = build(temporal=True)
    flat_mesh = build(temporal=False)

    tm_now, tm_asof = score(temporal_mesh, temporal=True)
    fl_now, fl_asof = score(flat_mesh, temporal=False)

    result = {
        "families": FAMILIES,
        "questions_per_type": FAMILIES,
        "arm_mesh_temporal": {"current_value_correct": tm_now, "point_in_time_correct": tm_asof},
        "arm_flat_vector": {"current_value_correct": fl_now, "point_in_time_correct": fl_asof},
        "note": (
            "Both arms: same engine, same embeddings, same facts. The temporal arm stores "
            "valid_from and chains supersedes; the flat arm only appends. Point-in-time "
            "questions have no flat equivalent, so the flat arm is scored on its best "
            "available operation (ordinary similarity search)."
        ),
    }

    out_dir = pathlib.Path("docs/evidence")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "temporal-accuracy.json").write_text(json.dumps(result, indent=2) + "\n")

    print(f"{'question type':34} {'mesh (temporal)':>16} {'flat vector':>12}")
    print("-" * 64)
    for label, (a, b) in (
        ("which value is current?", (tm_now, fl_now)),
        ("which value was current then?", (tm_asof, fl_asof)),
    ):
        print(f"{label:34} {a:>8}/{FAMILIES:<7} {b:>7}/{FAMILIES}")
    print()
    print(f"store sizes: temporal={temporal_mesh.stats()['total']} live nodes, "
          f"flat={flat_mesh.stats()['total']} nodes")
    print("wrote docs/evidence/temporal-accuracy.json")


if __name__ == "__main__":
    main()
