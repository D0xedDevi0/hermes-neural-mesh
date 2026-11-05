"""Measure what temporal structure buys, against a flat vector memory on the same data.

The claim under test: bi-temporal memory answers questions a flat vector store cannot —
"which value is current" and "which value was current on that date".

The ablation is deliberately fair. Both arms drive the *same plugin surface* — the shipped
`mesh_remember` / `mesh_recall` / `mesh_asof` tools — against the same engine with the same
facts. The only difference is whether the facts carry temporal structure (`valid_from` plus
`supersedes` chaining) or are simply appended, which is what a flat vector memory does. So the
measured difference is attributable to the temporal model and not to the engine underneath it,
and the numbers describe the plugin as shipped rather than the core in isolation.

    python3 scripts/evidence_temporal.py

Writes docs/evidence/temporal-accuracy.json.
"""

import importlib.util
import json
import pathlib
import re
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

FAMILIES = 15
SUBJECT = "service-{}"
ATTRIBUTE = "retry limit"
VALUES = (3, 7, 11)
DATES = ("2025-01-01", "2025-07-01", "2026-01-01")
# mid-history: the answer at this date differs from the answer now
ASOF_DATE = "2025-08-01"

spec = importlib.util.spec_from_file_location("nm_evidence", REPO / "__init__.py")
assert spec is not None and spec.loader is not None
nm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nm)


def facts(family: int):
    for value, date in zip(VALUES, DATES):
        yield f"the {ATTRIBUTE} for {SUBJECT.format(family)} is {value} attempts", date


def question(family: int) -> str:
    return f"what is the {ATTRIBUTE} for {SUBJECT.format(family)}?"


def remember(provider, content: str, valid_from: str = "", supersedes: str = "") -> str:
    args = {"content": content}
    if valid_from:
        args["valid_from"] = valid_from
    if supersedes:
        args["supersedes"] = supersedes
    out = provider.handle_tool_call("mesh_remember", args) or ""
    found = re.search(r"id (\w+)", out)
    return found.group(1) if found else ""


def build(temporal: bool):
    provider = nm.NeuralMeshProvider()
    provider.initialize("evidence", hermes_home=tempfile.mkdtemp(prefix="nm-evidence-"),
                        agent_context="primary")
    for family in range(FAMILIES):
        previous = ""
        for text, date in facts(family):
            if temporal:
                previous = remember(provider, text, valid_from=date, supersedes=previous)
            else:
                remember(provider, text)
    return provider


def score(provider, temporal: bool):
    now_hits = asof_hits = 0
    for family in range(FAMILIES):
        query = question(family)
        latest = f"is {VALUES[-1]} attempts"
        historical = f"is {VALUES[1]} attempts"

        if latest in provider.handle_tool_call("mesh_recall", {"query": query, "top_k": 1}):
            now_hits += 1

        if temporal:
            got = provider.handle_tool_call(
                "mesh_asof", {"query": query, "as_of": ASOF_DATE, "top_k": 1}
            )
        else:
            # A flat store has no point-in-time operation, so its best available answer is
            # ordinary similarity search, which is what this arm is given.
            got = provider.handle_tool_call("mesh_recall", {"query": query, "top_k": 1})
        if historical in got:
            asof_hits += 1
    return now_hits, asof_hits


def main():
    temporal = build(temporal=True)
    flat = build(temporal=False)

    tm_now, tm_asof = score(temporal, temporal=True)
    fl_now, fl_asof = score(flat, temporal=False)

    result = {
        "families": FAMILIES,
        "questions_per_type": FAMILIES,
        "arm_mesh_temporal": {"current_value_correct": tm_now, "point_in_time_correct": tm_asof},
        "arm_flat_vector": {"current_value_correct": fl_now, "point_in_time_correct": fl_asof},
        "as_of_date_used": ASOF_DATE,
        "note": (
            "Both arms run through the shipped tool surface (mesh_remember / mesh_recall / "
            "mesh_asof) on the same facts. The temporal arm stores valid_from and chains "
            "supersedes; the flat arm only appends. Point-in-time questions have no flat "
            "equivalent, so the flat arm is scored on its best available operation."
        ),
    }

    out_dir = REPO / "docs" / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "temporal-accuracy.json").write_text(json.dumps(result, indent=2) + "\n")

    print(f"{'question type':34}{'mesh (temporal)':>17}{'flat vector':>14}")
    print("-" * 65)
    for label, (a, b) in (
        ("which value is current?", (tm_now, fl_now)),
        ("which value was current then?", (tm_asof, fl_asof)),
    ):
        print(f"{label:34}{a:>8}/{FAMILIES:<8}{b:>8}/{FAMILIES}")
    print()
    print("wrote docs/evidence/temporal-accuracy.json")

    temporal.shutdown()
    flat.shutdown()


if __name__ == "__main__":
    main()
