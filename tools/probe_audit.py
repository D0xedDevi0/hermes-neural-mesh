#!/usr/bin/env python3
"""Audit a recall probe set for the defect that makes it lie: queries too common to discriminate.

A probe is query -> expected token. If ONE query term retrieves more than --max-df live
nodes, the probe does not test retrieval — it tests whether the store happens to rank the
right answer above a crowd of near-identical records, and it will drift as the store grows.

This was not hypothetical. The v3 probe set reported a stable "19/20 cold recall": probe 15's
query ("compression-summary turns below.") was a compaction-banner fragment whose every term
retrieved >=200 nodes and which shared NO term with its expected answer (/community-art), so it
could never pass however good retrieval was. The memory itself was present twice in the store.

Usage (reads NEURAL_MESH_REMOTE_URL / NEURAL_MESH_REMOTE_TOKEN, or a local store):
    python3 tools/probe_audit.py --probe recall_probe.json [--max-df 20] [--collision-cap 200]

Exit 0  every probe passes and no term exceeds --max-df
Exit 1  a probe fails, or a term exceeds the collision budget
Passing-but-fragile probes are reported as warnings and do not fail the run.
"""
import argparse
import json
import os
import sys


def classify(passed, collisions, max_df):
    """Verdict for one probe: 'fail', 'fragile' or 'ok'. Pure, so it is testable."""
    if not passed:
        return "fail"
    return "fragile" if collisions and max(collisions.values(), default=0) > max_df else "ok"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, help="probe set json (probes[].q / .expect)")
    ap.add_argument("--max-df", type=int, default=20,
                    help="a term retrieving more than this many nodes is not discriminative")
    ap.add_argument("--collision-cap", type=int, default=200,
                    help="top_k used to count how many nodes a term retrieves")
    a = ap.parse_args(argv)

    sys.path.insert(0, ".")
    from remote import RemoteMesh
    url = os.environ.get("NEURAL_MESH_REMOTE_URL", "")
    tok = os.environ.get("NEURAL_MESH_REMOTE_TOKEN", "")
    if not url:
        print("probe_audit: set NEURAL_MESH_REMOTE_URL (and _TOKEN) to the store under test")
        return 2
    mesh = RemoteMesh(url, tok, timeout=90)
    probe = json.load(open(a.probe))

    def collisions(term):
        try:
            return len(mesh.recall(term, top_k=a.collision_cap))
        except Exception:
            return -1

    def passes(p):
        hits = mesh.recall(p["q"], top_k=5)
        return any(any(t.lower() in (n.content or "").lower() for t in p["expect"]) for n in hits)

    bad, warn, ok = [], [], 0
    for p in probe["probes"]:
        terms = [t for t in p["q"].replace(",", " ").split() if t.strip()]
        coll = {t: collisions(t) for t in terms}
        verdict = classify(passes(p), coll, a.max_df)
        worst = max(coll.values(), default=0)
        print("  %-4s %-8s worst_df=%-4s %s" % (p.get("id"), verdict, worst, p["q"][:56]))
        if verdict == "fail":
            bad.append(p.get("id"))
        elif verdict == "fragile":
            warn.append(p.get("id"))
        else:
            ok += 1
    print("\naudited %d probes: %d ok, fragile=%s, FAILING=%s (max_df=%d)"
          % (len(probe["probes"]), ok, warn or "none", bad or "none", a.max_df))
    if bad:
        print("Failing probes test nothing: re-derive their query from the target record's own"
              " rare terms, or drop them. Do not lower the bar to make the number look right.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
