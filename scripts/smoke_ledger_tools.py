"""End-to-end check of the vendored Hermes plugin: does the new lane actually work
through the tool surface, and does a tampered store read as FAILED?

Run:  cd /opt/data/projects/hermes-neural-mesh && python3 scripts/smoke_ledger_tools.py
"""
import importlib.util
import json
import os
import sqlite3
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

spec = importlib.util.spec_from_file_location("nm_plugin", os.path.join(HERE, "__init__.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

home = tempfile.mkdtemp(prefix="nm-plugin-smoke-")
prov = mod.NeuralMeshProvider()
prov.initialize("smoke-session", hermes_home=home)

tools = [t["name"] for t in prov.get_tool_schemas()]
print("tools:", tools)
assert "mesh_verify" in tools and "mesh_provenance" in tools, "new tools not registered"

store = os.path.join(home, mod.PROVIDER_NAME, "mesh.db")
assert os.path.exists(store), f"store not created at {store}"

print("\n-- remember --")
print(prov.handle_tool_call("mesh_remember",
                            {"content": "the deploy key rotates every 90 days", "kind": "semantic"}))
print(prov.handle_tool_call("mesh_remember",
                            {"content": "the treasury multisig threshold is 2 of 3",
                             "kind": "semantic"}))

print("\n-- verify (clean) --")
clean = prov.handle_tool_call("mesh_verify", {})
print(clean)
assert "OK" in clean, f"clean store did not verify: {clean}"

print("\n-- provenance --")
nodes = prov._m()._load()
assert nodes, "no memories stored"
nid = sorted(nodes)[0]
prov_line = prov.handle_tool_call("mesh_provenance", {"node_id": nid[:8]})
print(prov_line)
assert "asserted by" in prov_line, prov_line

print("\n-- tamper: edit a stored memory in place, behind the API --")
conn = sqlite3.connect(store)
conn.execute("UPDATE nodes SET content='the deploy key rotates every 900 days' WHERE id=?",
             (nid,))
conn.commit()
conn.close()

verdict = prov.handle_tool_call("mesh_verify", {})
print(verdict)
assert "FAILED" in verdict, f"tampering not detected: {verdict}"
assert "content" in verdict, verdict

print("\n-- tamper: delete a memory row --")
conn = sqlite3.connect(store)
conn.execute("DELETE FROM nodes WHERE id=?", (nid,))
conn.commit()
conn.close()
verdict2 = prov.handle_tool_call("mesh_verify", {})
print(verdict2)
assert "FAILED" in verdict2, verdict2

print("\n-- pre-ledger store: adoption path is explicit, never silent --")
# a store written before the ledger existed reports coverage, and says how to fix it
pre = tempfile.mkdtemp(prefix="nm-plugin-preledger-")
prov2 = mod.NeuralMeshProvider()
prov2.initialize("pre", hermes_home=pre)
prov2.handle_tool_call("mesh_remember", {"content": "an old fact from before the ledger"})
conn = sqlite3.connect(os.path.join(pre, mod.PROVIDER_NAME, "mesh.db"))
conn.execute("DELETE FROM ledger")
conn.commit()
conn.close()
uncov = prov2.handle_tool_call("mesh_verify", {})
print(uncov)
assert "FAILED" in uncov and "coverage" in uncov, uncov
assert "adopt=true" in uncov, f"adoption advice missing: {uncov}"
adopted = prov2.handle_tool_call("mesh_verify", {"adopt": True})
print(adopted)
assert "OK" in adopted, adopted

print("\nSMOKE OK — plugin surface works, tampering is reported, adoption is explicit.")
print(json.dumps({"tools": tools, "store": store}, indent=2))


# ── v0.41.0 ──────────────────────────────────────────────────────────────────
print("\n-- tamper: relabel the author of a memory (metadata only) --")
prov3dir = tempfile.mkdtemp(prefix="nm-plugin-attrib-")
prov3 = mod.NeuralMeshProvider()
prov3.initialize("attrib", hermes_home=prov3dir)
prov3.handle_tool_call("mesh_remember", {"content": "the deploy key rotates every 900 days"})
store3 = os.path.join(prov3dir, mod.PROVIDER_NAME, "mesh.db")
conn = sqlite3.connect(store3)
nid3 = conn.execute("SELECT id FROM nodes LIMIT 1").fetchone()[0]
meta = json.loads(conn.execute("SELECT meta FROM nodes WHERE id=?", (nid3,)).fetchone()[0])
meta["agent_id"] = "attacker"
conn.execute("UPDATE nodes SET meta=? WHERE id=?", (json.dumps(meta), nid3))
conn.commit()
conn.close()
verdict3 = prov3.handle_tool_call("mesh_verify", {})
print(verdict3)
assert "FAILED" in verdict3, verdict3
assert "attribution" in verdict3, f"the report should say WHAT changed: {verdict3}"
assert "agent_id" in verdict3, f"the report should name the field: {verdict3}"

print("\n-- an anchor outside the store is the only thing that catches a clean cut --")
sink_dir = tempfile.mkdtemp(prefix="nm-plugin-sink-")
sink = os.path.join(sink_dir, "anchors.jsonl")
home = tempfile.mkdtemp(prefix="nm-plugin-anchored-")
prov4 = mod.NeuralMeshProvider()
os.environ["NEURAL_MESH_ANCHORS"] = sink
prov4.initialize("anchored", hermes_home=home)
for i in range(5):
    prov4.handle_tool_call("mesh_remember", {"content": f"anchored fact {i} about the treasury"})
anchored = prov4._m().ledger.anchor(target="smoke")
print("anchored:", anchored["seq"], "->", anchored["sink"])
assert anchored["sink"] == os.path.abspath(sink), anchored
del os.environ["NEURAL_MESH_ANCHORS"]

# attacker: cut the log, repair the store, and delete the anchor row with it
conn = sqlite3.connect(os.path.join(home, mod.PROVIDER_NAME, "mesh.db"))
orphans = [r[0] for r in conn.execute(
    "SELECT node_id FROM ledger WHERE node_id != '' GROUP BY node_id HAVING MIN(seq) >= 4"
).fetchall()]
conn.execute("DELETE FROM ledger WHERE seq >= 4")
conn.execute("DELETE FROM ledger_anchor")
for n in orphans:
    conn.execute("DELETE FROM nodes WHERE id=?", (n,))
conn.commit()
conn.close()

# a verifier that only knows the store sees nothing wrong — honestly reported
plain = mod.NeuralMeshProvider()
plain.initialize("anchored", hermes_home=home)
seen = plain.handle_tool_call("mesh_verify", {})
print("no sink  ->", seen[:120])
assert "OK" in seen, f"expected the in-database anchor to protect nothing: {seen}"

# the same verifier, told where the anchors are kept, catches it
sighted = mod.NeuralMeshProvider()
sighted.initialize("anchored", hermes_home=home)
caught = sighted.handle_tool_call("mesh_verify", {"anchors": sink})
print("with sink ->", caught[:200])
assert "FAILED" in caught and "anchor" in caught, caught

print("\nSMOKE.v41 OK — attribution is named, and the external sink catches what the "
      "in-database anchor cannot.")
