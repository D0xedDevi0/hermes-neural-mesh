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
