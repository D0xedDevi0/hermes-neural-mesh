
"""Gate 8: prove one shared memory between this box (brain) and the VPS worker.

Run on the brain host AFTER NEURAL_MESH_REMOTE_URL/TOKEN are set and the gateway restarted.
Exit 0 only if all five steps pass.
"""
import json, os, subprocess, sys, time, urllib.request, urllib.error

BASE = os.environ.get("NEURAL_MESH_REMOTE_URL", "https://api.d0xeddev.com")
LH = "http://" + "127" + ".0.0.1:4021"
TOKEN = os.environ.get("NEURAL_MESH_REMOTE_TOKEN") or subprocess.run(
    ["ssh", "-F", "/opt/data/.ssh/config", "dev-vps",
     "sudo -n grep -E '^NEURAL_MESH_API_TOKEN=' /opt/data/NEURAL_MESH/.env | cut -d= -f2-"],
    capture_output=True, text=True, timeout=60).stdout.strip()
SPOOL = os.environ.get("NEURAL_MESH_SPOOL", "/opt/data/neural-mesh/spool.jsonl")
fails = []

def ok(msg): print("  OK   " + msg)
def bad(msg): print("  FAIL " + msg); fails.append(msg)

def http(path, payload=None, token=TOKEN, method=None):
    data = json.dumps(payload).encode() if payload is not None else None
    hdr = {"Content-Type": "application/json"}
    if token: hdr["Authorization"] = "Bearer " + token
    req = urllib.request.Request(BASE + path, data=data, headers=hdr,
                                 method=method or ("POST" if data else "GET"))
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())

def vps(cmd):
    return subprocess.run(["ssh", "-F", "/opt/data/.ssh/config", "dev-vps", cmd],
                          capture_output=True, text=True, timeout=120).stdout.strip()

def local(path, payload):
    import base64
    blob = base64.b64encode(json.dumps(payload).encode()).decode()
    return vps("echo " + blob + " | base64 -d | curl -s -m 30 -X POST " + LH + path +
               " -H 'Authorization: Bearer " + TOKEN + "' -H 'Content-Type: application/json' -d @-")

sys.path.insert(0, "/opt/data/projects/hermes-neural-mesh")
from remote import RemoteMesh

mesh = RemoteMesh(BASE, TOKEN, timeout=90, spool_path=SPOOL)
stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

print("== 1. write here, read there ==")
fact = "gate8 shared-memory probe " + stamp + ": the canonical store answers to both hosts"
node = mesh.add(fact, type="semantic", provenance="gate8", by="portal-brain")
nid = getattr(node, "id", None) or (node.get("id") if isinstance(node, dict) else None)
if nid: ok("portal write -> " + str(nid))
else:
    bad("portal write returned no id"); sys.exit(1)

r = vps("curl -s -m 20 -w '\\nTIME:%{time_total}' -X POST " + LH + "/mesh/recall -H 'Authorization: Bearer " + TOKEN + "' -H 'Content-Type: application/json' -d '" + json.dumps({"query": "gate8 shared-memory probe " + stamp, "limit": 3}) + "'")
body, _, tval = r.rpartition("TIME:")
elapsed = float(tval) if tval.strip() else -1.0
hit = "gate8 shared-memory probe" in body and stamp in body
(ok if hit else bad)("vps recall saw the fact; server time %.2fs (limit 2s)" % elapsed)
if elapsed < 0 or elapsed >= 2: bad("server-side recall took %.2fs" % elapsed)

print("== 2. supersede there, visible here ==")
new_fact = "gate8 shared-memory probe " + stamp + " REVISED: the value changed on the worker"
local("/mesh/add", {"content": new_fact, "type": "semantic", "provenance": "gate8", "supersedes": nid})
time.sleep(1)
res = mesh.recall(new_fact, top_k=3)
texts = [getattr(n, "content", "") for n in (res or [])]
(ok if any("REVISED" in t for t in texts) else bad)("portal sees the revised value")

print("== 3. stats and verify agree on both hosts ==")
a = mesh.stats().get("total_nodes")
b = json.loads(vps("curl -s -m 10 -H 'Authorization: Bearer " + TOKEN + "' " + LH + "/mesh/stats")).get("total_nodes")
(ok if a == b else bad)("node counts: portal=%s vps=%s" % (a, b))
rep = mesh.verify_memory(deep=False)
(ok if getattr(rep, "ok", False) else bad)("mesh_verify from portal: %s" % getattr(rep, "ok", None))

print("== 4. writes without a token are refused ==")
try:
    http("/mesh/add", {"content": "should not land"}, token=None)
    bad("unauthenticated add was accepted")
except urllib.error.HTTPError as e:
    (ok if e.code == 401 else bad)("unauthenticated add -> %s" % e.code)

print("== 5. service down: spool, restart, replay ==")
before = os.path.getsize(SPOOL) if os.path.exists(SPOOL) else 0
vps("sudo -n systemctl stop neural-mesh")
try:
    spooled = mesh.add("gate8 spool probe " + stamp, type="semantic", provenance="gate8-offline", by="portal-brain")
    pending = bool(getattr(spooled, "pending", False))
    grew = (os.path.getsize(SPOOL) if os.path.exists(SPOOL) else 0) > before
    (ok if pending else bad)("offline write reported pending=%s" % pending)
    (ok if grew else bad)("spool grew by %d bytes" % ((os.path.getsize(SPOOL) if os.path.exists(SPOOL) else 0) - before))
finally:
    vps("sudo -n systemctl start neural-mesh")
    time.sleep(5)
mesh.add("gate8 replay trigger " + stamp, type="semantic", provenance="gate8", by="portal-brain")
time.sleep(2)
left = os.path.getsize(SPOOL) if os.path.exists(SPOOL) else 0
(ok if left == 0 else bad)("spool drained to %d bytes" % left)
r2 = local("/mesh/recall", {"query": "gate8 spool probe " + stamp, "top_k": 3})
(ok if ("gate8 spool probe " + stamp) in r2 else bad)("spooled fact is on the canonical store after replay")

# Leave no probe behind: the canonical store is real memory, not a scratch pad.
try:
    n = http("/mesh/forget-probes", {}).get("forgotten", -1)
    (ok if n >= 0 else bad)("retired %d gate8 probe node(s) from the canonical store" % n)
except Exception as e:
    bad("probe cleanup failed: %s" % e)

print()
print("GATE 8 OK" if not fails else "GATE 8 FAILED (%d)" % len(fails))
sys.exit(0 if not fails else 1)
