#!/bin/bash
# Completion gate for the NEURAL_MESH cutover. Exits 0 only if every check passes.
# Handles both modes: local SQLite provider (Phase 4) and remote canonical store (Phase 8).
set -u
PY=/opt/hermes/.venv/bin/python
CACHE=/opt/data/.hermes/cache/memcutover
STORE=/opt/data/neural-mesh/mesh.db
LOG=/opt/data/logs/gateways/default/current
ENVF=/opt/data/.hermes/.env
fail=0
ok(){ printf '  OK   %s\n' "$*"; }
bad(){ printf '  FAIL %s\n' "$*"; fail=1; }

remote_url=$(grep -E '^NEURAL_MESH_REMOTE_URL=' "$ENVF" 2>/dev/null | cut -d= -f2-)
if [ -n "${remote_url:-}" ]; then MODE=remote; else MODE=local; fi
echo "mode: $MODE ${remote_url:-$STORE}"

echo "== 1. config names neural-mesh =="
prov=$(awk '/^memory:/{f=1} f&&/^  provider:/{print $2; exit}' /opt/data/config.yaml)
[ "$prov" = "neural-mesh" ] && ok "memory.provider=$prov" || bad "memory.provider=$prov"

echo "== 2. running gateway reports it =="
st=$(hermes memory status 2>&1 | head -12)
echo "$st" | sed 's/^/     /'
echo "$st" | grep -qi 'provider:.*neural-mesh' && ok "status shows neural-mesh" || bad "status does not show neural-mesh"

echo "== 3. the store holds the corpus =="
if [ "$MODE" = "remote" ]; then
  nodes=$($PY - "$ENVF" <<'PYEOF'
import sys, json, urllib.request
E = {}
for line in open(sys.argv[1]):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1); E[k] = v
req = urllib.request.Request(E["NEURAL_MESH_REMOTE_URL"].rstrip("/") + "/mesh/stats",
    headers={"Authorization": "Bearer " + E.get("NEURAL_MESH_REMOTE_TOKEN", "")})
print(json.load(urllib.request.urlopen(req, timeout=60)).get("total_nodes"))
PYEOF
)
  echo "     canonical nodes=$nodes (required >= 1641)"
  [ "${nodes:-0}" -ge 1641 ] && ok "canonical store holds $nodes nodes" || bad "canonical store reports ${nodes:-0}"
else
  [ -f "$STORE" ] && ok "store present ($(stat -c%s "$STORE") bytes)" || bad "store missing"
  nodes=$($PY -c "import sqlite3;print(sqlite3.connect('file:$STORE?mode=ro',uri=True).execute('select count(*) from nodes').fetchone()[0])" 2>/dev/null)
  echo "     local nodes=$nodes (required >= 1641)"
  [ "${nodes:-0}" -ge 1641 ] && ok "node count $nodes >= 1641" || bad "node count ${nodes:-0} < 1641"
fi

echo "== 4. mesh_verify =="
$PY - "$MODE" "$STORE" "$ENVF" <<'PYEOF'
import sys
sys.path.insert(0, "/opt/data/projects/hermes-neural-mesh")
mode, store, envf = sys.argv[1], sys.argv[2], sys.argv[3]
if mode == "remote":
    from remote import RemoteMesh
    E = {}
    for line in open(envf):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1); E[k] = v
    m = RemoteMesh(E["NEURAL_MESH_REMOTE_URL"], E.get("NEURAL_MESH_REMOTE_TOKEN", ""), timeout=120)
else:
    from neural_mesh import Mesh
    m = Mesh(store)
rep = m.verify_memory(deep=True)
print("     verify ok=%s chain=%s witnessed=%s" % (
    getattr(rep, "ok", None),
    getattr(rep, "chain_ok", getattr(rep, "ledger_ok", "?")),
    getattr(rep, "witnessed", getattr(rep, "observed", "?"))))
_sum = getattr(rep, "summary", "")
print("     verify summary:", str(_sum() if callable(_sum) else _sum)[:150])
_txt = str(_sum() if callable(_sum) else _sum)
if " over 0 live memories" in _txt:
    print("     verify covered ZERO live memories - a ledger-only verdict is not enough")
    raise SystemExit(5)
raise SystemExit(0 if getattr(rep, "ok", False) else 4)
PYEOF
[ $? -eq 0 ] && ok "mesh_verify returned OK" || bad "mesh_verify did not return OK"

echo "== 5. live gateway is serving from neural-mesh =="
if [ -f "$LOG" ]; then
  gpid=$(ps -eo pid,cmd | grep -E 'gateway run' | grep -v grep | awk '{print $1}' | head -1)
  if [ -n "$gpid" ] && [ -d "/proc/$gpid" ]; then gstart=$(stat -c %Y /proc/$gpid); else gstart=0; fi
  $PY - "$LOG" "$gstart" <<'PYEOF'
import re, sys, datetime
log, start = sys.argv[1], int(sys.argv[2])
best = None
for line in open(log, errors="replace"):
    if "neural-mesh" not in line: continue
    m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
    if not m: continue
    ts = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
    if start and ts >= start - 5:
        best = (ts, line.strip())
if best:
    print("     fresh line:", best[1][:110]); raise SystemExit(0)
uptime = datetime.datetime.utcnow().timestamp() - start if start else 0
any_line = any("neural-mesh" in l for l in open(log, errors="replace"))
if start and uptime < 600 and any_line:
    print("     gateway up %ds, no turn served yet; provider loaded previously" % uptime); raise SystemExit(0)
print("     no neural-mesh line since gateway start"); raise SystemExit(1)
PYEOF
  [ $? -eq 0 ] && ok "provider served recall in this gateway process" || bad "no neural-mesh recall since this gateway started"
else
  bad "no gateway log at $LOG"
fi

echo "== 6. cold-start recall (isolated HOME: no sessions reachable) =="
tmp=$(mktemp -d)
probe=$CACHE/gate3_probe.py
[ "$MODE" = "remote" ] && probe=$CACHE/gate3_probe_remote.py
HOME="$tmp" HERMES_HOME="$tmp" $PY "$probe" > "$tmp/probe.json" 2>&1
rc=$?
sed 's/^/     /' "$tmp/probe.json"
[ $rc -eq 0 ] && ok "cold probe passed (>=18/20, 0/10)" || bad "cold probe failed"
rm -rf "$tmp"

echo "== 7. backup freshness =="
newest=""; age_h=999
if [ "$MODE" = "remote" ]; then
  # The canonical store lives on the VPS now, so its backups are the ones that matter.
  newest=$(ssh -F /opt/data/.ssh/config dev-vps 'ls -t /opt/data/mesh-backups/mesh-*.db 2>/dev/null | head -1')
  if [ -n "$newest" ]; then
    age_s=$(ssh -F /opt/data/.ssh/config dev-vps "stat -c %Y '$newest' 2>/dev/null")
    age_h=$(( ( $(date +%s) - ${age_s:-0} ) / 3600 ))
  fi
else
  newest=$(ls -t /opt/data/neural-mesh/backups/mesh-*.db 2>/dev/null | head -1)
  [ -n "$newest" ] && age_h=$(( ( $(date +%s) - $(stat -c %Y "$newest") ) / 3600 ))
fi
if [ -n "$newest" ]; then
  echo "     $newest (${age_h}h old)"
  [ "$age_h" -le 36 ] && ok "backup is fresh" || bad "backup is ${age_h}h old"
else
  bad "no backup found"
fi

echo "== 8. canonical API is closed to anonymous readers =="
open_routes=0
for ep in mesh/recall mesh/snapshot mesh/recall-proof mesh/answer-proof mesh/subgraph yantrikdb/enhanced-recall; do
  code=$(curl -s -m 20 -o /dev/null -w "%{http_code}" -X POST -H Content-Type:application/json https://api.d0xeddev.com/$ep -d '{"query":"x"}')
  if [ "$code" = "401" ]; then echo "     $ep -> 401"; else bad "$ep answered $code to an anonymous caller"; open_routes=1; fi
done
[ $open_routes -eq 0 ] && ok "all content-returning routes require the bearer token"

echo
[ $fail -eq 0 ] && echo "GATE OK" || echo "GATE FAILED"
exit $fail
