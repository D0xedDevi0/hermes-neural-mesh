#!/bin/bash
# One-command rollback. Three targets, in the order you would actually want them:
#   rollback.sh            -> stop using the remote store, run on the local SQLite mesh
#   rollback.sh --yantrikdb-> full pre-cutover state (yantrikdb provider)
#   rollback.sh --check    -> prove both hops resolve, without restarting anything
set -u
RJ=/opt/data/neural-mesh/remote.json
CACHE=/opt/data/.hermes/cache/memcutover
PY=/opt/hermes/.venv/bin/python

resolve() {  # prints the provider's resolved backend
  $PY - <<'PY'
import importlib.util, sys
ROOT = "/opt/data/projects/hermes-neural-mesh"; sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location("nm_plugin", ROOT + "/__init__.py",
        submodule_search_locations=[ROOT])
mod = importlib.util.module_from_spec(spec); sys.modules["nm_plugin"] = mod
spec.loader.exec_module(mod)
p = mod.NeuralMeshProvider()
url, _ = mod.remote_settings_file()
if url:
    print("remote", url); raise SystemExit
p.initialize("rollback-check", hermes_home="/opt/data/.hermes", agent_context="primary")
p._m()
import sqlite3
from pathlib import Path
store = Path("/opt/data/neural-mesh/mesh.db")
n = sqlite3.connect("file:%s?mode=ro" % store, uri=True).execute("select count(*) from nodes").fetchone()[0]
print("local", n, "nodes")
PY
}

case "${1:-}" in
  --check)
    if [ -f "$RJ" ]; then
      before=$(resolve); echo "now:    $before"
      mv "$RJ" "$RJ.disabled"
      after=$(resolve); mv "$RJ.disabled" "$RJ"
      echo "without: $after"
      case "$after" in local*) echo "CHECK OK: remote off -> local mesh is usable, remote.json restored";; *) echo "CHECK FAILED"; exit 1;; esac
    else
      echo "now:    $(resolve)"; echo "CHECK OK: already local"
    fi
    ;;
  --yantrikdb)
    cp "$RJ" "$RJ.disabled" 2>/dev/null || true
    cp "$CACHE/config.before-flip.yaml" /opt/data/config.yaml
    hermes gateway restart >/dev/null 2>&1 || true
    sleep 5
    hermes memory status | sed -n '1,10p'
    echo "ROLLBACK: provider is yantrikdb again. Canonical store untouched; remote.json kept as $RJ.disabled."
    ;;
  *)
    [ -f "$RJ" ] && mv "$RJ" "$RJ.disabled"
    hermes gateway restart >/dev/null 2>&1 || true
    sleep 5
    hermes memory status | sed -n '1,10p'
    echo "ROLLBACK: memory is local again (/opt/data/neural-mesh/mesh.db). Canonical store untouched; re-enable with: mv $RJ.disabled $RJ"
    ;;
esac
