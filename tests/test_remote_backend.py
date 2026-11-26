"""Contract tests for the remote backend (stdlib only, no network egress).

Runs a stub mesh service in-process on 127.0.0.1 and asserts the adapter's
surface, auth header, failure classes and spool replay.
"""

import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import remote  # noqa: E402

TOKEN = "test-token"

# The shape the live service answers /mesh/ledger/verify with (trimmed to one label),
# so the adapter is tested against the payload it really gets, not a two-key stub.
VERIFY_OK = {
    "ok": True,
    "summary": ("ledger OK \u2014 6369 entries verified over 5520 live memories, 2179 "
                "adopted, 2342 signed, 2 label(s) bound, 174 signed under an unbound "
                "label; head #6369; anchored: yes @#6025"),
    "kind": "", "reason": "", "advice": "", "broken_at": 0, "node_id": "",
    "checked": 6369, "nodes": 5520, "anchored": True, "anchor_seq": 6025,
    "sinks": ["/var/lib/neural-mesh/anchors.jsonl"], "sinks_in_store_only": False,
    "fields": [], "signed": 2342, "unsigned": 4027, "unchecked": 0,
    "unbound_signed": 174, "unbound_labels": ["gate8"],
    "bindings_source": "/var/lib/neural-mesh/identities.jsonl",
}


def _provider_module():
    """Load the plugin the way test_provider.py does: the real tool renderer, not a
    copy of its assumptions about what a verify report carries."""
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))
    if os.path.isdir("/opt/hermes"):
        sys.path.insert(0, "/opt/hermes")
    spec = importlib.util.spec_from_file_location("nm_provider_remote", repo / "__init__.py")
    nm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nm)
    return nm


class Stub:
    def __init__(self):
        self.adds = []
        self.last_recall_body = {}
        self.add_status = 200
        self.auth = []
        self.verify_status = 200


def handler_factory(stub: Stub):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            return

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def _send(self, payload, status=200):
            blob = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def do_GET(self):
            stub.auth.append(self.headers.get("Authorization"))
            if self.path == "/mesh/stats":
                return self._send({"total_nodes": 3, "active_nodes": 3})
            return self._send({"error": "not found"}, 404)

        def do_POST(self):
            stub.auth.append(self.headers.get("Authorization"))
            body = self._body()
            if self.path == "/mesh/add":
                if stub.add_status != 200:
                    return self._send({"error": "upstream down"}, stub.add_status)
                stub.adds.append(body)
                return self._send({"id": "n%d" % len(stub.adds),
                                   "content": body.get("content"), "type": body.get("type")})
            if self.path == "/mesh/recall":
                stub.last_recall_body = body
                return self._send({"results": [
                    {"id": "a", "content": "alpha", "type": "semantic", "score": 0.9},
                    {"id": "b", "content": "beta", "type": "semantic", "score": 0.2},
                ]})
            if self.path == "/mesh/snapshot":
                return self._send({"results": [{"id": "a", "content": "alpha"}], "count": 1})
            if self.path == "/mesh/ledger/verify":
                ok = stub.verify_status == 200
                body = dict(VERIFY_OK) if ok else {"ok": False, "summary": "chain break"}
                return self._send(body, stub.verify_status)
            if self.path == "/mesh/ledger/provenance":
                return self._send({"witnessed": True, "events": [{"op": "add"}]})
            return self._send({"error": "not found"}, 404)

    return Handler


def serve(stub: Stub):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_factory(stub))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, "http://127.0.0.1:%d" % srv.server_address[1]


class RemoteBackendTests(unittest.TestCase):
    def setUp(self):
        self.stub = Stub()
        self.srv, self.url = serve(self.stub)
        self.tmp = tempfile.mkdtemp()
        self.spool = os.path.join(self.tmp, "spool.jsonl")

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()

    def test_add_sends_token_and_returns_node(self):
        mesh = remote.RemoteMesh(self.url, token=TOKEN, spool_path=self.spool)
        node = mesh.add("the vault is at 0xdead", type="semantic", provenance="test")
        self.assertFalse(node.pending)
        self.assertEqual(node.id, "n1")
        self.assertEqual(self.stub.adds[0]["content"], "the vault is at 0xdead")
        self.assertEqual(self.stub.auth[-1], "Bearer " + TOKEN)

    def test_recall_reranks_locally_and_filters_by_coverage(self):
        # The service returns no score, so the adapter ranks the candidate set itself
        # with BM25 and reports IDF coverage: 1.0 for an exact match, and an
        # unrelated candidate is dropped rather than ordered last.
        mesh = remote.RemoteMesh(self.url, token=TOKEN)
        hits = mesh.recall("alpha", top_k=5, min_score=0.5)
        self.assertEqual([h.id for h in hits], ["a"])
        self.assertAlmostEqual(hits[0].score, 1.0)

    def test_recall_asks_for_a_wide_candidate_set(self):
        mesh = remote.RemoteMesh(self.url, token=TOKEN)
        mesh.recall("alpha", top_k=2)
        self.assertGreaterEqual(self.stub.last_recall_body["limit"], 60)

    def test_add_forwards_bitemporal_bounds(self):
        # valid_from/valid_to are what make a correction scope-correct; the server
        # drops unknown fields, so the adapter has to put them on the wire.
        mesh = remote.RemoteMesh(self.url, token=TOKEN)
        mesh.add("correction", type="semantic", valid_from=1764200000, valid_to=1764300000)
        sent = self.stub.adds[-1]
        self.assertEqual(sent["valid_from"], 1764200000)
        self.assertEqual(sent["valid_to"], 1764300000)

    def test_502_from_the_proxy_spools_the_write(self):
        # nginx answering 502 while the service is stopped must queue the write,
        # not raise: that is the failure the spool exists for.
        self.stub.add_status = 502
        mesh = remote.RemoteMesh(self.url, token=TOKEN, spool_path=self.spool)
        node = mesh.add("queued during outage", type="semantic", provenance="test")
        self.assertTrue(node.pending)
        self.assertTrue(node.id.startswith("spool-"))
        self.assertGreater(os.path.getsize(self.spool), 0)

    def test_snapshot_feeds_load_for_id_prefix_resolution(self):
        mesh = remote.RemoteMesh(self.url, token=TOKEN)
        self.assertEqual(list(mesh._load().keys()), ["a"])

    def test_verify_409_is_a_failed_verdict_not_an_exception(self):
        self.stub.verify_status = 409
        mesh = remote.RemoteMesh(self.url, token=TOKEN)
        rep = mesh.verify_memory()
        self.assertFalse(rep.ok)
        self.assertIn("chain break", rep.summary())

    def test_unreachable_write_spools_then_replays(self):
        # port 1 is refused, so this is a real unreachable service
        mesh = remote.RemoteMesh("http://127.0.0.1:1", token=TOKEN, spool_path=self.spool,
                                 timeout=0.5)
        node = mesh.add("written while the service was down")
        self.assertTrue(node.pending)
        self.assertEqual(node.id[:6], "spool-")
        self.assertEqual(mesh.pending, 1)

        mesh.base_url = self.url
        self.assertEqual(mesh.replay(), 0)
        self.assertEqual(len(self.stub.adds), 1)
        self.assertEqual(self.stub.adds[0]["content"], "written while the service was down")

    def test_reads_fail_loudly_when_unreachable(self):
        mesh = remote.RemoteMesh("http://127.0.0.1:1", token=TOKEN, timeout=0.5)
        with self.assertRaises(remote.RemoteUnavailable):
            mesh.recall("anything")

    def test_stats_is_a_get(self):
        mesh = remote.RemoteMesh(self.url, token=TOKEN)
        self.assertEqual(mesh.stats()["total_nodes"], 3)

    def test_verify_gets_a_timeout_long_enough_for_a_deep_scan(self):
        # Measured: a deep verify of 6,369 entries takes ~24.8s over the public
        # endpoint, and it grows with the ledger. With the 5s request default the
        # client always gave up before the service answered, so verify_memory needs
        # a budget of its own.
        seen = {}
        real = remote.urllib.request.urlopen

        def fake(req, timeout=None):
            seen["timeout"] = timeout
            raise OSError("stop after recording the timeout")

        remote.urllib.request.urlopen = fake
        try:
            with self.assertRaises(remote.RemoteUnavailable):
                remote.RemoteMesh(self.url, token=TOKEN).verify_memory(deep=True)
        finally:
            remote.urllib.request.urlopen = real
        self.assertIsNotNone(seen["timeout"])
        self.assertGreaterEqual(seen["timeout"], 60)

    def test_fast_calls_keep_the_short_default(self):
        # The verify budget must not leak into reads: a recall that hangs is a bug,
        # not a slow scan.
        seen = {}
        real = remote.urllib.request.urlopen

        def fake(req, timeout=None):
            seen["timeout"] = timeout
            raise OSError("stop after recording the timeout")

        remote.urllib.request.urlopen = fake
        try:
            with self.assertRaises(remote.RemoteUnavailable):
                remote.RemoteMesh(self.url, token=TOKEN).stats()
        finally:
            remote.urllib.request.urlopen = real
        self.assertEqual(seen["timeout"], remote.DEFAULT_TIMEOUT)


    def test_verify_exposes_every_field_the_provider_renders(self):
        # The provider reads anchored/anchor_seq/sinks on a pass, and kind/reason/
        # broken_at on a fail, without getattr defaults. An adapter that carried only
        # ok+summary raised AttributeError the moment remote verify got a timeout long
        # enough to answer -- the failure the timeout fix uncovered.
        mesh = remote.RemoteMesh(self.url, token=TOKEN)
        rep = mesh.verify_memory()
        self.assertTrue(rep.ok)
        self.assertTrue(rep.anchored)
        self.assertEqual(rep.anchor_seq, 6025)
        self.assertEqual(rep.sinks, ["/var/lib/neural-mesh/anchors.jsonl"])
        for attr in ("kind", "reason", "advice", "node_id", "checked", "nodes", "broken_at",
                     "sinks_in_store_only", "fields", "signed", "unsigned", "unchecked",
                     "unbound_signed", "unbound_labels", "bindings_source"):
            self.assertTrue(hasattr(rep, attr), attr)

    def test_provider_mesh_verify_renders_the_remote_verdict(self):
        # End to end over HTTP: the tool an operator actually calls must come back with
        # a verdict, not an AttributeError.
        home = tempfile.mkdtemp(prefix="nm-remote-home-")
        environ = {"NEURAL_MESH_REMOTE_URL": self.url, "NEURAL_MESH_REMOTE_TOKEN": TOKEN}
        with mock.patch.dict(os.environ, environ):
            provider = _provider_module().NeuralMeshProvider()
            provider.initialize("remote-session", hermes_home=home)
            try:
                out = provider.handle_tool_call("mesh_verify", {})
            finally:
                provider.shutdown()
        self.assertIn("integrity: OK", out, out)
        self.assertIn("6369 entries verified over 5520 live memories", out, out)
        self.assertIn("anchors.jsonl", out, out)
        self.assertIn("Authorship:", out, out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
