"""Contract tests for the remote backend (stdlib only, no network egress).

Runs a stub mesh service in-process on 127.0.0.1 and asserts the adapter's
surface, auth header, failure classes and spool replay.
"""

import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import remote  # noqa: E402

TOKEN = "test-token"


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
                return self._send({"ok": ok, "summary": "OK" if ok else "chain break"},
                                  stub.verify_status)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
