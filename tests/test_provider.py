"""Offline tests for the neural-mesh MemoryProvider.

No network, no credentials, no Hermes runtime required: the plugin falls back to a local
stub of the ABC when `agent.memory_provider` is not importable, so these run anywhere.

    python3 -m unittest discover -s tests -v
"""

import importlib.util
import json
import os
import re
import sys
import tempfile
from unittest import mock
import threading
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
if os.path.isdir("/opt/hermes"):  # real ABC when present, stub otherwise
    sys.path.insert(0, "/opt/hermes")

spec = importlib.util.spec_from_file_location("nm_provider", REPO / "__init__.py")
nm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nm)

JAN = "2026-01-01"
JUN = "2026-06-01"


class ProviderTest(unittest.TestCase):
    def setUp(self):
        # Hermetic by construction: an environment that points at a canonical store must
        # never be able to receive this suite's fixture writes.
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for key in ("NEURAL_MESH_REMOTE_URL", "NEURAL_MESH_REMOTE_TOKEN"):
            os.environ.pop(key, None)
        self.home = tempfile.mkdtemp(prefix="nm-test-")
        self.p = nm.NeuralMeshProvider()
        self.p.initialize("test-session", hermes_home=self.home, agent_context="primary")

    def tearDown(self):
        self.p.shutdown()

    # ── identity and lifecycle ──────────────────────────────────────────────────

    def test_name_and_availability_need_no_credentials(self):
        self.assertEqual(self.p.name, "neural-mesh")
        for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            os.environ.pop(key, None)
        self.assertTrue(self.p.is_available())

    def test_initialize_creates_provider_owned_storage(self):
        self.assertTrue((Path(self.home) / "neural-mesh").is_dir())
        self.assertEqual(self.p.backup_paths(), [str(Path(self.home) / "neural-mesh")])

    def test_tools_are_declared_and_callable(self):
        names = {t["name"] for t in self.p.get_tool_schemas()}
        self.assertEqual(names, {"mesh_remember", "mesh_recall", "mesh_asof", "mesh_stats",
                                 "mesh_verify", "mesh_provenance", "mesh_curate"})
        # Declared means reachable: call every tool for real. A schema with no
        # handler branch returns "unknown tool", which this catches.
        probes = {
            "mesh_remember": {"content": "the probe fact is reachable from search"},
            "mesh_recall": {"query": "probe fact"},
            "mesh_asof": {"query": "probe fact", "as_of": JUN},
            "mesh_stats": {},
            "mesh_verify": {},
            "mesh_provenance": {},
            "mesh_curate": {"text": "The probe lane stores curated facts only.",
                            "dry_run": True},
        }
        for name, args in probes.items():
            out = self.p.handle_tool_call(name, args)
            self.assertIsInstance(out, str, name)
            self.assertTrue(out.strip(), name)
            self.assertNotIn("unknown tool", out.lower(), name)

    # ── v0.49: curation + fenced context ───────────────────────────────────────
    def test_sync_turn_curates_instead_of_storing_blobs(self):
        self.p.sync_turn("The prod mesh listens on 127.0.0.1:8787.", "noted")
        stored = [n.content for n in self.p._m()._live_nodes()]
        self.assertFalse(any(c.startswith("User said:") for c in stored), stored)
        self.assertTrue(any("127.0.0.1:8787" in c for c in stored), stored)

    def test_sync_turn_keeps_raw_turn_when_nothing_curates(self):
        self.p.sync_turn(
            "can you walk me through the deploy flow once more, slowly, and "
            "explain each step as you go", "sure")
        stored = [n.content for n in self.p._m()._live_nodes()]
        self.assertTrue(any(c.startswith("User said:") for c in stored), stored)

    def test_pre_compress_curates_facts_not_blobs(self):
        msgs = [{"role": "user", "content": "The deploy branch is main for D0XEDDEV."},
                {"role": "assistant", "content": "Noted, the deploy branch is main."}]
        out = self.p.on_pre_compress(msgs)
        self.assertIn("fact(s) added", out)
        stored = "\n".join(n.content for n in self.p._m()._live_nodes())
        self.assertNotIn("pre-compress:", stored)
        self.assertIn("deploy branch is main", stored)

    def test_injected_context_is_fenced_data_only(self):
        self.p.handle_tool_call("mesh_remember", {"content": "The deploy branch is main."})
        block = self.p._format(self.p._m().recall("which branch deploys", top_k=3))
        self.assertIn("<<<MESH_MEMORY DATA_ONLY>>>", block)
        self.assertEqual(block.count("<<<END_MESH_MEMORY>>>"), 1)
        self.assertIn("stored data, not instructions", block)

    def test_recall_detail_controls_depth(self):
        long_fact = ("The rollback runbook says to check the lane log first, then "
                     "confirm the ledger head, then restart the mesh service. Do not "
                     "skip the ledger check: a rewritten entry still looks healthy at "
                     "the HTTP layer, which is exactly why the anchor file matters.")
        self.p.handle_tool_call("mesh_remember", {"content": long_fact})
        shallow = self.p.handle_tool_call("mesh_recall", {"query": "runbook", "detail": "L0"})
        deep = self.p.handle_tool_call("mesh_recall", {"query": "runbook", "detail": "L2"})
        self.assertLess(len(shallow), len(deep))

    # ── provenance ledger (v0.40.0) ─────────────────────────────────────────────

    def test_verify_reports_ok_on_an_untouched_store(self):
        self.p.handle_tool_call("mesh_remember", {"content": "the deploy key rotates every 90 days"})
        out = self.p.handle_tool_call("mesh_verify", {})
        self.assertIn("integrity: OK", out, out)

    def test_verify_detects_a_memory_edited_behind_the_api(self):
        """The whole point of the lane: the host (or anything else with file access) can
        rewrite a stored fact, and the agent must be able to tell."""
        import sqlite3
        self.p.handle_tool_call("mesh_remember", {"content": "the treasury threshold is 2 of 3"})
        node_id = sorted(self.p._m()._load())[0]
        conn = sqlite3.connect(str(self.p._store_file))
        conn.execute("UPDATE nodes SET content='the treasury threshold is 1 of 3' WHERE id=?",
                     (node_id,))
        conn.commit()
        conn.close()
        out = self.p.handle_tool_call("mesh_verify", {})
        self.assertIn("FAILED", out, out)
        self.assertIn("content", out, out)

    def test_verify_detects_a_deleted_memory(self):
        import sqlite3
        self.p.handle_tool_call("mesh_remember", {"content": "an inconvenient fact"})
        node_id = sorted(self.p._m()._load())[0]
        conn = sqlite3.connect(str(self.p._store_file))
        conn.execute("DELETE FROM nodes WHERE id=?", (node_id,))
        conn.commit()
        conn.close()
        out = self.p.handle_tool_call("mesh_verify", {})
        self.assertIn("FAILED", out, out)
        self.assertIn("orphan", out, out)

    def test_verify_says_when_a_green_verdict_rests_on_no_anchor(self):
        """\"intact\" is a weaker claim without an anchor. A clean truncation of the
        newest entries leaves the chain valid, so the tool has to say which of the
        two verdicts this is instead of implying the stronger one."""
        self.p.handle_tool_call("mesh_remember", {"content": "nothing is pinned yet"})
        out = self.p.handle_tool_call("mesh_verify", {})
        self.assertIn("integrity: OK", out, out)
        self.assertIn("Not anchored", out, out)
        self.assertIn("truncation", out, out)

    def test_verify_names_the_sink_outside_the_store_when_there_is_one(self):
        sink = os.path.join(tempfile.mkdtemp(), "anchors.jsonl")
        self.p.handle_tool_call("mesh_remember", {"content": "the release tag is v0.45.0"})
        mesh = self.p._m()
        mesh.ledger.anchor_file = sink        # what NEURAL_MESH_ANCHORS would set
        mesh.ledger.anchor(target="test")
        out = self.p.handle_tool_call("mesh_verify", {"anchors": sink})
        self.assertIn("integrity: OK", out, out)
        self.assertIn(sink, out, out)
        self.assertIn("outside the store", out, out)

    def test_verify_flags_an_anchor_that_lives_inside_the_store(self):
        """An anchor in the database is deleted together with the entries it
        protects, so a green verdict resting on one must not read as a guarantee."""
        self.p.handle_tool_call("mesh_remember", {"content": "anchored in place"})
        self.p._m().ledger.anchor(target="in-store")
        out = self.p.handle_tool_call("mesh_verify", {})
        self.assertIn("integrity: OK", out, out)
        self.assertIn("lives in this database", out, out)
        self.assertIn("NEURAL_MESH_ANCHORS", out, out)

    # ── signed entries and identity bindings (upstream v0.47/v0.48) ─────────────

    # The canonical, publicly documented throwaway test key (anvil/hardhat account #0).
    # Never funded, never used anywhere real: it exists so a signature test needs no secret.
    SIGNER_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

    def _signed_mesh(self):
        """(mesh, db_path) whose ledger signs every entry, or (None, None) without eth_account."""
        from neural_mesh import Mesh
        from neural_mesh.ledger import has_signing_lib
        if not has_signing_lib():
            return None, None
        path = os.path.join(tempfile.mkdtemp(prefix="nm-signed-"), "signed.db")
        return Mesh(path, signer=self.SIGNER_KEY), path

    def test_verify_names_a_forged_signature_rather_than_a_generic_failure(self):
        """For a signing ledger the sharpest verdict is a signature that no longer
        recovers to the author it claims -- the signature was edited. If this build
        renders that as "an integrity invariant failed" it has flattened the one result
        an operator must act on, so the kind has to name itself."""
        import sqlite3
        mesh, path = self._signed_mesh()
        if mesh is None:
            self.skipTest("eth_account not installed: cannot produce a signed entry")
        mesh.add("the release tag is v0.48.0")
        self.assertGreater(mesh.verify_memory().signed, 0, "the ledger should have signed")
        conn = sqlite3.connect(path)
        conn.execute("UPDATE ledger SET sig=? WHERE seq=(SELECT max(seq) FROM ledger)",
                     ("ab" * 65,))
        conn.commit()
        conn.close()
        self.p._m = lambda: mesh
        out = self.p.handle_tool_call("mesh_verify", {})
        self.assertIn("FAILED", out, out)
        self.assertIn("signature", out, out)
        self.assertIn("does not recover", out, out)
        self.assertNotIn("an integrity invariant failed", out, out)

    def test_verify_says_how_much_authorship_it_actually_checked(self):
        """A signed ledger proves a key signed each entry. "OK" must not read as
        "authorship was proven" unless it also says how many signatures verified -- and
        that no registry tied the key to a label."""
        mesh, _ = self._signed_mesh()
        if mesh is None:
            self.skipTest("eth_account not installed: cannot produce a signed entry")
        mesh.add("a signed fact")
        self.p._m = lambda: mesh
        out = self.p.handle_tool_call("mesh_verify", {})
        self.assertIn("integrity: OK", out, out)
        self.assertIn("Authorship:", out, out)
        self.assertIn("verified", out, out)
        self.assertIn("No identity registry", out, out)

    def test_verify_never_flattens_a_kind_it_does_not_know(self):
        """A verdict kind added upstream must surface as itself: the old fallback text
        ("an integrity invariant failed") is the vagueness this lane exists to remove."""
        from neural_mesh.ledger import VerifyReport

        class _Stub:
            ledger = object()

            def __init__(self, r): self._r = r

            def verify_memory(self): return self._r

        rep = VerifyReport(unexplained=[], head={"seq": 7}, fields=[], sinks=[],
                           bindings={}, unbound_labels=[])
        rep.ok, rep.kind, rep.broken_at = False, "future-kind", 7
        rep.reason = "something new broke"
        self.p._m = lambda: _Stub(rep)
        out = self.p.handle_tool_call("mesh_verify", {})
        self.assertIn("future-kind", out, out)
        self.assertIn("does not know failed", out, out)
        self.assertNotIn("an integrity invariant failed", out, out)

    def test_verify_names_labels_the_registry_does_not_bind(self):
        """An unbound signed entry is advisory, not fatal -- but a green verdict that
        omits it implies the signer's identity was established when it was not."""
        from neural_mesh.ledger import VerifyReport

        class _Stub:
            ledger = object()

            def __init__(self, r): self._r = r

            def verify_memory(self): return self._r

        rep = VerifyReport(unexplained=[], head={"seq": 9, "entry_hash": "ab" * 32},
                           fields=[], sinks=[], bindings={},
                           unbound_labels=["scanner-bot", "fresh-launch"])
        rep.ok, rep.checked, rep.nodes, rep.anchored = True, 9, 9, False
        rep.signed, rep.unbound_signed = 3, 2
        self.p._m = lambda: _Stub(rep)
        out = self.p.handle_tool_call("mesh_verify", {})
        self.assertIn("integrity: OK", out, out)
        self.assertIn("does not bind", out, out)
        self.assertIn("scanner-bot", out, out)
        self.assertNotIn("not which agent", out, out)

    def test_provenance_resolves_the_prefix_recall_shows(self):
        self.p.handle_tool_call("mesh_remember", {"content": "the bridge contract is 0xA11CE"})
        node_id = sorted(self.p._m()._load())[0]
        out = self.p.handle_tool_call("mesh_provenance", {"node_id": node_id[:8]})
        self.assertIn("asserted by", out, out)
        self.assertIn(node_id[:8], out)

    def test_provenance_says_so_rather_than_guessing_on_a_bad_id(self):
        self.p.handle_tool_call("mesh_remember", {"content": "the incident channel is #ops"})
        out = self.p.handle_tool_call("mesh_provenance", {"node_id": "zzzzzzzz"})
        self.assertIn("no memory matching", out, out)

    # ── memory round-trip ───────────────────────────────────────────────────────

    def test_remember_then_recall_returns_the_memory(self):
        out = self.p.handle_tool_call("mesh_remember", {"content": "the user's dog is called Kumo"})
        self.assertIn("Kumo", out)
        recalled = self.p.handle_tool_call("mesh_recall", {"query": "what is the dog called?"})
        self.assertIn("Kumo", recalled)

    def test_recall_of_unknown_query_says_so_rather_than_returning_junk(self):
        self.p.handle_tool_call("mesh_remember", {"content": "the deploy key rotates every 90 days"})
        out = self.p.handle_tool_call("mesh_recall", {"query": "zzzz-nonexistent-topic"})
        self.assertTrue(out.startswith("NEURAL_MESH") and "matches" in out, out)

    def test_unrelated_turns_inject_nothing(self):
        """Measured failure mode: with no floor, an unrelated query returned every stored
        memory and prefetch would have polluted the context. The calibrated floor stops it."""
        self.p.handle_tool_call("mesh_remember", {"content": "the deploy key rotates every 90 days"})
        self.p.handle_tool_call("mesh_remember", {"content": "the user's dog is called Kumo"})
        self.assertEqual(self.p.prefetch("quantum chromodynamics lattice spacing"), "")
        self.assertIsNone(self.p.recall_status())

    def test_a_relevant_query_still_injects_above_the_floor(self):
        self.p.handle_tool_call("mesh_remember", {"content": "the deploy key rotates every 90 days"})
        self.assertIn("90 days", self.p.prefetch("how often does the deploy key rotate?"))

    def test_stats_report_the_resonance_backend(self):
        self.p.handle_tool_call("mesh_remember", {"content": "a fact that exists"})
        out = self.p.handle_tool_call("mesh_stats", {})
        self.assertIn("resonance_backend", out)
        # the plugin ships pure Python, but a host that already has the rust
        # accelerator installed will use it — either way the backend is named, and
        # stats must not be the place the plugin breaks on a machine it did not build
        self.assertTrue(any(b in out for b in ("python", "rust")), out)

    # ── the differentiator: point-in-time recall ────────────────────────────────

    def test_mesh_asof_returns_the_value_that_was_true_then(self):
        self.p.handle_tool_call(
            "mesh_remember",
            {"content": "the deploy key rotates every 90 days", "valid_from": JAN},
        )
        time.sleep(0.01)
        self.p.handle_tool_call(
            "mesh_remember",
            {"content": "the deploy key rotates every 30 days", "valid_from": JUN},
        )
        early = self.p.handle_tool_call("mesh_asof", {"query": "how often does the deploy key rotate?", "as_of": "2026-03-01"})
        self.assertIn("90 days", early, f"as-of recall lost the historical value: {early}")
        self.assertNotIn("30 days", early, f"as-of recall leaked a later value: {early}")

    def test_mesh_asof_rejects_an_unparseable_date(self):
        out = self.p.handle_tool_call("mesh_asof", {"query": "anything", "as_of": "last tuesday"})
        self.assertIn("parseable", out)

    # ── gates: what the provider must refuse to do ───────────────────────────────

    def test_trivial_prompts_do_not_trigger_recall(self):
        self.p.handle_tool_call("mesh_remember", {"content": "the user's dog is called Kumo"})
        self.assertEqual(self.p.prefetch("thanks"), "")
        self.assertIsNone(self.p.recall_status())

    def test_non_primary_contexts_do_not_write_to_long_term_memory(self):
        self.p.shutdown()
        self.home2 = tempfile.mkdtemp(prefix="nm-cron-")
        self.p = nm.NeuralMeshProvider()
        self.p.initialize("cron-session", hermes_home=self.home2, agent_context="cron")
        self.p.sync_turn("this is a cron run and must not be remembered", "agreed", session_id="cron-session")
        self.assertIn("nothing", self.p.handle_tool_call("mesh_recall", {"query": "cron run"}).lower())

    def test_short_and_trivial_turns_are_not_stored(self):
        self.p.sync_turn("ok", "sure", session_id="s")
        self.p.sync_turn("hi", "hello", session_id="s")
        self.assertIn("nothing", self.p.handle_tool_call("mesh_recall", {"query": "hi"}).lower())

    # ── prefetch behaviour ──────────────────────────────────────────────────────

    def test_queue_prefetch_prewarms_and_prefetch_consumes_it(self):
        self.p.handle_tool_call("mesh_remember", {"content": "the hackathon deadline is November 11"})
        self.p.queue_prefetch("when is the hackathon deadline?")
        first = self.p.prefetch("when is the hackathon deadline?")
        self.assertIn("November 11", first)
        self.assertEqual(self.p.prefetch("when is the hackathon deadline?"), first)

    # ── mirroring ───────────────────────────────────────────────────────────────

    def test_native_memory_writes_are_mirrored_but_destructive_ones_are_skipped(self):
        self.p.on_memory_write("add", "MEMORY.md", "user prefers square bullets")
        self.assertIn("square bullets", self.p.handle_tool_call("mesh_recall", {"query": "bullet preference"}))
        self.p.on_memory_write("remove", "MEMORY.md", "user prefers square bullets")
        # a delete we cannot identify precisely is not mirrored — and must not crash
        self.assertIn("square bullets", self.p.handle_tool_call("mesh_recall", {"query": "bullet preference"}))

    # ── config ──────────────────────────────────────────────────────────────────

    def test_config_round_trips_through_our_own_file(self):
        self.p.save_config({"recall_top_k": 3, "store_turns": False}, self.home)
        written = json.loads((Path(self.home) / "neural-mesh" / "config.json").read_text())
        self.assertEqual(written["recall_top_k"], 3)
        self.p.shutdown()
        p2 = nm.NeuralMeshProvider()
        p2.initialize("s2", hermes_home=self.home, agent_context="primary")
        self.assertEqual(p2._top_k, 3)
        self.assertFalse(p2._store_turns)
        p2.shutdown()

    def test_kwargs_beat_saved_config(self):
        self.p.save_config({"recall_top_k": 9}, self.home)
        p2 = nm.NeuralMeshProvider()
        p2.initialize("s3", hermes_home=self.home, recall_top_k=2, agent_context="primary")
        self.assertEqual(p2._top_k, 2)
        p2.shutdown()

    # ── supersession and point-in-time: the differentiator ──────────────────────

    def test_a_correction_supersedes_without_deleting_the_history(self):
        """The headline behaviour: the correction wins going forward, and the earlier belief
        is still answerable for the date range in which it held."""
        first = self.p.handle_tool_call(
            "mesh_remember", {"content": "the deploy key rotates every 90 days", "valid_from": JAN}
        )
        old_id = re.search(r"id (\w+)", first).group(1)
        self.p.handle_tool_call(
            "mesh_remember",
            {
                "content": "the deploy key rotates every 30 days",
                "valid_from": JUN,
                "supersedes": old_id,
            },
        )

        now = self.p.handle_tool_call(
            "mesh_recall", {"query": "how often does the deploy key rotate"}
        )
        self.assertIn("every 30 days", now)
        self.assertNotIn("every 90 days", now, "the superseded value must not read as current")

        then = self.p.handle_tool_call(
            "mesh_asof", {"query": "how often does the deploy key rotate", "as_of": "2026-03-01"}
        )
        self.assertIn("every 90 days", then, "history must survive the correction")

        # and after the correction took effect, the dated view holds only the new value:
        # before the window-closing fix this returned every value that had ever held.
        after = self.p.handle_tool_call(
            "mesh_asof", {"query": "how often does the deploy key rotate", "as_of": "2026-07-01"}
        )
        self.assertIn("every 30 days", after)
        self.assertNotIn("every 90 days", after, "the closed window must not leak into a later view")

    def test_mesh_asof_rejects_an_unparseable_date_without_crashing(self):
        self.p.handle_tool_call("mesh_remember", {"content": "the hackathon deadline is Nov 11"})
        out = self.p.handle_tool_call("mesh_asof", {"query": "deadline", "as_of": "not-a-date"})
        self.assertIn("parseable", out)
        self.assertNotIn("Traceback", out)

    def test_a_supersedes_that_matches_nothing_says_so(self):
        """A correction that replaces nothing must not look like it replaced something."""
        out = self.p.handle_tool_call(
            "mesh_remember",
            {"content": "the retry limit for service-3 is 11 attempts", "supersedes": "not-a-real-id"},
        )
        self.assertIn("matched no stored memory", out)

    # ── thread safety: the failure the live run exposed ─────────────────────────

    def test_a_write_from_a_worker_thread_lands_and_is_readable(self):
        """Hermes runs sync_turn on a background worker and tool calls on an executor thread.
        One sqlite handle cannot cross threads; before the per-thread accessor the write was
        silently lost and the provider looked like it was remembering nothing."""
        errors = []

        def worker():
            try:
                self.p.sync_turn(
                    "the vault code is 4471 and it must survive a thread hop",
                    "noted",
                    session_id="s",
                )
                self.p.handle_tool_call("mesh_remember", {"content": "written from a worker thread"})
            except Exception as exc:  # pragma: no cover - only on regression
                errors.append(repr(exc))

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        self.assertEqual(errors, [])
        self.assertIn("4471", self.p.handle_tool_call("mesh_recall", {"query": "vault code"}))
        self.assertIn(
            "worker thread", self.p.handle_tool_call("mesh_recall", {"query": "written from a worker"})
        )

    def test_concurrent_writers_from_several_threads_all_land(self):
        errors = []

        def writer(i):
            try:
                self.p.handle_tool_call("mesh_remember", {"content": f"fact number {i} from thread {i}"})
            except Exception as exc:  # pragma: no cover - only on regression
                errors.append(repr(exc))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(self.p._m().stats()["total"], 6)

    def test_register_hands_the_provider_to_the_context(self):
        seen = {}

        class Ctx:
            def register_memory_provider(self, provider):
                seen["p"] = provider

        nm.register(Ctx())
        self.assertIsInstance(seen["p"], nm.NeuralMeshProvider)
        self.assertEqual(seen["p"].name, "neural-mesh")

    def test_vendored_version_matches_the_plugin_manifest(self):
        """The vendored package reported 0.40.0 while plugin.yaml said 0.41.0 for a whole
        release, because nothing compared the two. A stale version string is how an
        operator ends up unable to say which build a failure came from."""
        manifest = (REPO / "plugin.yaml").read_text(encoding="utf-8")
        declared = re.search(r"^version:\s*(\S+)\s*$", manifest, re.M).group(1)
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        import neural_mesh as vendored  # the package as an operator would import it
        self.assertEqual(vendored.__version__, declared)


class RemoteSettingsFileTest(unittest.TestCase):
    def test_reads_url_and_token_from_home(self):
        import json, os, tempfile
        tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(tmp, "neural-mesh"), exist_ok=True)
        with open(os.path.join(tmp, "neural-mesh", "remote.json"), "w") as fh:
            json.dump({"url": "https://mesh.example.test", "token": "t0ken"}, fh)
        url, token = nm.remote_settings_file(tmp)
        self.assertEqual(url, "https://mesh.example.test")
        self.assertEqual(token, "t0ken")

    def test_missing_file_is_empty_not_an_error(self):
        import tempfile
        self.assertEqual(nm.remote_settings_file(tempfile.mkdtemp()), ("", ""))

    def test_settings_file_is_read_from_the_home_the_provider_was_given(self):
        # Regression: __init__ used to read $HERMES_HOME's remote.json whatever
        # hermes_home was passed to initialize(), so a "local" test provider silently
        # opened the real canonical store and wrote its fixtures into it.
        import json, tempfile
        from unittest import mock
        real = tempfile.mkdtemp(prefix="nm-realhome-")
        os.makedirs(os.path.join(real, "neural-mesh"), exist_ok=True)
        with open(os.path.join(real, "neural-mesh", "remote.json"), "w") as fh:
            json.dump({"url": "https://canonical.example", "token": "t"}, fh)
        env = {"HERMES_HOME": real}
        with mock.patch.dict(os.environ, env):
            os.environ.pop("NEURAL_MESH_REMOTE_URL", None)
            other = nm.NeuralMeshProvider()
            other.initialize("s", hermes_home=tempfile.mkdtemp(prefix="nm-other-"),
                             agent_context="primary")
            self.assertEqual(other._remote_url, "")
            other.shutdown()
            same = nm.NeuralMeshProvider()
            same.initialize("s", hermes_home=real, agent_context="cron")
            self.assertEqual(same._remote_url, "https://canonical.example")


if __name__ == "__main__":
    unittest.main(verbosity=2)
