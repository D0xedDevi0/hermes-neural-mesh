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
        self.assertEqual(names, {"mesh_remember", "mesh_recall", "mesh_asof", "mesh_stats"})

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

    def test_stats_report_the_pure_python_backend(self):
        self.p.handle_tool_call("mesh_remember", {"content": "a fact that exists"})
        out = self.p.handle_tool_call("mesh_stats", {})
        self.assertIn("resonance_backend", out)
        self.assertIn("python", out)  # no rust module shipped; fallback is the working path

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
