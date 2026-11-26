import importlib.util
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("probe_audit", os.path.join(ROOT, "tools", "probe_audit.py"))
pa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pa)


class ClassifyTest(unittest.TestCase):
    """A failing probe is 'fail' however rare its terms; a passing probe whose terms crowd
    the store is 'fragile', not a pass. These verdicts gate CI, so they are pinned: a probe
    set quietly weakened must not read as coverage."""

    def test_failing_probe_is_fail(self):
        self.assertEqual(pa.classify(False, {"turns": 5}, 20), "fail")

    def test_passing_but_crowded_is_fragile(self):
        self.assertEqual(pa.classify(True, {"high": 69}, 20), "fragile")

    def test_passing_and_rare_is_ok(self):
        self.assertEqual(pa.classify(True, {"smkts17atneyabh1atw": 4}, 20), "ok")

    def test_no_terms_is_ok(self):
        self.assertEqual(pa.classify(True, {}, 20), "ok")


if __name__ == "__main__":
    unittest.main()
