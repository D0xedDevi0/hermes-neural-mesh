import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import bm25


class SlashCompoundTest(unittest.TestCase):
    """A path-style compound must be reachable by its own components.

    Regression: "Loopers/community-art" indexed as one atomic token, so a query for
    "community-art" scored coverage 0 and the injection floor dropped a memory the service
    had already returned at rank 12 of 300. Anything restored here must stay reachable."""

    def test_component_of_a_slash_compound_is_a_token(self):
        toks = bm25.tokenize("a Loopers/community-art specialist")
        self.assertIn("loopers/community-art", toks)
        self.assertIn("community-art", toks)

    def test_single_char_parts_are_dropped(self):
        self.assertNotIn("a", bm25.tokenize("x/a"))

    def test_hyphens_stay_atomic(self):
        self.assertNotIn("art", bm25.tokenize("community-art"))
        self.assertNotIn("community", bm25.tokenize("community-art"))

    def test_query_component_matches_the_document(self):
        doc = bm25.tokenize("built as a Loopers/community-art specialist")
        q = bm25.tokenize("community-art")
        self.assertTrue(set(q) & set(doc))


if __name__ == "__main__":
    unittest.main()
