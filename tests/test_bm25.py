"""Tests for the provider's BM25 ranking and its coverage floor."""

import unittest

import bm25


class Node:
    def __init__(self, nid, content):
        self.id = nid
        self.content = content


class FakeMesh:
    """Just enough of a Mesh for the index cache key."""

    def __init__(self):
        self.db = None


NODES = [
    Node("a", "the deploy key rotates every 90 days in the staging cluster"),
    Node("b", "square bullets are used for every X post and list"),
    Node("c", "the window cleaner comes on tuesday mornings"),
]


class TokenizeTest(unittest.TestCase):
    def test_stopwords_and_short_tokens_are_dropped(self):
        self.assertEqual(bm25.tokenize("What is the DEPLOY key?"), ["deploy", "key"])

    def test_paths_and_dots_survive(self):
        self.assertIn("mainnet.base.org", bm25.tokenize("rpc mainnet.base.org"))


class RankingTest(unittest.TestCase):
    def setUp(self):
        self.mesh = FakeMesh()
        self.nodes = NODES

    def test_relevant_node_ranks_first(self):
        hits = bm25.rank(self.mesh, "deploy key rotation", self.nodes, top_k=2)
        self.assertEqual(hits[0][1].id, "a")

    def test_plural_matches_singular(self):
        hits = bm25.rank(self.mesh, "square bullet preference", self.nodes, top_k=1)
        self.assertEqual(hits[0][1].id, "b")

    def test_unrelated_query_scores_zero(self):
        self.assertEqual(bm25.rank(self.mesh, "quantum chromodynamics", self.nodes, top_k=3), [])

    def test_coverage_gate_keeps_relevant_and_drops_unmatched(self):
        strong = bm25.rank(self.mesh, "deploy key rotates", self.nodes, top_k=3, min_coverage=0.6)
        weak = bm25.rank(self.mesh, "quantum chromodynamics", self.nodes, top_k=3, min_coverage=0.6)
        self.assertEqual([n.id for _, n in strong], ["a"])
        self.assertEqual(weak, [])

    def test_unseen_query_terms_dilute_coverage(self):
        # "rotation" is not in this store, so it is counted at max IDF: on a small
        # store that alone can push a correct hit under the floor, which is why the
        # provider drops the floor entirely below BM25_SMALL_STORE memories.
        hits = bm25.rank(self.mesh, "deploy key rotation", self.nodes, top_k=3, min_coverage=0.6)
        self.assertEqual(hits, [])
        unfiltered = bm25.rank(self.mesh, "deploy key rotation", self.nodes, top_k=1)
        self.assertEqual(unfiltered[0][1].id, "a")

    def test_top_k_is_respected(self):
        hits = bm25.rank(self.mesh, "the deploy key tuesday bullets", self.nodes, top_k=2)
        self.assertLessEqual(len(hits), 2)


if __name__ == "__main__":
    unittest.main()
