"""Okapi BM25 full-text lexical scoring — pure-stdlib reference.

This is the *proper* lexical retrieval scorer: term-frequency × inverse
document frequency with length normalization. It complements the hashed
bag-of-words cosine in `embed.py` — that scorer is a token-overlap measure,
while this ranks documents by classic information-retrieval relevance.

`okapi_bm25` / `bm25_score` / `_idf` here are the reference implementation.
`rust_mesh.bulk_bm25` / `rust_mesh.bm25_score` / `rust_mesh.bm25_idf` and the
persistent `rust_mesh.Bm25Index` are the accelerated (abi3, no-deps) twins.
They MUST stay formula-identical — the parity suite in
`tests/test_bm25_parity.py` enforces it. Tokenization stays in Python
(`embed.tokenize`) so there is a single tokenizer; the Rust hot path only ever
receives token lists.
"""
from __future__ import annotations

import math
from collections import Counter

from .embed import tokenize

# Okapi BM25 constants.
K1 = 1.5
B = 0.75


def _idf(n: int, df: int) -> float:
    """Okapi BM25 inverse document frequency with +0.5 smoothing."""
    return math.log(1.0 + (n - df + 0.5) / (df + 0.5))


def bm25_score(doc: list[str], query: list[str], df: dict[str, int],
               n_docs: int, avgdl: float, k1: float = K1, b: float = B) -> float:
    """Score one tokenized doc against a tokenized query.

    Sums over DISTINCT query terms present in `doc` (the classic Okapi form
    ignores query-side term frequency). `df` maps term -> number of docs in the
    corpus containing it. Identical to `rust_mesh.bm25_score`.
    """
    if not doc or not query:
        return 0.0
    tf = Counter(doc)
    doclen = len(doc)
    denom_norm = 1.0 - b + b * (doclen / avgdl) if avgdl > 0.0 else 1.0
    score = 0.0
    for t in set(query):
        f = tf.get(t, 0)
        if f == 0:
            continue
        score += _idf(n_docs, df.get(t, 0)) * f * (k1 + 1.0) / (f + k1 * denom_norm)
    return score


def _corpus_stats(corpus: list[list[str]]) -> tuple[dict[str, int], int, float]:
    """Return (df, n_docs, avgdl) for a tokenized corpus."""
    n = len(corpus)
    df: dict[str, int] = {}
    total_len = 0
    for doc in corpus:
        total_len += len(doc)
        for t in set(doc):
            df[t] = df.get(t, 0) + 1
    avgdl = (total_len / n) if n else 1.0
    return df, n, avgdl


def okapi_bm25(corpus: list[list[str]], query: list[str],
               k1: float = K1, b: float = B) -> list[float]:
    """Score every doc in a tokenized corpus against a tokenized query.

    One-shot: rebuilds document frequencies from scratch on every call.
    """
    n = len(corpus)
    if n == 0:
        return []
    df, n, avgdl = _corpus_stats(corpus)
    return [bm25_score(d, query, df, n, avgdl, k1, b) for d in corpus]


def rust_bm25_available() -> bool:
    """True when the abi3 extension exposes the BM25 API."""
    try:
        import rust_mesh
        return callable(getattr(rust_mesh, "bulk_bm25", None))
    except ImportError:
        return False


class BM25Index:
    """Pre-tokenized corpus BM25 index, used by `Mesh.bm25_recall` and the
    `lexical_backend=\"bm25\"` selector.

    The corpus is tokenized once and, when the Rust extension is present, kept
    in a persistent `rust_mesh.Bm25Index` so per-query scoring does NOT pay the
    PyO3 corpus-conversion cost. Pure-stdlib fallback reuses precomputed df /
    avgdl (no rebuild per query).
    """

    def __init__(self, docs: list[str], k1: float = K1, b: float = B):
        self.docs = docs
        self.corpus = [tokenize(d) for d in docs]
        self.k1 = k1
        self.b = b
        self._df, self._n, self._avgdl = _corpus_stats(self.corpus)
        self._rust = None
        if rust_bm25_available():
            import rust_mesh
            if hasattr(rust_mesh, "Bm25Index"):
                self._rust = rust_mesh.Bm25Index(self.corpus, k1, b)

    @property
    def backend(self) -> str:
        return "rust" if self._rust is not None else "python"

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        if self._rust is not None:
            return list(self._rust.score(q))
        df, n, avgdl = self._df, self._n, self._avgdl
        return [bm25_score(d, q, df, n, avgdl, self.k1, self.b) for d in self.corpus]
