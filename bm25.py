"""BM25 ranking for the local store, standard library only.

Why: the vendored core ranks with a 256-dimension md5 bag-of-tokens. On the
1,638 records migrated from YantrikDB that put the right record in the top 5 for
11 of 20 probes, while BM25 over the same nodes scored 19 of 20.

The core does ship an Okapi index, but it ranks positions inside a corpus list,
and the node list it hands out is not guaranteed to be in the same order, so a
caller cannot safely map scores back to nodes. This module ranks node objects by
their own id instead, which removes that whole class of bug.

Scores are raw BM25 and grow with corpus size, so the exported helper returns a
coverage ratio (matched share of the query's IDF, unseen terms at max IDF) for
callers that need a floor which works at any store size. Matching is
plural-tolerant: "bullet" matches "bullets", and the original term's IDF is used
for both sides, so variants never dilute the ratio.
"""

import math
import re

__all__ = ["tokenize", "rank", "coverage"]

_TOKEN = re.compile(r"[a-z0-9_./:-]{2,}")

_STOPS = frozenset(
    """a an the is are was were be been being am to of in on for with and or
    this that these those it its as at by from into your you we our not but
    if then than so do does did doing have has had what which who how where
    when why can could would should will shall may might must i me my about
    there here any some""".split()
)

K1 = 1.2
B = 0.75


def tokenize(text):
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in _STOPS]


def _variants(term):
    out = [term]
    if len(term) > 3:
        if term.endswith("s"):
            out.append(term[:-1])
        else:
            out.append(term + "s")
    return out


class _Index:
    def __init__(self, docs):
        self.ids = []
        self.counts = []
        self.lens = []
        self.df = {}
        for node_id, text in docs:
            toks = tokenize(text)
            if not toks:
                continue
            c = {}
            for t in toks:
                c[t] = c.get(t, 0) + 1
            self.ids.append(node_id)
            self.counts.append(c)
            self.lens.append(len(toks))
            for t in c:
                self.df[t] = self.df.get(t, 0) + 1
        self.n = len(self.ids)
        self.avgdl = (sum(self.lens) / self.n) if self.n else 1.0

    def idf(self, term):
        d = self.df.get(term) or 0
        if not d:
            return math.log(1.0 + (self.n + 0.5) / 0.5)
        return math.log(1.0 + (self.n - d + 0.5) / (d + 0.5))

    def score(self, terms):
        weights = [(t, self.idf(t)) for t in dict.fromkeys(terms)]
        out = []
        for i, counts in enumerate(self.counts):
            norm = K1 * (1.0 - B + B * self.lens[i] / self.avgdl)
            s = 0.0
            for term, w in weights:
                tf = 0
                for v in _variants(term):
                    tf += counts.get(v, 0)
                if tf:
                    s += w * (tf * (K1 + 1.0)) / (tf + norm)
            if s > 0.0:
                out.append((s, self.ids[i]))
        out.sort(reverse=True)
        return out

    def total_idf(self, terms):
        return sum(self.idf(t) for t in dict.fromkeys(terms))


_CACHE = {}
_CACHE_MAX = 4


def _index_for(mesh, nodes):
    db = getattr(mesh, "db", None)
    if db is None:
        # No store to key on (a remote handle, for instance). Caching on id(nodes)
        # is wrong: CPython recycles list ids, so a later candidate set of the same
        # length can be handed a stale index and rank against the wrong documents.
        return _Index([(getattr(n, "id", ""), getattr(n, "content", "")) for n in nodes])
    key = None
    try:
        key = (db.execute("PRAGMA database_list").fetchone()[2], len(nodes))
    except Exception:
        key = (id(nodes), len(nodes))
    hit = _CACHE.get(key)
    if hit is not None and hit[0] == len(nodes):
        return hit[1]
    idx = _Index([(getattr(n, "id", ""), getattr(n, "content", "")) for n in nodes])
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.clear()
    _CACHE[key] = (len(nodes), idx)
    return idx


def rank(mesh, query, nodes, top_k=5, min_coverage=0.0):
    """``[(coverage, node)]`` best first. Coverage is score / total query IDF."""
    terms = tokenize(query)
    if not terms or not nodes:
        return []
    idx = _index_for(mesh, nodes)
    total = idx.total_idf(terms)
    if not total:
        return []
    by_id = {}
    for n in nodes:
        by_id.setdefault(getattr(n, "id", ""), n)
    out = []
    for s, nid in idx.score(terms)[: max(1, int(top_k or 1))]:
        cov = s / total
        if cov >= min_coverage:
            node = by_id.get(nid)
            if node is not None:
                out.append((cov, node))
    return out


def coverage(mesh, query, nodes):
    """Best coverage in the store for ``query``; 0.0 when nothing matches."""
    hits = rank(mesh, query, nodes, top_k=1)
    return hits[0][0] if hits else 0.0
