"""Reader interface — turning retrieved context into an *answer*.

The QA harness needs a Reader that, given a query + retrieved passages, returns
a string answer. Two implementations ship:

  * ``ExtractiveReader`` — the model-free default. Picks the retrieved sentence
    with the best (oracle) token-F1 against the gold answer. Honest caveat: it
    is a *reader proxy* for benchmarking retrieval, NOT a generative model.
  * ``CallableReader`` — wraps any ``callable(query, context) -> str`` so a real
    local LLM (llama.cpp / vLLM / Ollama) can be dropped in with one line:
        reader = CallableReader(lambda q, c: my_llm(q, c))
    This is the swap-point: when a local generative model is available, generated
    answers become real and the harness stops being a proxy.

No network calls. The Reader never leaves the process.
"""
from __future__ import annotations

import re
from collections import Counter


def _tok(s: str):
    return re.findall(r"\w+", (s or "").lower())


def _tok_f1(pred: str, gold: str) -> float:
    """SQuAD-style token-level F1 between a predicted and gold answer."""
    gp, pp = _tok(gold), _tok(pred)
    if not gp and not pp:
        return 1.0
    if not gp or not pp:
        return 0.0
    gc, pc = Counter(gp), Counter(pp)
    inter = sum((gc & pc).values())
    if inter == 0:
        return 0.0
    p = inter / sum(pc.values())
    r = inter / sum(gc.values())
    return 2 * p * r / (p + r)


class Reader:
    """Base reader. Subclasses implement ``answer``."""

    def answer(self, query: str, passages: list[str]) -> str:
        raise NotImplementedError

    # oracle helpers for benchmarking (used by ExtractiveReader)
    @staticmethod
    def best_f1_sentence(sentences, gold):
        best, best_f = "", -1.0
        for s in sentences:
            f = _tok_f1(s, gold)
            if f > best_f:
                best, best_f = s, f
        return best


class ExtractiveReader(Reader):
    """Model-free reader: returns the retrieved sentence with highest oracle
    token-F1 vs the gold. *Not* a generative model — a benchmarking proxy."""

    def answer(self, query: str, passages: list[str], gold: str = "") -> str:
        if not passages:
            return ""
        # If we have gold (benchmark mode), pick best-F1 sentence. Otherwise
        # fall back to the most relevant-looking passage (first).
        if gold:
            return self.best_f1_sentence(passages, gold)
        return passages[0]


class CallableReader(Reader):
    """Drop-in wrapper for a real local LLM.

    ``fn`` signature: ``fn(query: str, context: str) -> str`` where ``context``
    is the joined retrieved passages. Set ``join="\n\n"`` to control formatting.
    """

    def __init__(self, fn, join: str = "\n\n"):
        self.fn = fn
        self.join = join

    def answer(self, query: str, passages: list[str], gold: str = "") -> str:
        return str(self.fn(query, self.join.join(passages))).strip()
