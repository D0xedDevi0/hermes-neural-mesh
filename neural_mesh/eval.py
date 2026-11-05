"""LoCoMo QA evaluation — LLM judge scores mesh answers against ground truth.

Pure stdlib. Follows the injectable ``_post_fn`` pattern from ``LLMReader``
so tests never hit a real API.

Usage::

    from neural_mesh.eval import QAJudge, run_qa_eval
    judge = QAJudge()          # reads OPENROUTER_API_KEY from env
    metrics = run_qa_eval(mesh, test_set, judge=judge)
"""

from __future__ import annotations

import json
import statistics
from typing import Any

from .reader_llm import LLMReader


# ── test set helpers ────────────────────────────────────────────────────

QAExample = dict[str, str]  # {"query": ..., "gold": ...}


def load_test_set(path: str) -> list[QAExample]:
    """Load a JSON-lines test set. Each line: ``{"query": "...", "gold": "..."}``."""
    examples: list[QAExample] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            examples.append(json.loads(line))
    return examples


# ── judge prompt ────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are an impartial QA judge. Score the ANSWER against the GROUND_TRUTH.

Scoring rubric:
- 1.0 — answer is fully correct, captures all key facts
- 0.7-0.9 — mostly correct, minor omission or imprecision
- 0.4-0.6 — partially correct but missing significant facts
- 0.1-0.3 — mostly wrong or irrelevant
- 0.0 — completely wrong or hallucinated

Return ONLY valid JSON: {{"score": <float 0-1>, "reasoning": "<one sentence>"}}

QUESTION: {question}

GROUND_TRUTH: {gold}

ANSWER: {answer}

JSON:"""


# ── QA judge ────────────────────────────────────────────────────────────

class QAJudge:
    """LLM-powered QA judge — scores answers against ground truth.

    Uses the same ``_post_fn`` injectable pattern as ``LLMReader``
    so tests can supply a mock API response without hitting a real endpoint.
    """

    def __init__(self, model: str | None = None, *, _post_fn=None):
        self._llm = LLMReader(model=model, _post_fn=_post_fn)

    def score(self, question: str, answer: str, gold: str) -> dict[str, Any]:
        """Score a single answer. Returns ``{score, reasoning, question, answer, gold}``."""
        if not self._llm.api_key:
            return _simple_score(question, answer, gold)

        prompt = JUDGE_PROMPT.format(question=question, gold=gold, answer=answer)
        try:
            body = self._llm._call_api(prompt)
            content = (
                body.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            parsed = _parse_json_snippet(str(content))
            if parsed and "score" in parsed:
                return {
                    "score": max(0.0, min(1.0, float(parsed["score"]))),
                    "reasoning": str(parsed.get("reasoning", "")),
                    "question": question,
                    "answer": answer,
                    "gold": gold,
                }
        except Exception:
            pass
        return _simple_score(question, answer, gold)

    @property
    def api_key(self) -> str:
        return self._llm.api_key


def _simple_score(question: str, answer: str, gold: str) -> dict[str, Any]:
    """Fallback scorer when LLM judge is unavailable — keyword overlap."""
    if not answer or not gold:
        return {"score": 0.0, "reasoning": "empty answer or ground truth",
                "question": question, "answer": answer, "gold": gold}
    gold_words = set(gold.lower().split())
    answer_words = set(answer.lower().split())
    if not gold_words:
        return {"score": 0.0, "reasoning": "empty ground truth",
                "question": question, "answer": answer, "gold": gold}
    overlap = len(gold_words & answer_words) / len(gold_words)
    score = round(min(overlap * 1.5, 1.0), 3)
    return {"score": score, "reasoning": "simple keyword overlap (LLM judge unavailable)",
            "question": question, "answer": answer, "gold": gold}


# ── JSON parsing ────────────────────────────────────────────────────────

def _parse_json_snippet(text: str) -> dict | None:
    """Extract a JSON object from an LLM response that may have extra text."""
    import re as _re
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find a {...} block containing "score"
    m = _re.search(r'\{[^{}]*"score"[^{}]*\}', text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


# ── end-to-end evaluation ───────────────────────────────────────────────

def run_qa_eval(
    mesh,
    test_set: list[QAExample],
    *,
    judge: QAJudge | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """End-to-end QA evaluation: recall -> answer -> judge -> aggregate."""
    from .proof_cards import answer_with_proofs

    if judge is None:
        judge = QAJudge()

    scores: list[float] = []
    per_item: list[dict] = []

    for ex in test_set:
        query = ex["query"]
        gold = ex.get("gold", "")

        try:
            out = answer_with_proofs(mesh, query, top_k=top_k)
            answer = out.get("answer", "")
            if not answer:
                answer = "\n".join(
                    r["text"] for r in out.get("retrieved", [])
                )
        except Exception:
            answer = ""

        result = judge.score(query, answer, gold)
        scores.append(result["score"])
        per_item.append(result)

    if not scores:
        return {"total": 0, "scores": [], "mean": 0.0, "median": 0.0,
                "min": 0.0, "max": 0.0, "per_item": []}

    return {
        "total": len(scores),
        "scores": [round(s, 4) for s in scores],
        "mean": round(statistics.mean(scores), 4),
        "median": round(statistics.median(scores), 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "per_item": per_item,
    }


__all__ = ["QAJudge", "run_qa_eval", "load_test_set", "QAExample"]
