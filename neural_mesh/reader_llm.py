"""LLM-powered reader — synthesizes answers from retrieved mesh passages.

Follows the proven OpenRouter pattern from ``muse.py`` (pure stdlib ``urllib``).
Callers can inject ``_post_fn`` for testing without hitting a real API.

Usage::

    from neural_mesh.reader_llm import LLMReader
    reader = LLMReader()  # reads OPENROUTER_API_KEY from env
    answer = reader.answer("query", ["passage one", "passage two"])
"""

from __future__ import annotations

import json
import os

from .reader import Reader


class LLMReader(Reader):
    """Drop-in Reader that calls an LLM via OpenRouter-compatible API.

    Falls back to extractive (first passage) when no API key is configured,
    passages are empty, or the API call fails — so the proof-backed answer
    endpoint always returns *something* useful.
    """

    # ── public API ──────────────────────────────────────────────────────

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        _post_fn=None,
    ):
        self.model = model or os.environ.get(
            "NEURAL_MESH_LLM", "deepseek/deepseek-v4-flash"
        )
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        self.base_url = base_url or os.environ.get(
            "OPENROUTER_BASE", "https://openrouter.ai/api/v1"
        )
        self._post = _post_fn  # injectable for testing

    def answer(self, query: str, passages: list[str], gold: str = "") -> str:
        """Synthesize an answer from retrieved passages, or fall back to first passage."""
        if not self.api_key or not passages:
            return self._fallback(query, passages)

        prompt = self._build_prompt(query, passages)
        try:
            body = self._call_api(prompt)
            answer = self._extract_answer(body)
            return answer or self._fallback(query, passages)
        except Exception:
            return self._fallback(query, passages)

    # ── internal helpers ────────────────────────────────────────────────

    def _build_prompt(self, query: str, passages: list[str]) -> str:
        """Construct a QA prompt with retrieved context passages."""
        context_blocks = []
        for i, p in enumerate(passages[:10], 1):  # cap at 10 for token budget
            context_blocks.append(f"[{i}] {p[:500]}")
        context = "\n\n".join(context_blocks)

        return (
            "You are NEURAL_MESH's answer engine. Answer the question concisely "
            "using ONLY the retrieved context passages below. If the context "
            "does not contain enough information, say so honestly.\n\n"
            "Always start your answer with 'ANSWER: '.\n\n"
            f"QUESTION: {query}\n\n"
            f"CONTEXT:\n{context}\n\n"
            "ANSWER:"
        )

    def _call_api(self, prompt: str) -> dict:
        """Post to the LLM API. Override ``_post_fn`` for testing."""
        if self._post is not None:
            import urllib.request as _urllib  # for type reference only
            return self._post(prompt)  # type: ignore[arg-type]

        import urllib.request

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps({
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
                "temperature": 0.3,
            }).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def _extract_answer(self, body: dict) -> str:
        """Pull the answer string from an API response body.

        Handles the ``choices[0].message.content`` shape (OpenAI/OpenRouter)
        and strips the ``ANSWER: `` prefix if present.
        """
        content = (
            body.get("choices", [{}])[0]
            .get("message", {})
            .get("content")
        )
        if not content:
            return ""
        text = str(content).strip()
        # Strip optional "ANSWER:" prefix
        for prefix in ("ANSWER: ", "ANSWER:", "Answer: ", "Answer:"):
            if text.startswith(prefix):
                return text[len(prefix):].strip()
        return text

    def _fallback(self, query: str, passages: list[str]) -> str:
        """Return the most relevant passage when LLM is unavailable."""
        return passages[0] if passages else ""


__all__ = ["LLMReader"]
