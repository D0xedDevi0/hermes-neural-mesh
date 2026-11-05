"""Real embedder adapters — drop into Mesh(embedder=RealEmbedder()).

Keeps the core pip-free; this module is the *optional* upgrade path.
Prefers fastembed (no torch, ONNX, ~tiny model) and falls back to
sentence-transformers if you already have it.
"""
from __future__ import annotations


class RealEmbedder:
    def __init__(self, backend: str = "fastembed", model: str | None = None):
        self.backend = backend
        self.model = model
        self._inst = None

    def _load(self):
        if self._inst is not None:
            return self._inst
        if self.backend == "fastembed":
            from fastembed import TextEmbedding
            name = self.model or "BAAI/bge-small-en-v1.5"
            self._inst = TextEmbedding(model_name=name)
        elif self.backend == "sentence_transformers":
            from sentence_transformers import SentenceTransformer
            name = self.model or "all-MiniLM-L6-v2"
            self._inst = SentenceTransformer(name)
        else:
            raise ValueError(f"unknown backend {self.backend}")
        return self._inst

    def __call__(self, text: str):
        inst = self._load()
        if self.backend == "fastembed":
            vec = list(inst.embed([text]))[0]
        else:
            vec = inst.encode(text)
        return tuple(float(x) for x in vec)

    def embed_many(self, texts: list[str]):
        """Batched embedding — orders of magnitude faster than per-call.

        Returns a list of tuples aligned to `texts`."""
        inst = self._load()
        if self.backend == "fastembed":
            vecs = list(inst.embed(texts))
        else:
            vecs = inst.encode(texts)
        return [tuple(float(x) for x in v) for v in vecs]
