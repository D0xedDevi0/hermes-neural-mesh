"""Zero-dependency text embedding + similarity.

This is a deliberately simple, deterministic hashed bag-of-words vector so the
core runs with NO pip installs. Swap in a real embedder (sentence-transformers,
OpenAI, etc.) by passing `embedder=your_fn` to `Mesh` — it only needs to return
an indexable sequence of floats.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

_STOPS = set(
    "the a an and or of to in on for with is are was were be been being it this "
    "that these those as at by from up out if then than so but not no".split()
)


def tokenize(text: str):
    return [w for w in re.findall(r"[a-z0-9_]+", text.lower()) if w not in _STOPS]


def embed(text: str, dims: int = 256) -> tuple[float, ...]:
    vec = [0.0] * dims
    for tok in tokenize(text):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % dims
        vec[h] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm:
        vec = [v / norm for v in vec]
    return tuple(vec)


def cosine(a, b) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))
