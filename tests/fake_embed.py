"""Deterministic fake embedder for the fast suite (no Ollama).

Hashed bag-of-words into a small fixed-dim vector: texts sharing words get
higher cosine similarity, so ranking assertions are meaningful, and the
same text always embeds identically (hashlib, not the randomized builtin
`hash`). Installed via `core.embeddings.set_embed_fn`.
"""

from __future__ import annotations

import hashlib
import re

DIM = 64


def fake_embed(texts: list[str]) -> list[list[float]]:
    out = []
    for text in texts:
        vec = [0.0] * DIM
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            idx = int(hashlib.sha1(word.encode()).hexdigest(), 16) % DIM
            vec[idx] += 1.0
        out.append(vec)
    return out


def broken_embed(texts: list[str]) -> list[list[float]]:
    """Simulates Ollama being down."""
    from core.embeddings import EmbeddingUnavailable

    raise EmbeddingUnavailable("simulated outage")
