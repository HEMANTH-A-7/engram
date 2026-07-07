"""Direct, lightweight embedding client for the owned incremental index.

One HTTP call to Ollama's native embed endpoint (`/api/embed`) per batch —
no Cognee import, no LLM, stdlib only. This is what makes `memory_note`
sub-second: the entire write path is one embed call (~0.1–0.3s locally)
plus SQLite inserts, versus the old full `reset_and_load` cognify pipeline
(~3.2s best case, minutes at batch scale).

Deliberately import-light: the Claude Code hook scripts (scripts/hooks/)
import this at every session start/stop, and importing Cognee costs seconds
of interpreter time the hooks can't afford.

`set_embed_fn()` is the test seam: fast tests inject a deterministic fake so
the whole index/notes/context stack is unit-testable offline with no Ollama.

EMBEDDINGS ARE OPTIONAL (engram-lite, Session 10). If Ollama is unreachable,
`embed_texts` raises `EmbeddingUnavailable` and a cooldown circuit-breaker
makes subsequent calls fail *instantly* for a short window — so a machine
with no Ollama at all pays one fast connection-refused per cooldown period,
not a timeout per search. Callers degrade honestly: writes store a NULL
vector ("pending", embedded later), search falls back to FTS5 keyword
ranking (core/index.py).
"""

from __future__ import annotations

import json
import math
import struct
import time
import urllib.error
import urllib.request
from typing import Callable

from core.config import EMBEDDING, bare_model

# Seconds. Local Ollama answers embeds in well under a second when up; a
# short timeout keeps the "Ollama is down" failure fast for hook scripts.
EMBED_TIMEOUT = 10.0

# After a failure, skip the endpoint entirely for this long. Keeps the
# no-Ollama configuration overhead-free: one failed connect per cooldown
# window instead of one per operation.
FAILURE_COOLDOWN_S = 60.0

_last_failure: float | None = None


class EmbeddingUnavailable(RuntimeError):
    """Raised when the embedding endpoint can't be reached or errors out."""


def _embed_url() -> str:
    """Normalize any configured endpoint to Ollama's native `/api/embed`."""
    base = EMBEDDING.endpoint.rstrip("/")
    for suffix in ("/api/embeddings", "/api/embed", "/api", "/v1"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return f"{base}/api/embed"


def _http_embed(texts: list[str]) -> list[list[float]]:
    payload = json.dumps(
        {"model": bare_model(EMBEDDING.model), "input": texts}
    ).encode("utf-8")
    req = urllib.request.Request(
        _embed_url(), data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=EMBED_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
        raise EmbeddingUnavailable(f"embed endpoint unreachable: {exc}") from exc
    vectors = body.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise EmbeddingUnavailable(f"unexpected embed response shape: {body!r:.200}")
    return vectors


# Test seam: swap in a deterministic fake with set_embed_fn().
_embed_fn: Callable[[list[str]], list[list[float]]] = _http_embed


def set_embed_fn(fn: Callable[[list[str]], list[list[float]]] | None) -> None:
    """Override (or with None, restore) the embedding backend. Tests only.
    Also resets the failure circuit-breaker so tests can flip between a
    broken and a working backend deterministically."""
    global _embed_fn, _last_failure
    _embed_fn = fn or _http_embed
    _last_failure = None


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Raises EmbeddingUnavailable on any failure —
    instantly, without touching the endpoint, while the cooldown from a
    recent failure is active."""
    global _last_failure
    if not texts:
        return []
    if _last_failure is not None and (time.monotonic() - _last_failure) < FAILURE_COOLDOWN_S:
        raise EmbeddingUnavailable("embed endpoint in failure cooldown")
    try:
        result = _embed_fn(texts)
    except EmbeddingUnavailable:
        _last_failure = time.monotonic()
        raise
    _last_failure = None
    return result


def cosine(a: list[float], b: list[float]) -> float:
    """Plain-Python cosine similarity. At this project's scale (thousands of
    rows, 768 dims) brute force is a few ms — not worth a numpy dependency
    in the import-light path."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def pack_vector(vec: list[float]) -> bytes:
    """float32 little-endian blob for SQLite storage."""
    return struct.pack(f"<{len(vec)}f", *vec)


def unpack_vector(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))
