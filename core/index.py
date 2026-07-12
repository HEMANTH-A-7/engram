"""Owned incremental search index over SQLite — the serving path.

Two retrieval channels over the same rows, maintained together on every
add/remove (engram-lite, Session 10):

- **FTS5 keyword index** (stdlib sqlite3, BM25). Always available: no
  Ollama, no extra process, ~ms queries. This is the floor — the layer is
  fully functional with nothing but Python installed.
- **Vector index** (embeddings via one direct Ollama call, float32 blobs,
  brute-force cosine). Optional upgrade: when the embed endpoint answers,
  `search()` fuses both channels with reciprocal-rank fusion; when it
  doesn't, results are keyword-ranked and the reported `mode` says so.
  A cooldown circuit-breaker in core/embeddings.py keeps the no-Ollama
  configuration overhead-free.

Replaces the old `store.reset_and_load` design (full wipe + re-cognify of
Cognee's ENTIRE data root on every write): rows are added/removed
individually, `dataset` scoping makes per-project isolation free, and the
"superseded text is never searchable" invariant is enforced by
`remove_item()` at write time. Cognee is not imported here — it remains the
benchmark/demo engine only.

`search()` NEVER raises for infrastructure reasons: embed endpoint down →
keyword mode; index storage itself unreachable → empty `mode="unavailable"`
result. Writes always land: a vector that can't be computed is stored as
pending and filled in opportunistically later.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from core import embeddings, resolver
from core.embeddings import EmbeddingUnavailable  # re-export for callers

__all__ = [
    "EmbeddingUnavailable",
    "Hit",
    "SearchResult",
    "index_item",
    "remove_item",
    "remove_dataset",
    "embed_pending",
    "pending_count",
    "indexed_count",
    "backfill_facts",
    "search",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset TEXT NOT NULL,
    kind TEXT NOT NULL,            -- 'fact' | 'note'
    ref_id INTEGER NOT NULL,       -- facts.id or notes.id
    text TEXT NOT NULL,
    vector BLOB,                   -- NULL = pending (embed endpoint was down)
    dim INTEGER,
    created_at TEXT NOT NULL,
    UNIQUE (dataset, kind, ref_id)
);
CREATE INDEX IF NOT EXISTS idx_embeddings_dataset ON embeddings (dataset);
CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
    text, dataset UNINDEXED, kind UNINDEXED, ref_id UNINDEXED
);
"""

# Reciprocal-rank-fusion constant (standard default). Fusion by rank, not by
# raw score, sidesteps "cosine and BM25 live on incomparable scales".
_RRF_K = 60


@dataclass(frozen=True)
class Hit:
    kind: str  # 'fact' | 'note'
    ref_id: int
    text: str
    score: float


@dataclass(frozen=True)
class SearchResult:
    hits: list[Hit]
    # 'hybrid' (keyword + semantic) | 'keyword' (FTS5 only) |
    # 'unavailable' (index storage unreachable — empty result, never an exception)
    mode: str


def _connect() -> sqlite3.Connection:
    # Shares the resolver's DB file so `resolver.configure(tmp)` in tests
    # (and MEMORY_DB_PATH in hook-script tests) redirects this module too.
    conn = sqlite3.connect(resolver.db_path())
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fts_replace(conn: sqlite3.Connection, dataset: str, kind: str, ref_id: int, text: str) -> None:
    conn.execute(
        "DELETE FROM items_fts WHERE dataset = ? AND kind = ? AND ref_id = ?",
        (dataset, kind, ref_id),
    )
    conn.execute(
        "INSERT INTO items_fts (text, dataset, kind, ref_id) VALUES (?, ?, ?, ?)",
        (text, dataset, kind, ref_id),
    )


def index_item(dataset: str, kind: str, ref_id: int, text: str) -> bool:
    """Index one item incrementally in BOTH channels. Returns True if the
    vector was embedded now, False if it was stored as pending (endpoint
    down) — the FTS5 row lands either way, so the item is immediately
    findable by keyword regardless of Ollama.

    Upsert semantics on (dataset, kind, ref_id): re-indexing replaces.
    """
    vector: bytes | None = None
    dim: int | None = None
    try:
        vec = embeddings.embed_texts([text])[0]
        vector, dim = embeddings.pack_vector(vec), len(vec)
    except EmbeddingUnavailable:
        pass  # pending; embed_pending() picks it up later
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO embeddings (dataset, kind, ref_id, text, vector, dim, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (dataset, kind, ref_id)
            DO UPDATE SET text = excluded.text, vector = excluded.vector,
                          dim = excluded.dim
            """,
            (dataset, kind, ref_id, text, vector, dim, _now()),
        )
        _fts_replace(conn, dataset, kind, ref_id, text)
        conn.commit()
    return vector is not None


def remove_item(dataset: str, kind: str, ref_id: int) -> None:
    """Drop one item from both channels (supersede/evict path). Idempotent."""
    with _connect() as conn:
        conn.execute(
            "DELETE FROM embeddings WHERE dataset = ? AND kind = ? AND ref_id = ?",
            (dataset, kind, ref_id),
        )
        conn.execute(
            "DELETE FROM items_fts WHERE dataset = ? AND kind = ? AND ref_id = ?",
            (dataset, kind, ref_id),
        )
        conn.commit()


def remove_dataset(dataset: str) -> int:
    """Drop every index row for ONE dataset (tests/cleanup). Scoped on purpose:
    unlike the old `store.reset_and_load`, nothing here can touch another
    project's index. Returns embedding rows removed."""
    with _connect() as conn:
        cur = conn.execute("DELETE FROM embeddings WHERE dataset = ?", (dataset,))
        conn.execute("DELETE FROM items_fts WHERE dataset = ?", (dataset,))
        conn.commit()
        return cur.rowcount


def pending_count(dataset: str | None = None) -> int:
    q = "SELECT COUNT(*) FROM embeddings WHERE vector IS NULL"
    args: tuple = ()
    if dataset is not None:
        q += " AND dataset = ?"
        args = (dataset,)
    with _connect() as conn:
        return conn.execute(q, args).fetchone()[0]


def indexed_count(dataset: str) -> int:
    with _connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE dataset = ?", (dataset,)
        ).fetchone()[0]


def embed_pending(dataset: str | None = None) -> int:
    """Embed rows stored while the endpoint was down. Returns how many were
    filled in. Never raises: if the endpoint is still down (or in cooldown),
    they simply stay pending."""
    with _connect() as conn:
        q = "SELECT id, text FROM embeddings WHERE vector IS NULL"
        args: tuple = ()
        if dataset is not None:
            q += " AND dataset = ?"
            args = (dataset,)
        rows = conn.execute(q, args).fetchall()
    if not rows:
        return 0
    try:
        vecs = embeddings.embed_texts([r["text"] for r in rows])
    except EmbeddingUnavailable:
        return 0
    with _connect() as conn:
        for row, vec in zip(rows, vecs):
            conn.execute(
                "UPDATE embeddings SET vector = ?, dim = ? WHERE id = ?",
                (embeddings.pack_vector(vec), len(vec), row["id"]),
            )
        conn.commit()
    return len(rows)


def _ensure_fts_synced(dataset: str) -> None:
    """Backfill FTS5 rows for items indexed before the FTS channel existed
    (Session 9 data). Cheap count comparison; rebuilds only on mismatch."""
    with _connect() as conn:
        n_emb = conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE dataset = ?", (dataset,)
        ).fetchone()[0]
        n_fts = conn.execute(
            "SELECT COUNT(*) FROM items_fts WHERE dataset = ?", (dataset,)
        ).fetchone()[0]
        if n_emb == n_fts:
            return
        conn.execute("DELETE FROM items_fts WHERE dataset = ?", (dataset,))
        rows = conn.execute(
            "SELECT kind, ref_id, text FROM embeddings WHERE dataset = ?", (dataset,)
        ).fetchall()
        for r in rows:
            conn.execute(
                "INSERT INTO items_fts (text, dataset, kind, ref_id) VALUES (?, ?, ?, ?)",
                (r["text"], dataset, r["kind"], r["ref_id"]),
            )
        conn.commit()


def backfill_facts(dataset: str) -> int:
    """Index any current resolver facts that predate this index (migration
    path for datasets populated under the old Cognee-rebuild design, e.g.
    the real `main_dataset`). Idempotent; returns how many were added."""
    with _connect() as conn:
        have = {
            r["ref_id"]
            for r in conn.execute(
                "SELECT ref_id FROM embeddings WHERE dataset = ? AND kind = 'fact'",
                (dataset,),
            )
        }
    added = 0
    for fact in resolver.current_facts(dataset):
        if fact.id not in have:
            index_item(dataset, "fact", fact.id, fact.text)
            added += 1
    _ensure_fts_synced(dataset)
    return added


def _fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 OR-query. FTS5 has its own operator
    syntax that chokes on quotes/hyphens/parens in natural language, so we
    reduce the query to bare word tokens. OR (not AND) for recall — BM25
    still ranks fuller matches higher."""
    tokens = re.findall(r"\w+", query.lower())
    return " OR ".join(f'"{t}"' for t in tokens)


def _keyword_search(dataset: str, query: str, k: int) -> list[Hit]:
    fts_q = _fts_query(query)
    if not fts_q:
        return []
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT kind, ref_id, text, bm25(items_fts) AS rank
            FROM items_fts WHERE items_fts MATCH ? AND dataset = ?
            ORDER BY rank LIMIT ?
            """,
            (fts_q, dataset, k),
        ).fetchall()
    # bm25() is "lower = better"; expose a "higher = better" score.
    return [
        Hit(kind=r["kind"], ref_id=int(r["ref_id"]), text=r["text"], score=-r["rank"])
        for r in rows
    ]


def _semantic_search(dataset: str, query: str, k: int) -> list[Hit]:
    """Raises EmbeddingUnavailable if the query can't be embedded."""
    embed_pending(dataset)
    qvec = embeddings.embed_texts([query])[0]
    with _connect() as conn:
        rows = conn.execute(
            "SELECT kind, ref_id, text, vector FROM embeddings "
            "WHERE dataset = ? AND vector IS NOT NULL",
            (dataset,),
        ).fetchall()
    scored = [
        Hit(
            kind=r["kind"],
            ref_id=r["ref_id"],
            text=r["text"],
            score=embeddings.cosine(qvec, embeddings.unpack_vector(r["vector"])),
        )
        for r in rows
    ]
    scored.sort(key=lambda h: h.score, reverse=True)
    return scored[:k]


def search(dataset: str, query: str, k: int = 5) -> SearchResult:
    """Search one dataset. Never raises for infrastructure reasons.

    Keyword (FTS5/BM25) always runs. If the embed endpoint answers, semantic
    results are fused in via reciprocal-rank fusion (`mode="hybrid"`);
    otherwise results are keyword-only (`mode="keyword"`) — reported, never
    silently swapped. Pending vectors are embedded opportunistically first,
    so items written during an Ollama outage upgrade to semantic reach the
    moment it's back.
    """
    k = max(1, k)
    try:
        _ensure_fts_synced(dataset)
        kw = _keyword_search(dataset, query, k)
    except sqlite3.Error:
        # The DB file itself is unopenable/corrupt (seen: sqlite over a
        # read-only mount). A search tool must degrade, not crash the caller:
        # empty result, mode says why.
        return SearchResult(hits=[], mode="unavailable")
    try:
        sem = _semantic_search(dataset, query, k)
    except (EmbeddingUnavailable, sqlite3.Error):
        return SearchResult(hits=kw[:k], mode="keyword")

    # Reciprocal-rank fusion across the two ranked lists.
    fused: dict[tuple[str, int], float] = {}
    texts: dict[tuple[str, int], str] = {}
    for hits in (kw, sem):
        for rank, hit in enumerate(hits, start=1):
            key = (hit.kind, hit.ref_id)
            fused[key] = fused.get(key, 0.0) + 1.0 / (_RRF_K + rank)
            texts[key] = hit.text
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return SearchResult(
        hits=[
            Hit(kind=key[0], ref_id=key[1], text=texts[key], score=round(score, 6))
            for key, score in ranked
        ],
        mode="hybrid",
    )
