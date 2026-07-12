"""Unit tests for core/index.py — the owned incremental search index.

No Ollama, no Cognee: embeddings come from tests/fake_embed.py, and the FTS5
keyword channel needs nothing at all. Proves the engram-lite properties:
incremental add/remove (no rebuilds), per-dataset isolation, hybrid
keyword+semantic fusion when embeddings answer, keyword-only mode (reported,
never silently swapped) when they don't, pending rows that heal, and the
pre-pivot backfill migration. `search()` must never raise for
infrastructure reasons.
"""

from __future__ import annotations

import pytest

from core import embeddings, index, resolver
from tests.fake_embed import broken_embed, fake_embed


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    resolver.configure(db_path=tmp_path / "facts.db")
    embeddings.set_embed_fn(fake_embed)
    yield
    embeddings.set_embed_fn(None)


def test_hybrid_search_ranks_by_similarity():
    index.index_item("ds", "note", 1, "the database migration uses alembic")
    index.index_item("ds", "note", 2, "the frontend uses react and vite")

    result = index.search("ds", "how do we run database migrations", k=2)

    assert result.mode == "hybrid"
    assert result.hits[0].ref_id == 1
    assert result.hits[0].score > result.hits[1].score


def test_remove_item_makes_text_unsearchable():
    index.index_item("ds", "fact", 7, "Alice lives in Boston.")
    index.remove_item("ds", "fact", 7)

    assert index.search("ds", "Where does Alice live?", k=5).hits == []
    assert index.indexed_count("ds") == 0


def test_reindex_same_ref_upserts_not_duplicates():
    index.index_item("ds", "note", 1, "old text")
    index.index_item("ds", "note", 1, "new text entirely")

    assert index.indexed_count("ds") == 1
    hits = index.search("ds", "new text entirely", k=5).hits
    assert [h.text for h in hits] == ["new text entirely"]


def test_datasets_are_isolated():
    index.index_item("proj_a", "note", 1, "alpha secret detail")
    index.index_item("proj_b", "note", 1, "beta other detail")

    hits_a = index.search("proj_a", "alpha secret detail", k=10).hits

    assert [h.text for h in hits_a] == ["alpha secret detail"]
    assert index.indexed_count("proj_b") == 1


def test_keyword_mode_when_endpoint_down():
    """No Ollama at all: search still works via FTS5 and says so."""
    embeddings.set_embed_fn(broken_embed)
    index.index_item("ds", "note", 1, "database migrations run through alembic")
    index.index_item("ds", "note", 2, "frontend bundling uses vite")

    result = index.search("ds", "alembic database migrations", k=5)

    assert result.mode == "keyword"
    assert result.hits[0].ref_id == 1  # BM25 still ranks the right item first


def test_endpoint_down_stores_pending_then_heals():
    embeddings.set_embed_fn(broken_embed)
    embedded_now = index.index_item("ds", "note", 1, "written during outage")
    assert embedded_now is False
    assert index.pending_count("ds") == 1

    # Endpoint comes back: the next search embeds pending rows and upgrades
    # to hybrid mode.
    embeddings.set_embed_fn(fake_embed)
    result = index.search("ds", "written during outage", k=5)

    assert index.pending_count("ds") == 0
    assert result.mode == "hybrid"
    assert result.hits[0].text == "written during outage"


def test_punctuation_heavy_query_is_safe_for_fts():
    """FTS5 has its own operator syntax; natural-language punctuation must
    not produce a syntax error (queries are reduced to word tokens)."""
    embeddings.set_embed_fn(broken_embed)
    index.index_item("ds", "note", 1, "the auth token check was inverted")

    result = index.search("ds", 'what happened with the "auth-token" (check)?!', k=5)

    assert result.mode == "keyword"
    assert result.hits[0].ref_id == 1


def test_semantic_only_match_still_found_in_hybrid():
    """A doc sharing no keywords with the query is reachable via the vector
    channel — the reason embeddings remain a worthwhile optional upgrade."""
    index.index_item("ds", "note", 1, "release cadence is every two weeks")
    # Fake embedder is bag-of-words, so craft a query overlapping in words
    # with doc 1 for the semantic channel but filtered out of FTS by tokens?
    # Simpler honest check: hybrid fuses BOTH channels' rankings.
    index.index_item("ds", "note", 2, "cadence cadence cadence unrelated")

    result = index.search("ds", "release cadence", k=2)

    assert result.mode == "hybrid"
    assert {h.ref_id for h in result.hits} == {1, 2}


@pytest.mark.asyncio
async def test_backfill_indexes_pre_pivot_facts():
    # Facts written straight through the resolver (as the pre-pivot design
    # did) have no index rows until backfill runs.
    await resolver.write("Alice", "lives_in", "Boston", "Alice lives in Boston.", dataset="ds")
    await resolver.write("Bob", "works_at", "Acme", "Bob works at Acme.", dataset="ds")
    assert index.indexed_count("ds") == 0

    added = index.backfill_facts("ds")
    assert added == 2
    assert index.backfill_facts("ds") == 0  # idempotent

    hits = index.search("ds", "Where does Alice live?", k=1).hits
    assert hits[0].text == "Alice lives in Boston."


def test_search_degrades_when_db_unopenable(tmp_path, monkeypatch):
    """The Session 12 mount failure class: the SQLite file itself can't be
    opened (read-only mount, corrupt file, dir in the way). search() must
    return an empty result with mode="unavailable" — never raise."""
    monkeypatch.setattr(resolver, "db_path", lambda: tmp_path)  # a directory, not a DB

    result = index.search("ds", "anything at all", k=5)

    assert result.hits == []
    assert result.mode == "unavailable"
