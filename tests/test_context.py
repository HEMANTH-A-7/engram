"""Unit tests for core/context.py — the budgeted memory digest.

No Ollama, no Cognee (fake embedder). Proves the memory_context contract:
current-only (superseded/evicted can never appear), deduplicated, budget
respected, semantic ranking when a query is given, honest recency fallback
when the endpoint is down, and self-reported digest-vs-raw measurement.
"""

from __future__ import annotations

import pytest

from core import context, embeddings, notes, resolver
from core.live_metrics import _estimate_tokens
from tests.fake_embed import broken_embed, fake_embed


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    resolver.configure(db_path=tmp_path / "facts.db")
    embeddings.set_embed_fn(fake_embed)
    yield
    embeddings.set_embed_fn(None)


@pytest.mark.asyncio
async def test_superseded_and_evicted_never_appear():
    await resolver.write("Alice", "lives_in", "Boston", "Alice lives in Boston.", dataset="ds")
    await resolver.write("Alice", "lives_in", "Seattle", "Alice now lives in Seattle.", dataset="ds")
    evicted = notes.add_note("note to forget", kind="progress", dataset="ds")
    notes.evict_note(evicted.note.id, dataset="ds")
    notes.add_note("Chose flock for cross-process locking.", kind="decision", dataset="ds")

    result = context.build_context(dataset="ds")

    assert "Seattle" in result["digest"]
    assert "Boston" not in result["digest"]  # superseded — current-only by construction
    assert "note to forget" not in result["digest"]
    assert "flock" in result["digest"]


def test_budget_is_respected_and_reported():
    for i in range(40):
        notes.add_note(
            f"progress item {i}: " + ("detail " * 30), kind="progress", dataset="ds"
        )

    result = context.build_context(token_budget=400, dataset="ds")

    assert result["digest_tokens"] <= 400
    assert _estimate_tokens(result["digest"]) == result["digest_tokens"]
    assert result["included"] > 0
    assert result["excluded"] > 0
    assert result["included"] + result["excluded"] == result["current_items"]


def test_digest_is_deduplicated():
    notes.add_note("The API key lives in .env", kind="gotcha", dataset="ds")
    notes.add_note("the api  key lives in .env", kind="reference", dataset="ds")  # same, normalized

    result = context.build_context(dataset="ds")

    assert result["current_items"] == 1
    assert result["digest"].lower().count("api") == 1


def test_query_ranks_by_relevance_and_reports_mode():
    notes.add_note("database migrations run through alembic", kind="gotcha", dataset="ds")
    notes.add_note("frontend bundling uses vite", kind="gotcha", dataset="ds")

    result = context.build_context(
        query="how do database migrations work", token_budget=120, dataset="ds"
    )

    assert result["ranking"] == "hybrid"
    assert "alembic" in result["digest"]


def test_endpoint_down_uses_keyword_mode_honestly():
    """No Ollama: relevance comes from FTS5, and the mode says so. Items the
    query doesn't match are still admitted by recency within budget — the
    digest is a broad recall payload, not a search result."""
    notes.add_note("some remembered thing", kind="progress", dataset="ds")
    embeddings.set_embed_fn(broken_embed)

    result = context.build_context(query="remembered thing", dataset="ds")

    assert result["ranking"] == "keyword"
    assert "some remembered thing" in result["digest"]


def test_empty_memory_yields_empty_digest():
    result = context.build_context(dataset="empty_ds")

    assert result["digest"] == ""
    assert result["included"] == 0
    assert result["raw_history_tokens"] == 0


@pytest.mark.asyncio
async def test_measurement_counts_full_history_as_baseline():
    # Realistic-length items: the digest carries fixed markdown scaffolding
    # (header, section titles, date prefixes), so on a handful of ten-char
    # strings it can legitimately be BIGGER than raw history — savings come
    # from dropping superseded revisions, which needs revisions of real size.
    fact_v1 = "The ingestion service writes to the staging bucket " + "x" * 100
    fact_v2 = "The ingestion service now writes to the prod bucket " + "y" * 100
    await resolver.write("ingestion", "writes_to", "staging", fact_v1, dataset="ds")
    await resolver.write("ingestion", "writes_to", "prod", fact_v2, dataset="ds")
    notes.add_note("session state v1: " + "z" * 200, kind="progress", dataset="ds", key="s")
    notes.add_note("session state v2: " + "w" * 200, kind="progress", dataset="ds", key="s")

    result = context.build_context(dataset="ds")

    # Baseline = every version ever written (2 facts + 2 note revisions);
    # digest = current-only (1 fact + 1 note), so it must be smaller.
    assert result["digest_tokens"] < result["raw_history_tokens"]
    assert 0 < result["pct_of_raw"] < 1


def test_oversized_item_is_truncated_not_budget_eating():
    notes.add_note("huge " * 2000, kind="progress", dataset="ds")
    notes.add_note("small decision that matters", kind="decision", dataset="ds")

    result = context.build_context(token_budget=500, dataset="ds")

    assert result["included"] == 2  # truncation kept room for both
    assert "…" in result["digest"]
    assert "small decision that matters" in result["digest"]
