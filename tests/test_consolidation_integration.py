"""Integration test for core/consolidation.py's resync_active() + eviction.

Runs the real Cognee + Ollama pipeline (offline), so it's slow (three
cognify passes: initial index, post-eviction reindex, post-revival
reindex). Marked `integration` — run explicitly with:

    uv run pytest -m integration -q tests/test_consolidation_integration.py

Asserts that a fact evicted by `run_consolidation_pass` is actually gone
from Cognee's search index once `resync_active` rebuilds it (not just
outranked), and that reviving the same fact (re-writing it, which flips
`revived_from_eviction`) makes it searchable again after another
`resync_active`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.modules.search.types import SearchType

from core import consolidation, resolver, store

pytestmark = pytest.mark.integration


async def _search_text(query: str, dataset: str) -> str:
    """Search and join hit text, treating an empty index as empty results.

    `resync_active` rebuilds Cognee's index from only the hot tier -- when
    everything's evicted, that's zero facts, which wipes the dataset
    entirely rather than leaving a queryable-but-empty index. Cognee raises
    `NoDataError` in that case; that's exactly "unsearchable", so it's
    treated the same as an empty hit list rather than a test failure.
    """
    try:
        hits = await store.search(query, k=3, query_type=SearchType.CHUNKS, dataset=dataset)
    except NoDataError:
        return ""
    return " ".join(str(h) for h in hits).lower()


@pytest.mark.asyncio
async def test_evicted_fact_becomes_unsearchable_and_revival_restores_it():
    dataset = "test_consolidation_dataset"
    resolver.reset(dataset)

    written = await resolver.write(
        "Carol", "hobby", "pottery", "Carol's hobby is pottery.", dataset=dataset
    )
    await consolidation.resync_active(dataset)

    joined = await _search_text("What is Carol's hobby?", dataset)
    assert "pottery" in joined, f"fact missing before eviction: {joined!r}"

    future = datetime.now(timezone.utc) + timedelta(days=60)
    report = await consolidation.run_consolidation_pass(dataset, now=future)
    assert written.fact.id in report.evicted_ids
    assert resolver.current_facts(dataset) == []

    await consolidation.resync_active(dataset)
    joined = await _search_text("What is Carol's hobby?", dataset)
    assert "pottery" not in joined, f"evicted fact still searchable: {joined!r}"

    revival = await resolver.write(
        "Carol", "hobby", "pottery", "Carol's hobby is pottery.", dataset=dataset
    )
    assert revival.revived_from_eviction is True

    await consolidation.resync_active(dataset)
    joined = await _search_text("What is Carol's hobby?", dataset)
    assert "pottery" in joined, f"revived fact not searchable again: {joined!r}"
