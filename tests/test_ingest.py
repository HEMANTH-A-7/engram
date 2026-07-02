"""Integration test for core/ingest.py — the Bucket 3 conflict resolver.

Runs the real Cognee + Ollama pipeline (offline), so it's slow (multiple
cognify passes: initial write + supersede). Marked `integration` — run
explicitly with:

    uv run pytest -m integration -q tests/test_ingest.py

Asserts that after a superseding update, Cognee's search index reflects only
the current fact — the superseded value's text should be gone entirely, not
just outranked.
"""

from __future__ import annotations

import pytest

from cognee.modules.search.types import SearchType

from core import ingest, resolver, store

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_supersede_removes_stale_text_from_search():
    dataset = "test_ingest_dataset"
    resolver.reset(dataset)

    first = await ingest.remember("Alice lives in Boston.", dataset=dataset)
    assert first.write.changed is True
    assert first.write.superseded is None

    second = await ingest.remember(
        "Alice now lives in Seattle.",
        dataset=dataset,
    )
    assert second.write.changed is True
    assert second.write.superseded is not None
    assert second.write.superseded.object.lower() != second.write.fact.object.lower()

    hits = await store.search(
        "Where does Alice live?", k=3, query_type=SearchType.CHUNKS, dataset=dataset
    )
    joined = " ".join(str(h) for h in hits)
    assert "Seattle" in joined, f"current value missing from search: {joined!r}"
    assert "Boston" not in joined, f"superseded value leaked into search: {joined!r}"

    current = resolver.current_facts(dataset)
    assert len(current) == 1
    assert current[0].object == "Seattle"
