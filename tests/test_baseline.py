"""Integration test for the Bucket 1 baseline pipeline.

Runs the real Cognee pipeline against Ollama (offline), so it's slow (~1-3 min:
gemma4 structured extraction). Marked `integration` — run explicitly with:

    uv run pytest -m integration -q

Asserts a known fact is retrievable end-to-end.
"""

from __future__ import annotations

import pytest

from cognee.modules.search.types import SearchType

from core import store

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_known_fact_is_retrievable():
    await store.reset()
    await store.add("Grace Hopper developed the first compiler and popularized the term 'debugging'.")
    await store.add("The Python programming language was created by Guido van Rossum in 1991.")
    await store.cognify()

    # Graph-completion answer should name the right person.
    answer = await store.search(
        "Who created the Python programming language?",
        k=5,
        query_type=SearchType.GRAPH_COMPLETION,
    )
    joined = " ".join(str(a) for a in answer)
    assert "Guido" in joined or "Rossum" in joined, f"unexpected answer: {joined!r}"

    # Raw retrieval should return at least one hit.
    hits = await store.search("compiler debugging", k=3, query_type=SearchType.CHUNKS)
    assert len(hits) >= 1
