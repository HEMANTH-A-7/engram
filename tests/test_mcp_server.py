"""Fast unit tests for mcp_server/server.py -- no Ollama, no Cognee.

Exercises the two tools whose logic is pure Python (`memory_stats`
aggregation, `memory_forget`'s not-found short-circuit) against an isolated
`tmp_path` resolver DB, mirroring tests/test_resolver.py's offline pattern.
The two pipeline-touching paths (`memory_write`, `memory_forget` on a real
hit -> Cognee reindex) are covered by tests/test_mcp_server_integration.py.
Every assertion here also confirms the tool's return value is JSON-
serializable, since these cross an MCP (JSON-RPC) boundary in production.
"""

from __future__ import annotations

import json

import pytest

from core import resolver, router
from mcp_server import server


def _fresh(tmp_path):
    resolver.configure(db_path=tmp_path / "facts.db")
    router.reset_ledger()


@pytest.mark.asyncio
async def test_memory_stats_counts_facts_by_tier(tmp_path):
    _fresh(tmp_path)
    a = await resolver.write("Alice", "lives_in", "Boston", "Alice lives in Boston.")
    b = await resolver.write("Bob", "favorite_color", "green", "Bob's favorite color is green.")
    resolver.set_tier(b.fact.id, "warm")

    stats = await server.memory_stats()

    assert stats["facts"]["total_current"] == 2
    assert stats["facts"]["by_tier"]["hot"] == 1
    assert stats["facts"]["by_tier"]["warm"] == 1
    assert stats["facts"]["by_tier"]["cold"] == 0
    assert a.fact.id != b.fact.id  # sanity: two distinct facts seeded
    json.dumps(stats)  # must survive the JSON-RPC boundary


@pytest.mark.asyncio
async def test_memory_stats_includes_cost_report(tmp_path):
    _fresh(tmp_path)
    router._ledger.append(
        router.CallRecord(
            task="extract",
            model_tier="small",
            prompt_tokens=100,
            completion_tokens=20,
            attempt=1,
            succeeded=True,
            latency_s=0.1,
        )
    )

    stats = await server.memory_stats()

    assert stats["cost"]["total_calls"] == 1
    assert stats["cost"]["by_task"]["extract"]["successes"] == 1
    assert "pricing_note" in stats["cost"]  # honesty caveat surfaced to the dashboard
    json.dumps(stats)


@pytest.mark.asyncio
async def test_memory_stats_benchmarks_is_a_dict(tmp_path):
    _fresh(tmp_path)
    stats = await server.memory_stats()
    # Loaded from benchmark/results/*.json if present; always a dict either way.
    assert isinstance(stats["benchmarks"], dict)
    json.dumps(stats)


@pytest.mark.asyncio
async def test_memory_forget_missing_id_returns_not_found(tmp_path):
    _fresh(tmp_path)
    await resolver.write("Alice", "lives_in", "Boston", "Alice lives in Boston.")

    result = await server.memory_forget(id=999_999)

    assert result == {"found": False, "evicted_id": None}
    # The fact we wrote is untouched -- not-found must not evict anything.
    assert len(resolver.current_facts()) == 1
    json.dumps(result)


@pytest.mark.asyncio
async def test_memory_search_returns_serializable_shape_on_empty_index(tmp_path, monkeypatch):
    _fresh(tmp_path)

    # Stub the consolidation search so this stays LLM/Cognee-free while still
    # exercising memory_search's response shaping (the real search path is
    # covered end-to-end in the integration test).
    async def _fake_search(query, dataset="main_dataset", k=5):
        return ["Alice lives in Boston."]

    monkeypatch.setattr(server.consolidation, "search", _fake_search)

    result = await server.memory_search("Where does Alice live?", k=3)

    assert result["query"] == "Where does Alice live?"
    assert result["hits"] == ["Alice lives in Boston."]
    # hit_facts is index-aligned with hits; with an empty resolver the hit
    # text can't be mapped back to a fact, so id is null (the honest anomaly
    # path) -- but the text is still echoed so the shape stays uniform.
    assert result["hit_facts"] == [
        {
            "id": None,
            "subject": None,
            "relation": None,
            "object": None,
            "text": "Alice lives in Boston.",
        }
    ]
    json.dumps(result)


@pytest.mark.asyncio
async def test_memory_search_hit_facts_carry_ids_for_forget(tmp_path, monkeypatch):
    """The id a client feeds to memory_forget is discoverable from a search.

    Seeds a real fact in the resolver, stubs the Cognee search to return that
    fact's exact text (the 1:1 indexing invariant), and asserts the hit is
    resolved back to its id + triple -- no Ollama, no Cognee. The forget
    round-trip that consumes this id is covered in the integration suite.
    """
    _fresh(tmp_path)
    written = await resolver.write(
        "Alice", "lives_in", "Boston", "Alice lives in Boston."
    )

    async def _fake_search(query, dataset="main_dataset", k=5):
        return ["Alice lives in Boston."]

    monkeypatch.setattr(server.consolidation, "search", _fake_search)

    result = await server.memory_search("Where does Alice live?", k=3)

    assert result["hits"] == ["Alice lives in Boston."]
    assert result["hit_facts"] == [
        {
            "id": written.fact.id,
            "subject": "Alice",
            "relation": "lives_in",
            "object": "Boston",
            "text": "Alice lives in Boston.",
        }
    ]
    json.dumps(result)
