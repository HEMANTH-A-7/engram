"""Fast tests for the live dashboard metrics -- no Ollama, no cognify.

Each test seeds an isolated throwaway resolver DB (via `resolver.configure`,
same pattern as `tests/test_resolver_integration.py`) with a known shape: two
distinct facts, one in-place update (supersede), and one eviction. All writes
use non-overlapping objects so `resolver.write` stays fully deterministic and
never calls the conflict-judge LLM. The metrics are then asserted against that
known history, so `core.live_metrics` is verified without the slow pipeline.
"""

from __future__ import annotations

from core import live_metrics, resolver


async def _seed(dataset: str) -> None:
    # Alice/London, then Grace/debugging, then update Alice -> Paris (supersede),
    # then evict Grace. Final history = 3 rows; final current = 1 (Alice/Paris).
    await resolver.write("Alice", "lives_in", "London", "Alice lives in London.", dataset=dataset)
    grace = await resolver.write(
        "Grace", "coined", "debugging", "Grace coined the term debugging.", dataset=dataset
    )
    await resolver.write("Alice", "lives_in", "Paris", "Alice lives in Paris.", dataset=dataset)
    resolver.evict(grace.fact.id, dataset=dataset)


async def test_all_versions_keeps_history_current_facts_dedups(tmp_path):
    resolver.configure(db_path=tmp_path / "facts.db")
    await _seed("t")
    assert len(resolver.all_versions("t")) == 3  # London, Grace, Paris rows all kept
    assert len(resolver.current_facts("t")) == 1  # only Alice/Paris (Grace evicted)


async def test_token_savings_raw_history_exceeds_current(tmp_path):
    resolver.configure(db_path=tmp_path / "facts.db")
    await _seed("t")
    ts = live_metrics.token_savings("t")
    assert ts["n_writes"] == 3
    assert ts["n_current_facts"] == 1
    assert ts["raw_history_tokens"] > ts["extended_tokens"] > 0
    assert 0 < ts["pct_saved"] <= 1


async def test_storage_series_diverges_and_ends_at_current(tmp_path):
    resolver.configure(db_path=tmp_path / "facts.db")
    await _seed("t")
    ss = live_metrics.storage_series("t")
    assert [p["without"] for p in ss["series"]] == [1, 2, 3]  # raw history keeps every write
    assert ss["series"][-1]["with_"] == 1  # versioned layer ends at the current-fact count


async def test_revision_stats_counts_supersede_and_evict(tmp_path):
    resolver.configure(db_path=tmp_path / "facts.db")
    await _seed("t")
    rs = live_metrics.revision_stats("t")
    assert rs["revisions"] == 1  # London superseded by Paris
    assert rs["evicted"] == 1  # Grace evicted
    assert rs["current"] == 1
    assert rs["by_tier"] == {"hot": 1, "warm": 0, "cold": 0}


def test_estimate_tokens_is_chars_over_four():
    assert live_metrics._estimate_tokens("") == 0
    assert live_metrics._estimate_tokens("abcd") == 1
    assert live_metrics._estimate_tokens("a" * 40) == 10
