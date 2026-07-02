"""Unit tests for core/consolidation.py -- tiered storage scoring + eviction.

Pure Python: no LLM calls, no Ollama, no Cognee. Mirrors
tests/test_resolver.py's pattern (isolated SQLite file per test via
pytest's tmp_path), so this runs in the fast default suite.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core import consolidation, resolver


def _fresh(tmp_path):
    resolver.configure(db_path=tmp_path / "facts.db")


def _fact(
    access_count=0,
    last_accessed=None,
    ingestion_time=None,
    importance=0.5,
):
    now = datetime.now(timezone.utc)
    return resolver.Fact(
        id=1,
        dataset="d",
        subject="s",
        relation="r",
        object="o",
        text="t",
        event_time=now.isoformat(),
        ingestion_time=ingestion_time or now.isoformat(),
        valid_to=None,
        superseded_by=None,
        access_count=access_count,
        last_accessed=last_accessed,
        tier="hot",
        importance=importance,
        evicted_at=None,
    )


# ---- score_fact ----------------------------------------------------------


def test_score_fresh_never_accessed_default_importance():
    now = datetime.now(timezone.utc)
    fact = _fact(ingestion_time=now.isoformat())
    score = consolidation.score_fact(fact, now, max_access_count=0)
    assert score == pytest.approx(0.6, abs=1e-6)  # 0.5*1 + 0.3*0 + 0.2*0.5


def test_score_decays_by_half_at_half_life():
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=consolidation.RECENCY_HALF_LIFE_DAYS)
    fact = _fact(ingestion_time=old.isoformat())
    score = consolidation.score_fact(fact, now, max_access_count=0)
    assert score == pytest.approx(0.35, abs=1e-3)  # 0.5*0.5 + 0.3*0 + 0.2*0.5


def test_score_uses_last_accessed_over_ingestion_time_when_present():
    now = datetime.now(timezone.utc)
    long_ago = now - timedelta(days=1000)
    fact = _fact(ingestion_time=long_ago.isoformat(), last_accessed=now.isoformat())
    score = consolidation.score_fact(fact, now, max_access_count=0)
    assert score == pytest.approx(0.6, abs=1e-6)  # recency computed off last_accessed, not ancient ingestion_time


def test_frequency_score_scales_with_max_access_count():
    now = datetime.now(timezone.utc)
    maxed = _fact(access_count=9, last_accessed=now.isoformat())
    zeroed = _fact(access_count=0, last_accessed=now.isoformat())

    maxed_score = consolidation.score_fact(maxed, now, max_access_count=9)
    zeroed_score = consolidation.score_fact(zeroed, now, max_access_count=9)

    assert maxed_score == pytest.approx(0.9, abs=1e-6)  # 0.5*1 + 0.3*1 + 0.2*0.5
    assert zeroed_score == pytest.approx(0.6, abs=1e-6)


def test_importance_shifts_score_directly():
    now = datetime.now(timezone.utc)
    high = _fact(last_accessed=now.isoformat(), importance=1.0)
    low = _fact(last_accessed=now.isoformat(), importance=0.0)

    assert consolidation.score_fact(high, now, 0) == pytest.approx(0.7, abs=1e-6)
    assert consolidation.score_fact(low, now, 0) == pytest.approx(0.5, abs=1e-6)


# ---- assign_tier -----------------------------------------------------------


def test_tier_boundaries():
    assert consolidation.assign_tier(0.6) == "hot"
    assert consolidation.assign_tier(consolidation.HOT_THRESHOLD) == "hot"
    assert consolidation.assign_tier(consolidation.HOT_THRESHOLD - 1e-6) == "warm"
    assert consolidation.assign_tier(consolidation.WARM_THRESHOLD) == "warm"
    assert consolidation.assign_tier(consolidation.WARM_THRESHOLD - 1e-6) == "cold"
    assert consolidation.assign_tier(0.0) == "cold"


# ---- run_consolidation_pass / eviction -------------------------------------


def test_fresh_fact_stays_hot(tmp_path):
    _fresh(tmp_path)
    resolver.write("Alice", "lives_in", "Boston", "Alice lives in Boston.")

    report = consolidation.run_consolidation_pass("main_dataset")

    assert report.tier_counts == {"hot": 1, "warm": 0, "cold": 0}
    assert report.evicted_count == 0
    assert resolver.current_facts()[0].tier == "hot"


def test_never_accessed_fact_evicted_after_enough_simulated_time(tmp_path):
    _fresh(tmp_path)
    written = resolver.write("Alice", "lives_in", "Boston", "Alice lives in Boston.")

    future = datetime.now(timezone.utc) + timedelta(days=60)
    report = consolidation.run_consolidation_pass("main_dataset", now=future)

    assert report.evicted_ids == (written.fact.id,)
    assert report.tier_counts["cold"] == 1
    assert resolver.current_facts() == []


def test_consolidation_pass_is_idempotent(tmp_path):
    _fresh(tmp_path)
    resolver.write("Alice", "lives_in", "Boston", "Alice lives in Boston.")

    now = datetime.now(timezone.utc)
    first = consolidation.run_consolidation_pass("main_dataset", now=now)
    second = consolidation.run_consolidation_pass("main_dataset", now=now)

    assert first.tier_counts == second.tier_counts
    assert first.evicted_ids == second.evicted_ids


def test_accessed_fact_resists_eviction_longer(tmp_path):
    _fresh(tmp_path)
    written = resolver.write("Alice", "lives_in", "Boston", "Alice lives in Boston.")
    resolver.record_access(written.fact.id)

    future = datetime.now(timezone.utc) + timedelta(days=60)
    report = consolidation.run_consolidation_pass("main_dataset", now=future)

    # record_access bumped access_count to 1, and this is the only fact in
    # the dataset so it's also max_access_count=1 -- full frequency score
    # (0.3) plus the importance floor (0.1) keeps it above EVICT_THRESHOLD
    # even after recency fully decays, unlike the never-accessed case above.
    assert report.evicted_ids == ()
    assert resolver.current_facts()[0].tier == "warm"


# ---- revived_from_eviction / regret detection ------------------------------


def test_revival_flag_set_when_rewriting_an_evicted_fact(tmp_path):
    _fresh(tmp_path)
    resolver.write("Alice", "lives_in", "Boston", "Alice lives in Boston.")

    future = datetime.now(timezone.utc) + timedelta(days=60)
    consolidation.run_consolidation_pass("main_dataset", now=future)
    assert resolver.current_facts() == []

    revival = resolver.write("Alice", "lives_in", "Boston", "Alice lives in Boston.")

    assert revival.revived_from_eviction is True
    assert revival.changed is True
    assert revival.superseded is None
    assert len(resolver.current_facts()) == 1


def test_revival_flag_false_for_a_genuinely_new_fact(tmp_path):
    _fresh(tmp_path)
    result = resolver.write("Bob", "favorite_color", "green", "Bob's favorite color is green.")

    assert result.revived_from_eviction is False


def test_revival_flag_false_for_an_ordinary_supersede(tmp_path):
    _fresh(tmp_path)
    resolver.write("Alice", "lives_in", "Boston", "Alice lives in Boston.")
    second = resolver.write("Alice", "lives_in", "Seattle", "Alice moved to Seattle.")

    assert second.revived_from_eviction is False
    assert second.superseded is not None


# ---- current_facts(tiers=...) filtering ------------------------------------


def test_current_facts_tier_filter(tmp_path):
    _fresh(tmp_path)
    a = resolver.write("Alice", "lives_in", "Boston", "Alice lives in Boston.")
    b = resolver.write("Bob", "favorite_color", "green", "Bob's favorite color is green.")
    resolver.set_tier(b.fact.id, "warm")

    hot_only = resolver.current_facts("main_dataset", tiers={"hot"})
    warm_only = resolver.current_facts("main_dataset", tiers={"warm"})
    both = resolver.current_facts("main_dataset")

    assert [f.id for f in hot_only] == [a.fact.id]
    assert [f.id for f in warm_only] == [b.fact.id]
    assert len(both) == 2
