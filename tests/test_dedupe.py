"""Unit tests for core/dedupe.py and the write-time near-duplicate guard.

No Ollama, no Cognee — isolated SQLite per test. `core.router` imports the
LLM stack at module load, so a no-LLM install (and this suite) can't import
it; the guard's judge behavior is controlled by planting a stub module in
`sys.modules["core.router"]`, exactly the seam `resolver.write`'s local
import goes through. The motivating case throughout is the real Session 11
pair: `("Hemanth", "prefers", ...)` vs `("Hemanth", "storage_preference",
...)` — different canonical keys, same real-world attribute.
"""

from __future__ import annotations

import sys
import types

import pytest

import core
from core import dedupe, resolver


def _fresh(tmp_path):
    resolver.configure(db_path=tmp_path / "facts.db")


def _plant_judge(monkeypatch, judge) -> None:
    """Install a fake core.router whose judge_conflict is `judge`."""
    mod = types.ModuleType("core.router")
    mod.judge_conflict = judge
    monkeypatch.setitem(sys.modules, "core.router", mod)
    monkeypatch.setattr(core, "router", mod, raising=False)


def _no_judge(monkeypatch) -> None:
    """Simulate the no-LLM floor: the judge import path raises."""

    async def raiser(old_text, new_text):
        raise RuntimeError("no judge in this test")

    _plant_judge(monkeypatch, raiser)


# --- similarity floor -------------------------------------------------------


def test_relations_similar_catches_the_session11_pair():
    assert dedupe.relations_similar("prefers", "storage_preference")


def test_relations_similar_rejects_unrelated_relations():
    assert not dedupe.relations_similar("developing", "prefers")
    assert not dedupe.relations_similar("lives_in", "works_in")  # stopword-only overlap
    assert not dedupe.relations_similar("likes", "dislikes")


def test_real_main_dataset_vocabulary_has_no_false_positives():
    # The seven relations current in the real main_dataset as of Session 12.
    vocab = [
        "supersedes",
        "developing",
        "prefers",
        "testing_memory_layer_through_live_dashboard",
        "exposes",
        "uses",
        "favorite_language",
    ]
    hits = [
        (a, b)
        for i, a in enumerate(vocab)
        for b in vocab[i + 1 :]
        if dedupe.relations_similar(a, b)
    ]
    assert hits == []


# --- write-time guard, no judge (the floor) ---------------------------------


@pytest.mark.asyncio
async def test_guard_flags_near_duplicate_without_merging(tmp_path, monkeypatch):
    _fresh(tmp_path)
    _no_judge(monkeypatch)

    first = await resolver.write(
        "Hemanth", "prefers", "offline storage", "Hemanth prefers offline storage.",
        dataset="d",
    )
    second = await resolver.write(
        "Hemanth", "storage_preference", "cloud", "Hemanth now prefers cloud storage.",
        dataset="d",
    )

    # Floor behavior: BOTH stay current (no silent merge), pair is flagged.
    assert second.changed is True
    assert second.superseded is None
    assert second.flagged_duplicate == first.fact.id
    assert len(resolver.current_facts("d")) == 2
    flags = dedupe.unresolved_flags("d")
    assert len(flags) == 1
    assert {f["id"] for f in flags[0]["facts"]} == {first.fact.id, second.fact.id}


@pytest.mark.asyncio
async def test_guard_ignores_different_subjects_and_relations(tmp_path, monkeypatch):
    _fresh(tmp_path)
    _no_judge(monkeypatch)

    await resolver.write("Hemanth", "prefers", "offline", "t1", dataset="d")
    r2 = await resolver.write("Alice", "storage_preference", "cloud", "t2", dataset="d")
    r3 = await resolver.write("Hemanth", "developing", "engram", "t3", dataset="d")

    assert r2.flagged_duplicate is None
    assert r3.flagged_duplicate is None
    assert dedupe.unresolved_flags("d") == []


@pytest.mark.asyncio
async def test_exact_key_supersede_path_is_untouched(tmp_path, monkeypatch):
    _fresh(tmp_path)
    _no_judge(monkeypatch)

    first = await resolver.write("Alice", "lives_in", "Boston", "t1", dataset="d")
    second = await resolver.write("Alice", "lives_in", "Seattle", "t2", dataset="d")

    assert second.superseded is not None
    assert second.superseded.id == first.fact.id
    assert second.flagged_duplicate is None


@pytest.mark.asyncio
async def test_guard_is_per_dataset(tmp_path, monkeypatch):
    _fresh(tmp_path)
    _no_judge(monkeypatch)

    await resolver.write("Hemanth", "prefers", "offline", "t1", dataset="d1")
    second = await resolver.write(
        "Hemanth", "storage_preference", "cloud", "t2", dataset="d2"
    )

    assert second.flagged_duplicate is None  # other dataset: invisible


# --- write-time guard, judge available --------------------------------------


@pytest.mark.asyncio
async def test_guard_with_judge_update_supersedes_near_duplicate(tmp_path, monkeypatch):
    _fresh(tmp_path)

    async def judge(old_text, new_text):
        return "update"

    _plant_judge(monkeypatch, judge)

    first = await resolver.write("Hemanth", "prefers", "offline storage", "t1", dataset="d")
    second = await resolver.write("Hemanth", "storage_preference", "cloud", "t2", dataset="d")

    assert second.superseded is not None
    assert second.superseded.id == first.fact.id
    assert second.flagged_duplicate is None
    assert len(resolver.current_facts("d")) == 1


@pytest.mark.asyncio
async def test_guard_with_judge_same_is_a_no_op(tmp_path, monkeypatch):
    _fresh(tmp_path)

    async def judge(old_text, new_text):
        return "same"

    _plant_judge(monkeypatch, judge)

    first = await resolver.write("Hemanth", "prefers", "offline", "t1", dataset="d")
    second = await resolver.write("Hemanth", "storage_preference", "offline", "t2", dataset="d")

    assert second.changed is False
    assert second.fact.id == first.fact.id
    assert len(resolver.current_facts("d")) == 1


@pytest.mark.asyncio
async def test_guard_with_judge_distinct_writes_without_flag(tmp_path, monkeypatch):
    _fresh(tmp_path)

    async def judge(old_text, new_text):
        return "distinct"

    _plant_judge(monkeypatch, judge)

    await resolver.write("Hemanth", "prefers", "offline", "t1", dataset="d")
    second = await resolver.write("Hemanth", "storage_preference", "cloud", "t2", dataset="d")

    assert second.changed is True
    assert second.flagged_duplicate is None
    assert len(resolver.current_facts("d")) == 2
    assert dedupe.unresolved_flags("d") == []


# --- flags lifecycle ---------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_goes_stale_when_one_side_is_evicted(tmp_path, monkeypatch):
    _fresh(tmp_path)
    _no_judge(monkeypatch)

    first = await resolver.write("Hemanth", "prefers", "offline", "t1", dataset="d")
    await resolver.write("Hemanth", "storage_preference", "cloud", "t2", dataset="d")
    assert len(dedupe.unresolved_flags("d")) == 1

    resolver.evict(first.fact.id, dataset="d")  # user forgets one side
    assert dedupe.unresolved_flags("d") == []  # auto-resolved as stale


def test_flag_pair_is_idempotent(tmp_path):
    _fresh(tmp_path)
    assert dedupe.flag_pair("d", 1, 2, "r") is True
    assert dedupe.flag_pair("d", 2, 1, "r") is False  # order-insensitive


# --- batch sweep -------------------------------------------------------------


@pytest.mark.asyncio
async def test_sweep_flag_only_finds_preexisting_pair(tmp_path, monkeypatch):
    _fresh(tmp_path)

    # Simulate data written BEFORE the guard existed: judge says 'distinct'
    # at write time (both rows land, nothing flagged)...
    async def judge(old_text, new_text):
        return "distinct"

    _plant_judge(monkeypatch, judge)
    await resolver.write("Hemanth", "prefers", "offline", "t1", dataset="d")
    await resolver.write("Hemanth", "storage_preference", "cloud", "t2", dataset="d")

    # ...then the floor sweep takes a fresh look.
    report = await dedupe.sweep("d", use_judge=False)

    assert report["mode"] == "flag-only"
    assert report["pairs_found"] == 1
    assert report["flagged"] == 1
    assert len(dedupe.unresolved_flags("d")) == 1


@pytest.mark.asyncio
async def test_sweep_with_judge_update_closes_older_row(tmp_path, monkeypatch):
    _fresh(tmp_path)

    verdicts = iter(["distinct", "update"])  # write-time, then sweep-time

    async def judge(old_text, new_text):
        return next(verdicts)

    _plant_judge(monkeypatch, judge)
    first = await resolver.write("Hemanth", "prefers", "offline", "t1", dataset="d")
    second = await resolver.write("Hemanth", "storage_preference", "cloud", "t2", dataset="d")

    report = await dedupe.sweep("d", use_judge=True)

    assert report["mode"] == "judge"
    assert report["resolved"] == [
        {"older": first.fact.id, "newer": second.fact.id, "verdict": "update"}
    ]
    assert [f.id for f in resolver.current_facts("d")] == [second.fact.id]
    closed = next(f for f in resolver.all_versions("d") if f.id == first.fact.id)
    assert closed.valid_to is not None
    assert closed.superseded_by == second.fact.id


@pytest.mark.asyncio
async def test_sweep_is_idempotent_on_flags(tmp_path, monkeypatch):
    _fresh(tmp_path)
    _no_judge(monkeypatch)

    await resolver.write("Hemanth", "prefers", "offline", "t1", dataset="d")
    await resolver.write("Hemanth", "storage_preference", "cloud", "t2", dataset="d")

    r1 = await dedupe.sweep("d", use_judge=False)
    r2 = await dedupe.sweep("d", use_judge=False)

    assert r1["pairs_found"] == r2["pairs_found"] == 1
    assert r1["flagged"] == 0  # already flagged by the write-time guard
    assert r2["flagged"] == 0
    assert len(dedupe.unresolved_flags("d")) == 1
