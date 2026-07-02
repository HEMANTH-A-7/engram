"""Unit tests for core/resolver.py — the bi-temporal fact store.

No Ollama, no Cognee. Each test gets an isolated SQLite file under pytest's
tmp_path, so this runs in the fast default suite (no `integration` marker)
and proves the versioning logic is correct on its own, independent of
extraction quality. `write()` is `async def` since Bucket 5 (it may await
`core/router.py`'s `judge_conflict` on an ambiguous conflict) — every case
below uses either a clean conflict (no object-string overlap) or an exact
restatement, neither of which trigger that path, so these stay LLM-free
despite the `async`/`await` plumbing.
"""

from __future__ import annotations

import pytest

from core import resolver


def _fresh(tmp_path):
    resolver.configure(db_path=tmp_path / "facts.db")


@pytest.mark.asyncio
async def test_first_write_has_no_supersede(tmp_path):
    _fresh(tmp_path)
    result = await resolver.write("Alice", "lives_in", "Boston", "Alice lives in Boston.")

    assert result.changed is True
    assert result.superseded is None
    assert result.fact.object == "Boston"
    assert result.fact.valid_to is None


@pytest.mark.asyncio
async def test_conflicting_write_supersedes_old_fact(tmp_path):
    _fresh(tmp_path)
    first = await resolver.write("Alice", "lives_in", "Boston", "Alice lives in Boston.")
    second = await resolver.write("Alice", "lives_in", "Seattle", "Alice moved to Seattle.")

    assert second.changed is True
    assert second.superseded is not None
    assert second.superseded.id == first.fact.id
    assert second.superseded.valid_to is not None
    assert second.superseded.superseded_by == second.fact.id
    assert second.fact.valid_to is None


@pytest.mark.asyncio
async def test_current_facts_only_returns_the_latest_version(tmp_path):
    _fresh(tmp_path)
    await resolver.write("Alice", "lives_in", "Boston", "Alice lives in Boston.")
    await resolver.write("Alice", "lives_in", "Seattle", "Alice moved to Seattle.")
    await resolver.write("Bob", "favorite_color", "green", "Bob's favorite color is green.")

    current = {f"{f.subject}:{f.relation}": f.object for f in resolver.current_facts()}

    assert current == {
        "Alice:lives_in": "Seattle",
        "Bob:favorite_color": "green",
    }


@pytest.mark.asyncio
async def test_restating_the_same_fact_is_a_no_op(tmp_path):
    _fresh(tmp_path)
    first = await resolver.write("Alice", "lives_in", "Boston", "Alice lives in Boston.")
    again = await resolver.write("Alice", "lives_in", "boston", "Alice still lives in Boston.")

    assert again.changed is False
    assert again.superseded is None
    assert again.fact.id == first.fact.id
    assert len(resolver.current_facts()) == 1


def test_canonical_key_ignores_case_and_whitespace():
    assert resolver.canonical_key("Alice", "lives_in") == resolver.canonical_key(
        "  alice  ", "Lives_In"
    )


@pytest.mark.asyncio
async def test_history_preserves_all_versions_oldest_first(tmp_path):
    _fresh(tmp_path)
    await resolver.write("Alice", "lives_in", "Boston", "Alice lives in Boston.")
    await resolver.write("Alice", "lives_in", "Seattle", "Alice moved to Seattle.")
    await resolver.write("Alice", "lives_in", "Denver", "Alice moved to Denver.")

    versions = resolver.history("Alice", "lives_in")

    assert [f.object for f in versions] == ["Boston", "Seattle", "Denver"]
    assert versions[0].valid_to is not None
    assert versions[1].valid_to is not None
    assert versions[2].valid_to is None


@pytest.mark.asyncio
async def test_different_relations_do_not_conflict(tmp_path):
    _fresh(tmp_path)
    await resolver.write("Alice", "lives_in", "Boston", "Alice lives in Boston.")
    result = await resolver.write(
        "Alice", "favorite_color", "blue", "Alice's favorite color is blue."
    )

    assert result.superseded is None
    assert len(resolver.current_facts()) == 2


@pytest.mark.asyncio
async def test_datasets_are_isolated(tmp_path):
    _fresh(tmp_path)
    await resolver.write("Alice", "lives_in", "Boston", "Alice lives in Boston.", dataset="a")
    await resolver.write("Alice", "lives_in", "Seattle", "Alice moved to Seattle.", dataset="b")

    assert [f.object for f in resolver.current_facts("a")] == ["Boston"]
    assert [f.object for f in resolver.current_facts("b")] == ["Seattle"]


@pytest.mark.asyncio
async def test_reset_scoped_to_dataset(tmp_path):
    _fresh(tmp_path)
    await resolver.write("Alice", "lives_in", "Boston", "Alice lives in Boston.", dataset="a")
    await resolver.write("Bob", "lives_in", "Denver", "Bob lives in Denver.", dataset="b")

    resolver.reset(dataset="a")

    assert resolver.current_facts("a") == []
    assert len(resolver.current_facts("b")) == 1
