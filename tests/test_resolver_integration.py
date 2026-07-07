"""Integration test for core/resolver.py's ambiguity heuristic -- real
Ollama, no Cognee. Marked `integration` — run explicitly with:

    uv run pytest -m integration -q tests/test_resolver_integration.py

`write()`'s fast-suite tests (tests/test_resolver.py) only exercise clean
conflicts (no object-string overlap, e.g. "Boston" -> "Seattle") and exact
restatements -- neither trips `_is_ambiguous()`. This file covers the branch
that does: an overlapping object pair (e.g. "Boston" -> "Boston, MA") routes
through `core/router.py`'s `judge_conflict`, and the real verdict it returns
isn't fully deterministic run-to-run. Rather than asserting one exact
verdict, each test asserts that (a) the ambiguous branch was actually taken
(a `conflict_judge` CallRecord was logged) and (b) `write()`'s resulting
state is internally consistent with *whichever* verdict came back.
"""

from __future__ import annotations

import pytest

from core import resolver, router

# The docstring above always said "Marked `integration`", but this line was
# missing — so these real-Ollama tests silently ran inside the "LLM-free"
# fast suite (passing only when Ollama happened to be up). Caught in Session
# 10 when the fast suite ran in a no-Ollama sandbox for the first time.
pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_ambiguous_conflict_routes_through_judge_conflict(tmp_path):
    resolver.configure(db_path=tmp_path / "facts.db")
    router.reset_ledger()

    await resolver.write("Alice", "lives_in", "Boston", "Alice lives in Boston.")
    result = await resolver.write(
        "Alice", "lives_in", "Boston, MA", "Alice lives in Boston, MA."
    )

    judge_records = [r for r in router._ledger if r.task == "conflict_judge"]
    assert len(judge_records) == 1, "overlapping object strings should trigger judge_conflict"

    if result.changed is False:
        # verdict == "same": treated as a no-op, like an exact restatement.
        assert result.superseded is None
        current = resolver.current_facts()
        assert len(current) == 1
    elif result.distinct:
        # verdict == "distinct": both facts remain current under the same
        # canonical key rather than one superseding the other.
        assert result.superseded is None
        current = [f for f in resolver.current_facts() if f.subject == "Alice"]
        assert len(current) == 2
        assert {f.object for f in current} == {"Boston", "Boston, MA"}
    else:
        # verdict == "update": ordinary supersede.
        assert result.changed is True
        assert result.superseded is not None
        assert result.superseded.object == "Boston"
        current = resolver.current_facts()
        assert len(current) == 1
        assert current[0].object == "Boston, MA"


@pytest.mark.asyncio
async def test_non_overlapping_conflict_never_calls_judge_conflict(tmp_path):
    resolver.configure(db_path=tmp_path / "facts.db")
    router.reset_ledger()

    await resolver.write("Alice", "lives_in", "Boston", "Alice lives in Boston.")
    result = await resolver.write("Alice", "lives_in", "Seattle", "Alice moved to Seattle.")

    judge_records = [r for r in router._ledger if r.task == "conflict_judge"]
    assert judge_records == [], "non-overlapping objects must stay fully deterministic"
    assert result.changed is True
    assert result.superseded is not None
