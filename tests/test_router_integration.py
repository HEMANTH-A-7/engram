"""Integration tests for core/router.py's cascade extraction, conflict
judgment, and summarization -- real Ollama, no Cognee. Marked `integration`
— run explicitly with:

    uv run pytest -m integration -q tests/test_router_integration.py

These don't assert on `llama3.2:3b`'s exact success/failure for a specific
fact -- Ollama's output is not fully deterministic run-to-run, and the
specific 5/10 facts that failed the small model in this bucket's original
manual test run were not preserved verbatim. Instead each test asserts
invariants that hold regardless of which tier actually succeeds:
`route_extract_triple` always returns a well-formed `Triple` or raises after
exhausting all attempts, and the ledger it leaves behind is always
internally consistent (small-tier attempts, if any, precede large-tier
attempts for the same call; the call only returns once some attempt
succeeded). The "hard" fact below (multi-clause, several candidate
relations) is chosen to make small-model escalation *likely* without
depending on it.
"""

from __future__ import annotations

import pytest

from core import router

pytestmark = pytest.mark.integration


def _extract_records_since(start_index: int) -> list[router.CallRecord]:
    return [r for r in router._ledger[start_index:] if r.task == "extract"]


@pytest.mark.asyncio
async def test_simple_fact_extracts_or_exhausts_the_full_cascade():
    """Local 3B/gemma4 extraction is genuinely unreliable -- even a trivial
    fact and the *full* 5-attempt cascade (2 small + up to
    `MAX_EXTRACT_ATTEMPTS` large) can fail outright and raise `RuntimeError`.
    That's an honest, observed outcome of this design (see PROGRESS.md), not
    a bug to paper over -- this test accepts either outcome and checks the
    invariant that actually matters: on success, `subject` is populated; on
    total failure, every attempt across both tiers was logged.
    """
    router.reset_ledger()

    try:
        triple = await router.route_extract_triple("Alice lives in Boston.")
    except RuntimeError:
        records = _extract_records_since(0)
        assert len(records) == 2 + router.MAX_EXTRACT_ATTEMPTS
        assert all(r.succeeded is False for r in records)
        return

    assert triple.subject.strip()
    records = _extract_records_since(0)
    assert records, "expected at least one extract CallRecord"
    assert records[-1].succeeded is True


@pytest.mark.asyncio
async def test_cascade_tries_small_tier_before_large_tier():
    """A hard, multi-clause fact is likely (not guaranteed) to make the
    small model stumble, exercising escalation. Real extraction quality on
    a sentence this dense can be poor even on success -- a non-primary
    field like `relation` may legitimately come back empty rather than
    semantically wrong -- so this only asserts the cascade's *ordering*
    invariant, not the semantic content of a successful result.
    """
    router.reset_ledger()

    hard_fact = (
        "Alice, who works as a senior software engineer at a large tech "
        "company in Boston, mentioned during our call yesterday that she "
        "also volunteers at an animal shelter on weekends and recently "
        "adopted a rescue dog named Max."
    )
    try:
        await router.route_extract_triple(hard_fact)
    except RuntimeError:
        pass

    records = _extract_records_since(0)
    assert records, "expected at least one extract CallRecord"

    # Whatever happened, every small-tier attempt must be logged before any
    # large-tier attempt (the cascade tries cheap first, always) -- and the
    # attempt numbers must be strictly increasing.
    tiers = [r.model_tier for r in records]
    if "large" in tiers:
        first_large = tiers.index("large")
        assert all(t == "small" for t in tiers[:first_large])
    assert [r.attempt for r in records] == sorted(r.attempt for r in records)


@pytest.mark.asyncio
async def test_judge_conflict_returns_a_valid_verdict():
    verdict = await router.judge_conflict(
        "Alice lives in Boston.", "Alice lives in Boston, Massachusetts."
    )
    assert verdict in ("update", "same", "distinct")


@pytest.mark.asyncio
async def test_judge_conflict_logs_a_call_record():
    router.reset_ledger()
    await router.judge_conflict("Alice's favorite color is blue.", "Alice's favorite color is red.")

    records = [r for r in router._ledger if r.task == "conflict_judge"]
    assert len(records) == 1
    assert records[0].model_tier == "large"


@pytest.mark.asyncio
async def test_summarize_fact_returns_a_nonempty_sentence():
    summary = await router.summarize_fact(
        "Alice moved to Seattle in 2024 to take a new job at a startup."
    )
    assert summary.strip()
    assert len(summary) < 500  # a "one concise sentence" summary, not a wall of text


@pytest.mark.asyncio
async def test_summarize_fact_logs_a_call_record():
    router.reset_ledger()
    await router.summarize_fact("Bob's favorite color is green.")

    records = [r for r in router._ledger if r.task == "summarize"]
    assert len(records) == 1
    assert records[0].model_tier == "large"
    assert records[0].succeeded is True
