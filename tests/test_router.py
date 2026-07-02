"""Unit tests for core/router.py -- cost-report math over the call ledger.

No Ollama, no Cognee, no network. Exercises `cost_report()`'s aggregation
logic directly against synthetic `CallRecord`s appended to `router._ledger`,
rather than driving it through a real LLM call -- that's what
tests/test_router_integration.py is for. Every test resets the ledger first
(and the autouse fixture resets it again after), so these tests can run in
any order without polluting each other.
"""

from __future__ import annotations

from core import router


def _record(**overrides) -> router.CallRecord:
    defaults = dict(
        task="extract",
        model_tier="small",
        prompt_tokens=0,
        completion_tokens=0,
        attempt=1,
        succeeded=True,
        latency_s=0.1,
    )
    defaults.update(overrides)
    return router.CallRecord(**defaults)


def setup_function(_fn) -> None:
    router.reset_ledger()


def teardown_function(_fn) -> None:
    router.reset_ledger()


def test_reset_ledger_clears_state():
    router._ledger.append(_record())
    router.reset_ledger()

    report = router.cost_report()

    assert report["total_calls"] == 0
    assert report["by_task"] == {}
    assert report["total_cost_usd"] == 0.0
    assert report["successful_extracts"] == 0
    assert report["usd_per_1000_memories"] is None


def test_cost_report_aggregates_by_task():
    router._ledger.append(_record(task="extract", succeeded=True))
    router._ledger.append(_record(task="extract", succeeded=False))
    router._ledger.append(_record(task="conflict_judge", model_tier="large", succeeded=True))
    router._ledger.append(_record(task="summarize", model_tier="large", succeeded=True))

    report = router.cost_report()

    assert report["total_calls"] == 4
    assert report["by_task"]["extract"] == {"calls": 2, "successes": 1}
    assert report["by_task"]["conflict_judge"] == {"calls": 1, "successes": 1}
    assert report["by_task"]["summarize"] == {"calls": 1, "successes": 1}


def test_cost_report_aggregates_by_tier():
    router._ledger.append(
        _record(model_tier="small", prompt_tokens=100, completion_tokens=20, succeeded=True)
    )
    router._ledger.append(
        _record(model_tier="small", prompt_tokens=50, completion_tokens=0, succeeded=False)
    )
    router._ledger.append(
        _record(
            task="summarize",
            model_tier="large",
            prompt_tokens=200,
            completion_tokens=40,
            succeeded=True,
        )
    )

    report = router.cost_report()

    small = report["by_tier"]["small"]
    large = report["by_tier"]["large"]
    assert small == {"calls": 2, "successes": 1, "prompt_tokens": 150, "completion_tokens": 20}
    assert large == {"calls": 1, "successes": 1, "prompt_tokens": 200, "completion_tokens": 40}


def test_total_cost_uses_price_per_tier():
    # 1,000,000 small-tier tokens -> exactly $0.10 at _PRICE_PER_M_TOKENS["small"].
    router._ledger.append(
        _record(model_tier="small", prompt_tokens=800_000, completion_tokens=200_000)
    )
    # 500,000 large-tier tokens -> exactly $0.10 at _PRICE_PER_M_TOKENS["large"] (0.20/M).
    router._ledger.append(
        _record(
            task="summarize",
            model_tier="large",
            prompt_tokens=400_000,
            completion_tokens=100_000,
        )
    )

    report = router.cost_report()

    assert report["total_cost_usd"] == 0.2


def test_failed_calls_still_counted_toward_cost():
    # A failed call still burned tokens (e.g. a malformed-but-billed completion);
    # cost_report must count it, not just successes.
    router._ledger.append(
        _record(model_tier="small", prompt_tokens=1_000_000, completion_tokens=0, succeeded=False)
    )

    report = router.cost_report()

    assert report["total_cost_usd"] == 0.1
    assert report["by_tier"]["small"]["successes"] == 0
    assert report["by_tier"]["small"]["calls"] == 1


def test_usd_per_1000_memories_computed_from_successful_extracts():
    # Two successful extracts, $0.02 total cost -> $10/1000 memories.
    router._ledger.append(
        _record(task="extract", model_tier="small", prompt_tokens=100_000, succeeded=True)
    )
    router._ledger.append(
        _record(task="extract", model_tier="small", prompt_tokens=100_000, succeeded=True)
    )

    report = router.cost_report()

    assert report["successful_extracts"] == 2
    assert report["usd_per_1000_memories"] == round(report["total_cost_usd"] / 2 * 1000, 4)


def test_usd_per_1000_memories_none_without_successful_extracts():
    router._ledger.append(_record(task="extract", succeeded=False))
    router._ledger.append(_record(task="conflict_judge", model_tier="large", succeeded=True))

    report = router.cost_report()

    assert report["successful_extracts"] == 0
    assert report["usd_per_1000_memories"] is None


def test_multiple_extract_attempts_on_same_fact_count_once_toward_successful_extracts():
    # route_extract_triple logs one failed small-model attempt then one
    # successful large-model attempt for a single fact -- cost_report should
    # count that as one successful extract, not two.
    router._ledger.append(_record(task="extract", model_tier="small", attempt=1, succeeded=False))
    router._ledger.append(
        _record(task="extract", model_tier="large", attempt=2, succeeded=True, prompt_tokens=500)
    )

    report = router.cost_report()

    assert report["successful_extracts"] == 1
    assert report["by_task"]["extract"]["calls"] == 2
