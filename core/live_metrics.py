"""Live, read-only metrics computed from the resolver's real fact history.

These are the honest "raw history vs. memory layer" numbers the dashboard
renders for the *actual* dataset the user has populated (e.g. `main_dataset`),
replacing the fixed benchmark corpus in `benchmark/eval_set.py`. Everything here
reads straight from `core.resolver` -- it never writes, cognifies, or resets, so
it is safe to recompute on every dashboard poll without touching live data.

The token- and storage-savings formulas mirror `benchmark/harness.py`'s
`run_tokens_eval` / `run_storage_growth_eval` exactly; the only difference is the
data source: real `resolver.all_versions()` history instead of synthetically
written `_VERSIONED_FACTS`. `_estimate_tokens` lives here (the harness imports it
back) so both paths share one estimator.
"""

from __future__ import annotations

from core import resolver


def _estimate_tokens(text: str) -> int:
    """Rough token count: ~4 chars/token. A deliberate *estimate*, not a real
    tokenizer -- `tiktoken`'s vocab downloads on first use, which would break
    this project's offline guarantee, and the memory-layer's value shows up in
    the raw-vs-extended *ratio*, which a consistent estimator captures fine.
    Documented as an estimate the same way the cost router's pricing is
    documented as reference-not-real.
    """
    return round(len(text) / 4)


def token_savings(dataset: str = "main_dataset") -> dict:
    """Tokens saved vs. dumping raw history into an agent's context.

    `raw_history_tokens` = every version ever written (the no-memory-layer
    baseline); `extended_tokens` = only the current, non-superseded facts the
    layer would actually surface. The gap is what supersede-not-overwrite saves.
    """
    history = resolver.all_versions(dataset)
    current = resolver.current_facts(dataset)

    raw_history_tokens = sum(_estimate_tokens(f.text) for f in history)
    extended_tokens = sum(_estimate_tokens(f.text) for f in current)
    tokens_saved = raw_history_tokens - extended_tokens
    pct_saved = round(tokens_saved / raw_history_tokens, 4) if raw_history_tokens else 0.0

    return {
        "n_writes": len(history),
        "n_current_facts": len(current),
        "raw_history_tokens": raw_history_tokens,
        "extended_tokens": extended_tokens,
        "tokens_saved": tokens_saved,
        "pct_saved": pct_saved,
        "note": (
            "Live from this dataset's real history. Token counts are estimates "
            "(~4 chars/token, offline). raw_history = every version ever written "
            "(no-memory-layer baseline); extended = current non-superseded facts."
        ),
    }


def storage_series(dataset: str = "main_dataset") -> dict:
    """Storage footprint over time, replayed from real write history.

    Walks every version in insertion order. At step `i`: `without` = `i`
    (raw history keeps every write) and `with_` = how many of the facts the
    layer *currently retains* had appeared by then. The series ends exactly at
    `len(current_facts)`; the growing gap to `without` is superseded/evicted
    writes that never re-enter context.
    """
    history = resolver.all_versions(dataset)
    current = resolver.current_facts(dataset)
    retained_keys = {resolver.canonical_key(f.subject, f.relation) for f in current}

    series: list[dict] = []
    seen: set[str] = set()
    for i, fact in enumerate(history, start=1):
        key = resolver.canonical_key(fact.subject, fact.relation)
        if key in retained_keys:
            seen.add(key)
        series.append({"facts_ingested": i, "without": i, "with_": len(seen)})

    return {
        "n_writes": len(history),
        "series": series,
        "note": (
            "`without` = cumulative raw writes (no memory layer); `with_` = "
            "current retained facts that had appeared by that step. Divergence is "
            "supersede-not-overwrite compaction; ends at the live current-fact count."
        ),
    }


def revision_stats(dataset: str = "main_dataset") -> dict:
    """How much in-place revision work the resolver has actually done.

    `revisions` counts superseded versions -- real conflicts the layer resolved
    by versioning instead of duplicating (the "supersede-not-overwrite" thesis,
    measured on live data rather than a labeled corpus). `evicted` counts
    forgotten facts. `by_tier` is the current hot/warm/cold split.
    """
    history = resolver.all_versions(dataset)
    current = resolver.current_facts(dataset)

    revisions = sum(1 for f in history if f.superseded_by is not None)
    evicted = sum(1 for f in history if f.evicted_at is not None)

    by_tier = {"hot": 0, "warm": 0, "cold": 0}
    for f in current:
        by_tier[f.tier] = by_tier.get(f.tier, 0) + 1

    return {
        "revisions": revisions,
        "evicted": evicted,
        "current": len(current),
        "by_tier": by_tier,
        "note": (
            "Live counts. revisions = superseded versions (conflicts the resolver "
            "handled by versioning); evicted = forgotten facts; by_tier = current split."
        ),
    }


def live_report(dataset: str = "main_dataset") -> dict:
    """All live metrics for one dataset in a single payload for the dashboard."""
    return {
        "dataset": dataset,
        "tokens": token_savings(dataset),
        "storage": storage_series(dataset),
        "revisions": revision_stats(dataset),
    }
