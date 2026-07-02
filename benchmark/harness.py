"""Benchmark harness v1: recall@k and retrieval latency over the baseline pipeline.

Runs the eval set (benchmark/eval_set.py) end-to-end against vanilla Cognee
(core/store.py): reset -> add all facts -> cognify -> search each eval query
with raw CHUNKS retrieval, then computes recall@{1,3,5} and latency p50/p95.
Results are written to benchmark/results/<label>.json so later buckets
(conflict resolver, consolidation, cost router) can diff before/after numbers
against this baseline.

Conflict-resolution accuracy and $ cost/query are intentionally not measured
here — those need components that don't exist yet (Buckets 3 and 5). This is
the "minimal" v1 the build-order decision (D3, PROGRESS.md) calls for.

Cognee's Ollama adapter hardcodes 2 structured-output retries (not
configurable) and batches every fact into one `cognify()` pipeline run, so a
single flaky extraction (e.g. gemma4 emitting a null `description`) aborts the
whole run. `cognify()` is retried a bounded number of times at the harness
level to absorb that — logged, not masked, via `result["cognify_attempts"]`.

`run_conflict_eval()` (Bucket 3) adds conflict-resolution-accuracy: for each
`CONFLICT_CASES` case, an ordered sequence of updates to the same fact is
run two ways — through `core/resolver.py` (supersede-not-overwrite, only the
current value ever reaches Cognee's index) and through naive cumulative
`core/store.py` adds (every version stays in the index, unresolved) — and the
"does the top hit reflect the current value" accuracy is compared between
the two. $ cost/query is intentionally still not measured (needs Bucket 5).

Run: `uv run python -m benchmark.harness`
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from pathlib import Path

from cognee.modules.search.types import SearchType

from benchmark.eval_set import CONFLICT_CASES, EVAL_CASES
from core import ingest, resolver, store
from core.config import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "benchmark" / "results"
K_VALUES = (1, 3, 5)
MAX_COGNIFY_ATTEMPTS = 3


def _hit_text(hit) -> str:
    if isinstance(hit, dict):
        return str(hit.get("text") or hit.get("name") or hit)
    return str(getattr(hit, "text", None) or hit)


def _is_relevant(hit_text: str, expected: tuple[str, ...]) -> bool:
    lowered = hit_text.lower()
    return any(s.lower() in lowered for s in expected)


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (pct / 100) * (len(sorted_vals) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


async def run(dataset: str = "benchmark_dataset", label: str = "baseline") -> dict:
    """Reset storage, ingest the eval set, and measure recall@k + latency."""
    all_facts = [fact for case in EVAL_CASES for fact in case.facts]
    cognify_attempts = await store.reset_and_load(dataset, all_facts, MAX_COGNIFY_ATTEMPTS)

    max_k = max(K_VALUES)
    hits_at_k = {k: 0 for k in K_VALUES}
    latencies_s: list[float] = []

    for case in EVAL_CASES:
        start = time.perf_counter()
        results = await store.search(
            case.query, k=max_k, query_type=SearchType.CHUNKS, dataset=dataset
        )
        latencies_s.append(time.perf_counter() - start)

        texts = [_hit_text(h) for h in results]
        for k in K_VALUES:
            if any(_is_relevant(t, case.expected_substrings) for t in texts[:k]):
                hits_at_k[k] += 1

    n = len(EVAL_CASES)
    recall = {f"recall@{k}": round(hits_at_k[k] / n, 4) for k in K_VALUES}

    latencies_ms = sorted(s * 1000 for s in latencies_s)
    latency = {
        "p50_ms": round(_percentile(latencies_ms, 50), 2),
        "p95_ms": round(_percentile(latencies_ms, 95), 2),
        "mean_ms": round(statistics.mean(latencies_ms), 2),
    }

    return {
        "label": label,
        "n_cases": n,
        "n_facts": len(all_facts),
        "cognify_attempts": cognify_attempts,
        "recall": recall,
        "latency": latency,
    }


async def _query_top_hits(query: str, dataset: str, k: int = 3) -> list[str]:
    hits = await store.search(query, k=k, query_type=SearchType.CHUNKS, dataset=dataset)
    return [_hit_text(h) for h in hits]


async def run_conflict_eval(dataset: str = "conflict_dataset", label: str = "conflict") -> dict:
    """Compare resolver-mode vs. naive-overwrite on CONFLICT_CASES.

    Resolver mode: each update goes through `core/resolver.py` (via
    `ingest.remember(..., sync=False)`, one `ingest.resync()` per case) so
    only the currently-valid fact's text ever reaches Cognee's index. Naive
    baseline: every update's raw text is added cumulatively via
    `core/store.py` with no resolution at all — old and new both stay
    indexed, exactly Bucket 1's baseline behavior.

    Accuracy = fraction of cases where the top-ranked hit reflects the
    current value. `stale_leak_rate` = fraction of cases where a superseded
    value still shows up anywhere in the top-3 hits (diagnostic).
    """
    resolver_correct = 0
    resolver_leaked = 0
    naive_correct = 0
    naive_leaked = 0

    for case in CONFLICT_CASES:
        resolver_dataset = f"{dataset}_{case.id}_resolver"
        resolver.reset(resolver_dataset)
        for text in case.updates:
            await ingest.remember(text, dataset=resolver_dataset, sync=False)
        await ingest.resync(resolver_dataset)

        top_hits = await _query_top_hits(case.query, resolver_dataset)
        if top_hits and _is_relevant(top_hits[0], case.current_substrings):
            resolver_correct += 1
        if any(_is_relevant(t, case.stale_substrings) for t in top_hits):
            resolver_leaked += 1

        naive_dataset = f"{dataset}_{case.id}_naive"
        await store.reset_and_load(naive_dataset, list(case.updates))

        top_hits = await _query_top_hits(case.query, naive_dataset)
        if top_hits and _is_relevant(top_hits[0], case.current_substrings):
            naive_correct += 1
        if any(_is_relevant(t, case.stale_substrings) for t in top_hits):
            naive_leaked += 1

    n = len(CONFLICT_CASES)
    return {
        "label": label,
        "n_cases": n,
        "resolver": {
            "accuracy": round(resolver_correct / n, 4),
            "stale_leak_rate": round(resolver_leaked / n, 4),
        },
        "naive_overwrite": {
            "accuracy": round(naive_correct / n, 4),
            "stale_leak_rate": round(naive_leaked / n, 4),
        },
    }


def save(result: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{result['label']}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path


async def main() -> int:
    result = await run()
    path = save(result)
    print(json.dumps(result, indent=2))
    print(f"\nWrote {path}")

    conflict_result = await run_conflict_eval()
    conflict_path = save(conflict_result)
    print(json.dumps(conflict_result, indent=2))
    print(f"\nWrote {conflict_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
