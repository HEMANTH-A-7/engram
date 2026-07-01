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

Run: `uv run python -m benchmark.harness`
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from pathlib import Path

from cognee.modules.search.types import SearchType

from benchmark.eval_set import EVAL_CASES
from core import store
from core.config import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "benchmark" / "results"
K_VALUES = (1, 3, 5)


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
    await store.reset()

    all_facts = [fact for case in EVAL_CASES for fact in case.facts]
    for fact in all_facts:
        await store.add(fact, dataset=dataset)
    await store.cognify(dataset=dataset)

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
        "recall": recall,
        "latency": latency,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
