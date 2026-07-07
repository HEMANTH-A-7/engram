"""Measure the pivot's two headline metrics (Session 9).

1. **Write latency, before vs after.** "After" = `notes.add_note` (the
   memory_note path: one real embed call + SQLite) against a throwaway tmp
   DB, N times, p50/p95. "Before" (--with-before) = the pre-pivot single-fact
   write path, `store.reset_and_load(graph=False)` into an isolated tmp
   Cognee data root — the same ~3.2s-per-fact path Session 6 measured. It is
   slow and needs Ollama warm; skip it and the script cites Session 6's
   measured number, labeled as such.

2. **Digest tokens vs raw history** on REAL data: `context.build_context`
   over the live dataset (default `main_dataset`), read-only apart from the
   idempotent index backfill. Token figures are the project's documented
   ~4 chars/token estimates.

Run on the machine with Ollama up:

    uv run python scripts/measure_pivot.py --n 20
    uv run python scripts/measure_pivot.py --n 20 --with-before   # full before/after

Results print to stdout and land in benchmark/results/pivot.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SESSION6_BEFORE_S = 3.2  # measured 2026-07-03, single-fact reset_and_load(graph=False)


def measure_note_writes(n: int) -> dict:
    """N real memory_note writes (real embed calls) into a throwaway tmp DB."""
    from core import notes, resolver

    with tempfile.TemporaryDirectory() as tmp:
        resolver.configure(db_path=Path(tmp) / "facts.db")
        latencies: list[float] = []
        embedded_ok = 0
        for i in range(n):
            text = f"benchmark note {i}: measured write latency of the pivot's fast path."
            t0 = time.perf_counter()
            result = notes.add_note(text, kind="progress", dataset="pivot_bench")
            latencies.append(time.perf_counter() - t0)
            embedded_ok += int(result.embedded)
    # Restore the default DB: the tmp dir above is gone once this block exits,
    # and measure_digest() runs after us against the real dataset.
    resolver.configure()
    latencies.sort()
    return {
        "n": n,
        "embedded_ok": embedded_ok,  # n means Ollama was up for every write
        "p50_s": round(statistics.median(latencies), 4),
        "p95_s": round(latencies[max(0, int(len(latencies) * 0.95) - 1)], 4),
        "max_s": round(latencies[-1], 4),
        "note": "memory_note path: one embed HTTP call + SQLite. Sub-second target.",
    }


def measure_before_write() -> dict:
    """Pre-pivot single-fact write: reset_and_load into an isolated tmp root."""
    from core import store

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        store.configure(data_dir=tmp_path / "data", system_dir=tmp_path / "system")
        t0 = time.perf_counter()
        attempts = asyncio.run(
            store.reset_and_load("pivot_before_bench", ["Benchmark fact for timing."])
        )
        elapsed = time.perf_counter() - t0
    return {
        "single_fact_write_s": round(elapsed, 2),
        "cognify_attempts": attempts,
        "note": "Pre-pivot path: full reset + embed-only cognify (isolated tmp root).",
    }


def measure_digest(dataset: str) -> dict:
    from core import index
    from core.context import build_context

    index.backfill_facts(dataset)  # idempotent pre-pivot migration
    result = build_context(token_budget=2000, dataset=dataset)
    return {k: v for k, v in result.items() if k != "digest"} | {
        "digest_preview": result["digest"][:400]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20, help="note writes to time")
    parser.add_argument("--dataset", default="main_dataset", help="digest dataset")
    parser.add_argument(
        "--with-before",
        action="store_true",
        help="also time the slow pre-pivot write path (needs Ollama warm)",
    )
    args = parser.parse_args()

    report = {
        "write_latency_after": measure_note_writes(args.n),
        "write_latency_before": (
            measure_before_write()
            if args.with_before
            else {
                "single_fact_write_s": SESSION6_BEFORE_S,
                "note": (
                    "Not re-run (pass --with-before). Value is Session 6's real "
                    "measurement of the same path (2026-07-03), cited not remeasured."
                ),
            }
        ),
        "digest_vs_raw": measure_digest(args.dataset),
        "meta": {
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "note": "Token figures are ~4 chars/token estimates (offline-safe).",
        },
    }

    out = REPO_ROOT / "benchmark" / "results" / "pivot.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    before = report["write_latency_before"]["single_fact_write_s"]
    after = report["write_latency_after"]["p50_s"]
    if after > 0:
        print(
            f"\nwrite p50: {after}s vs {before}s before → ~{before / after:.0f}x faster",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
