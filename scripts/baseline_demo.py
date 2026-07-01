"""Bucket 1 demo: prove the baseline Cognee pipeline works end-to-end, offline.

Ingests a handful of facts, cognifies them into a graph, then runs both a raw
retrieval query (CHUNKS) and an LLM-answered query (GRAPH_COMPLETION).

Run: `uv run python scripts/baseline_demo.py`
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognee.modules.search.types import SearchType  # noqa: E402

from core import store  # noqa: E402

FACTS = [
    "Ada Lovelace wrote the first algorithm intended for a machine, Babbage's Analytical Engine.",
    "Alan Turing proposed the Turing test in his 1950 paper 'Computing Machinery and Intelligence'.",
    "Grace Hopper developed the first compiler and popularized the term 'debugging'.",
    "The Python programming language was created by Guido van Rossum and first released in 1991.",
]


async def main() -> int:
    print("Resetting store for a clean baseline...")
    await store.reset()

    print(f"Ingesting {len(FACTS)} facts...")
    for fact in FACTS:
        await store.add(fact)

    print("Cognifying (building graph + embeddings via Ollama)...")
    await store.cognify()

    print("\n--- CHUNKS retrieval: 'Who created Python?' (top 3) ---")
    hits = await store.search("Who created Python?", k=3, query_type=SearchType.CHUNKS)
    for i, hit in enumerate(hits, 1):
        if isinstance(hit, dict):
            text = hit.get("text") or hit.get("name") or str(hit)
        else:
            text = getattr(hit, "text", None) or str(hit)
        print(f"  {i}. {text[:120]}")

    print("\n--- GRAPH_COMPLETION: 'Who popularized the term debugging?' ---")
    answer = await store.search(
        "Who popularized the term debugging?",
        k=5,
        query_type=SearchType.GRAPH_COMPLETION,
    )
    for a in answer:
        print(f"  {a}")

    print("\nBaseline pipeline: DONE ✅ (offline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
