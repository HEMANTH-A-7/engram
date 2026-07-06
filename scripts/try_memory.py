"""Interactive-ish CLI to test the memory layer by hand and watch the stats move.

Writes to `main_dataset` -- the same dataset the dashboard's live header reads
(GET /api/stats) -- so after `remember` you can refresh http://127.0.0.1:8010
and watch "Current facts" / tier counts change. Runs the full real pipeline
(cost-aware extraction -> bi-temporal resolver -> Cognee index), 100% offline.

Usage (each is one command; extraction takes ~10-60s on CPU Ollama):

    uv run python scripts/try_memory.py remember "Alice lives in Boston."
    uv run python scripts/try_memory.py remember "Alice now lives in Seattle."   # supersedes
    uv run python scripts/try_memory.py search   "Where does Alice live?"
    uv run python scripts/try_memory.py stats
    uv run python scripts/try_memory.py forget   <id>       # id from a search hit
    uv run python scripts/try_memory.py reset               # wipe main_dataset

`stats` calls the exact same function the dashboard does, so the two always agree.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Skip Cognee's 30s Ollama pre-flight and keep everything offline before imports.
os.environ.setdefault("COGNEE_SKIP_CONNECTION_TEST", "true")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import ingest, resolver  # noqa: E402
from mcp_server import server  # noqa: E402  (memory_stats/forget = dashboard's own source)

DATASET = "main_dataset"


async def cmd_remember(text: str) -> None:
    result = await ingest.remember(text, dataset=DATASET)
    write = result.write
    f = write.fact
    print(f"remembered #{f.id}: ({f.subject!r}, {f.relation!r}, {f.object!r})")
    if write.superseded is not None:
        print(f"  -> superseded #{write.superseded.id} ({write.superseded.object!r})")
    elif not write.changed:
        print("  -> no change (restates the already-current fact)")


async def cmd_search(query: str) -> None:
    out = await server.memory_search(query, k=5, dataset=DATASET)
    if not out["hits"]:
        print("(no hits)")
        return
    for hf in out["hit_facts"]:
        print(f"  #{hf['id']}  {hf['text']}")


async def cmd_stats() -> None:
    st = await server.memory_stats(DATASET)
    facts, cost = st["facts"], st["cost"]
    print(f"current facts: {facts['total_current']}  "
          f"(hot {facts['by_tier']['hot']} / warm {facts['by_tier']['warm']} "
          f"/ cold {facts['by_tier']['cold']})")
    print(f"extract LLM calls so far this process: {cost['total_calls']}")


async def cmd_forget(fact_id: int) -> None:
    out = await server.memory_forget(fact_id, dataset=DATASET)
    print(json.dumps(out))


def cmd_reset() -> None:
    resolver.reset(DATASET)
    print(f"reset {DATASET}")


def main() -> None:
    p = argparse.ArgumentParser(description="Test the memory layer by hand.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("remember").add_argument("text")
    sub.add_parser("search").add_argument("query")
    sub.add_parser("stats")
    sub.add_parser("forget").add_argument("id", type=int)
    sub.add_parser("reset")
    args = p.parse_args()

    if args.cmd == "remember":
        asyncio.run(cmd_remember(args.text))
    elif args.cmd == "search":
        asyncio.run(cmd_search(args.query))
    elif args.cmd == "stats":
        asyncio.run(cmd_stats())
    elif args.cmd == "forget":
        asyncio.run(cmd_forget(args.id))
    elif args.cmd == "reset":
        cmd_reset()


if __name__ == "__main__":
    main()
