"""Stdio MCP server exposing the memory-layer pipeline (Bucket 6).

Wraps Buckets 1–5 as four MCP tools over stdio transport so a coding agent
(Claude Code, Antigravity, Codex, ...) can call the memory layer directly.
The *same binary* serves every client with no code changes between them --
that's the spec's definition of done; clients differ only in a few lines of
registration config (see mcp_server/REGISTRATION.md).

Honest constraints, stated up front:

- **No auth, by design.** The spec pins this to local stdio only -- there's
  no network boundary to authenticate across, so there's deliberately no
  Firebase/Supabase/token layer. Anything that can spawn this process can
  call it; that's the intended trust model for a v1 local memory layer.
- **Single active dataset.** `memory_write` / `memory_forget` rebuild
  Cognee's index via `store.reset_and_load`, which wipes Cognee's *entire*
  configured data root, not just one dataset -- identical to the Bucket 1–5
  constraint. Callers should treat one dataset as active at a time; the
  `dataset` args exist for the resolver's own SQLite scoping, not for
  concurrent multi-dataset Cognee indexes.
- **Write/forget are serialized under two locks.** Both call the full
  reset+cognify cycle; two overlapping calls would race on that shared data
  root. `_pipeline_lock` (asyncio) makes them mutually exclusive *within*
  this process; `core.locks.pipeline_lock` (flock) extends that across
  processes, since every MCP client session spawns its own server process
  against the same data root. Reads (`memory_search`, `memory_stats`)
  don't take either -- they only read.

The four tool bodies are plain `async def` functions registered with
`@mcp.tool()` (which returns the original callable in mcp >= 1.x), so the
test suite imports and calls them directly without a client round-trip.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime

from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from mcp.server.fastmcp import FastMCP

from core import consolidation, ingest, resolver, router
from core.config import REPO_ROOT
from core.locks import pipeline_lock

# memory_search's k, clamped: 0/negative make no sense, and an enormous k is
# just a slow no-op against Cognee -- bound it rather than pass garbage through.
MAX_SEARCH_K = 50

RESULTS_DIR = REPO_ROOT / "benchmark" / "results"

mcp = FastMCP("memory-layer")

# Serializes the two tools that call store.reset_and_load (full reset+cognify
# of the shared Cognee data root). See module docstring.
_pipeline_lock = asyncio.Lock()


def _fact_summary(fact: resolver.Fact | None) -> dict | None:
    """Compact, JSON-serializable view of a Fact for tool responses."""
    if fact is None:
        return None
    return {
        "id": fact.id,
        "subject": fact.subject,
        "relation": fact.relation,
        "object": fact.object,
    }


@mcp.tool()
async def memory_write(content: str, metadata: dict | None = None) -> dict:
    """Store a fact. Extracts a triple, versions it bi-temporally, reindexes.

    `metadata` is optional and may carry:
      - `dataset` (str): resolver dataset to write into (default "main_dataset").
      - `event_time` (str, ISO-8601): when the fact became true, for the
        bi-temporal model (default: now).
      - `importance` (float): accepted but currently ignored -- it isn't yet
        threaded through `ingest.remember`, so this is honest about not
        pretending to support it rather than silently dropping it unnoticed.

    Returns the extracted triple plus what the conflict resolver did:
    whether it changed anything, what (if anything) it superseded, and
    whether it revived a previously-evicted fact or judged an ambiguous
    conflict as distinct.
    """
    metadata = metadata or {}
    dataset = metadata.get("dataset", "main_dataset")
    event_time = metadata.get("event_time")
    if event_time is not None:
        # Validate before any LLM call: a malformed timestamp would otherwise
        # land in the resolver's bi-temporal columns as-is and quietly corrupt
        # version ordering. Fail loud and cheap instead.
        try:
            datetime.fromisoformat(str(event_time))
        except ValueError as exc:
            raise ValueError(
                f"metadata.event_time must be an ISO-8601 timestamp, got {event_time!r}"
            ) from exc

    async with _pipeline_lock, pipeline_lock():
        result = await ingest.remember(content, dataset=dataset, event_time=event_time)

    write = result.write
    return {
        "subject": result.triple.subject,
        "relation": result.triple.relation,
        "object": result.triple.object,
        "changed": write.changed,
        "superseded": _fact_summary(write.superseded),
        "revived": write.revived_from_eviction,
        "distinct": write.distinct,
        "cognify_attempts": result.cognify_attempts,
    }


@mcp.tool()
async def memory_search(query: str, k: int = 5, dataset: str = "main_dataset") -> dict:
    """Search stored memories. Records access, which feeds tiering (Bucket 4).

    Uses the consolidation-aware search path (not raw Cognee search), so a
    retrieved fact's access stats are bumped -- retrieval keeps a memory
    "warm" and resistant to eviction, exactly as a real memory system should.

    Returns two views of the same ranked results, index-aligned:
      - `hits`: the ranked hit texts (unchanged legacy shape).
      - `hit_facts`: one object per hit, `{id, subject, relation, object,
        text}`. The `id` is what `memory_forget` needs, so a client can
        search and then forget a specific result without any out-of-band id
        lookup. Cognee doesn't carry resolver ids through its own pipeline,
        so we recover them by exact-text match against current facts (the
        index stores exactly `Fact.text`, 1:1). If a hit text can't be
        mapped back -- an anomaly, not the norm -- its `id` is `null` and the
        client simply can't forget that one by id; we surface that honestly
        rather than guess.

    An empty index (e.g. every fact evicted, or nothing written yet) raises
    Cognee's `NoDataError` rather than returning no hits -- for an
    agent-facing tool that's just "no results," so it's caught and reported
    as empty lists, not surfaced as a tool error.
    """
    k = max(1, min(int(k), MAX_SEARCH_K))
    try:
        hits = await consolidation.search(query, dataset=dataset, k=k)
    except NoDataError:
        return {"query": query, "hits": [], "hit_facts": []}

    texts = [consolidation._hit_text(h) for h in hits]
    by_text = {f.text: f for f in resolver.current_facts(dataset)}
    hit_facts = []
    for text in texts:
        summary = _fact_summary(by_text.get(text)) or {
            "id": None,
            "subject": None,
            "relation": None,
            "object": None,
        }
        hit_facts.append({**summary, "text": text})
    return {"query": query, "hits": texts, "hit_facts": hit_facts}


@mcp.tool()
async def memory_stats(dataset: str = "main_dataset") -> dict:
    """Live pipeline + cost numbers, plus the latest saved benchmark results.

    This is the payload the Bucket 7 dashboard renders. Combines three
    sources: current fact counts by tier (resolver), the cost ledger
    (router -- reference pricing, not real spend; see `pricing_note`), and
    whatever benchmark result JSONs have been written to
    benchmark/results/ (baseline / conflict / forgetting / cost).
    """
    facts = resolver.current_facts(dataset)
    by_tier = {"hot": 0, "warm": 0, "cold": 0}
    for f in facts:
        by_tier[f.tier] = by_tier.get(f.tier, 0) + 1

    benchmarks: dict[str, dict] = {}
    if RESULTS_DIR.exists():
        for path in sorted(RESULTS_DIR.glob("*.json")):
            try:
                benchmarks[path.stem] = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue  # a half-written or unreadable result file shouldn't sink stats

    return {
        "dataset": dataset,
        "facts": {"total_current": len(facts), "by_tier": by_tier},
        "cost": router.cost_report(),
        "benchmarks": benchmarks,
    }


@mcp.tool()
async def memory_forget(id: int, dataset: str = "main_dataset") -> dict:
    """Evict a fact by id and drop it from the search index.

    The `id` comes from `memory_search`'s `hit_facts[*].id` -- that's how a
    client discovers what to forget without an out-of-band lookup.

    Looks the id up among currently-valid facts first: a miss returns
    `{found: false}` and touches nothing (no Cognee call). A hit evicts the
    fact (kept in SQLite for regret analysis, stamped `evicted_at`) and
    rebuilds the active index from the survivors so it's actually
    unsearchable afterwards.
    """
    target = next((f for f in resolver.current_facts(dataset) if f.id == id), None)
    if target is None:
        return {"found": False, "evicted_id": None}

    async with _pipeline_lock, pipeline_lock():
        resolver.evict(id, dataset=dataset)
        await consolidation.resync_active(dataset)

    remaining = resolver.current_facts(dataset)
    return {"found": True, "evicted_id": id, "remaining_current": len(remaining)}


def main() -> None:
    """Entry point: run the server on stdio (blocks until the client detaches).

    MCP's stdio transport carries the JSON-RPC wire on **stdout**, so anything
    else that writes to stdout corrupts it. Cognee's DB layer emits chatter to
    stdout during `cognify()` (e.g. "table already exists, skipping creation"),
    which a real stdio client rejects as invalid JSON-RPC -- a failure the
    in-process tests can't surface. Fix: hand the MCP writer a *private*
    duplicate of the real stdout, then point the process's own stdout at
    stderr, both at the fd level (`dup2(2, 1)` catches raw/handler writes to
    fd 1 regardless of any cached stream reference) and the Python level
    (`sys.stdout = sys.stderr` catches `print`). Only the protocol writer keeps
    a path to the real stdout.
    """
    import anyio
    from io import TextIOWrapper

    from mcp.server.stdio import stdio_server

    wire_fd = os.dup(1)  # private handle on the real stdout, for JSON-RPC only
    os.dup2(2, 1)  # fd 1 -> stderr: nothing else can reach the wire by accident
    sys.stdout = sys.stderr  # print()/sys.stdout.write -> stderr

    async def _run() -> None:
        out = anyio.wrap_file(TextIOWrapper(os.fdopen(wire_fd, "wb"), encoding="utf-8"))
        in_ = anyio.wrap_file(
            TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
        )
        async with stdio_server(stdin=in_, stdout=out) as (read_stream, write_stream):
            await mcp._mcp_server.run(
                read_stream,
                write_stream,
                mcp._mcp_server.create_initialization_options(),
            )

    anyio.run(_run)


if __name__ == "__main__":
    main()
