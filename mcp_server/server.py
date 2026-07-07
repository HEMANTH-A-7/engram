"""Stdio MCP server: seamless dev-context memory for coding agents.

Post-pivot (Session 9) this serves SIX tools over stdio. The serving path is
the owned incremental index (`core/index.py`: SQLite FTS5 always, embeddings
fused in when Ollama answers) — writes are sub-second and per-project
datasets coexist because nothing destructive is ever global.

NO LOCAL LLM REQUIRED (engram-lite, Session 10): the calling agent supplies
the extracted triple to `memory_write` (it's a frontier LLM already — better
at extraction than any local model we ran); the local extraction cascade
survives only as an optional fallback behind a lazy import. Nothing in this
module imports Cognee or the LLM stack at startup, which also makes server
spawn fast. Cognee remains the engine for the graph/baseline benchmarks
only.

Tool map:
- `memory_note`     fast raw-text write (decision/progress/gotcha/reference)
- `memory_write`    triple-extraction path for discrete facts (bi-temporal
                    conflict resolution — the one tool that calls an LLM)
- `memory_context`  budgeted digest of current memory (session injection)
- `memory_search`   semantic search over facts + notes
- `memory_stats`    live counts + metrics
- `memory_forget`   evict a fact or note by id

Project keying: every tool accepts an explicit dataset or derives one from a
project directory (`core/projects.py`; env `MEMORY_PROJECT_DIR` as fallback,
`main_dataset` for full back-compat). The old single-active-dataset
constraint is GONE — `store.reset_and_load` is no longer on any serving
path, so no tool can wipe another project's index.

Honest constraints, still true:
- **No auth, by design** — local stdio only, no network boundary.
- **Writes serialized** under the same asyncio + flock pair as before. The
  destructive rationale is gone; what remains is cheap SQLite write
  serialization across concurrent agent sessions.
- **stdout is the JSON-RPC wire** — `main()` still hands the protocol a
  private dup of fd 1 and redirects everything else to stderr. Nothing may
  print to real stdout in this process.

Tool bodies are plain `async def`s registered with `@mcp.tool()` (which
returns the original callable), so tests import and call them directly.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime

from mcp.server.fastmcp import FastMCP

from core import context, index, notes, projects, resolver
from core.config import REPO_ROOT
from core.locks import pipeline_lock

MAX_SEARCH_K = 50

RESULTS_DIR = REPO_ROOT / "benchmark" / "results"

mcp = FastMCP("memory-layer")

# Serializes writers within this process; the flock in core.locks extends
# that across processes (each agent session spawns its own server).
_pipeline_lock = asyncio.Lock()


def _dataset(dataset: str | None, project_dir: str | None) -> str:
    """Explicit dataset wins; else derive from project_dir / env / default."""
    return dataset or projects.dataset_for(project_dir)


def _fact_summary(fact: resolver.Fact | None) -> dict | None:
    if fact is None:
        return None
    return {
        "id": fact.id,
        "subject": fact.subject,
        "relation": fact.relation,
        "object": fact.object,
    }


@mcp.tool()
async def memory_note(
    content: str,
    kind: str = "progress",
    project_dir: str | None = None,
    dataset: str | None = None,
    key: str | None = None,
) -> dict:
    """Record a development note the moment it happens — this is the DEFAULT
    way to save context, and it is fast (sub-second, no LLM).

    USE THIS PROACTIVELY, without being asked: after making or learning a
    design decision (kind="decision"), after completing or abandoning a piece
    of work (kind="progress"), immediately upon discovering a pitfall, bug
    cause, or non-obvious constraint (kind="gotcha"), and when a URL, path,
    or command is worth keeping (kind="reference"). If a session is about to
    run long, save progress EARLY — anything not recorded is lost when the
    context window compacts.

    The note is stored verbatim (no extraction) and is searchable seconds
    later via memory_search / memory_context. Passing the same `key` twice
    replaces the earlier note (rolling state) instead of accumulating.
    Provide `project_dir` (the project you are working in) so the note lands
    in that project's own memory.

    For a discrete, contradiction-prone fact ("the API key lives in X",
    "service Y owns table Z"), prefer memory_write, which does real
    conflict resolution.
    """
    ds = _dataset(dataset, project_dir)
    async with _pipeline_lock, pipeline_lock():
        result = notes.add_note(content, kind=kind, dataset=ds, key=key)
    return {
        "id": result.note.id,
        "kind": result.note.kind,
        "dataset": ds,
        "changed": result.changed,
        "superseded_note_id": result.superseded_id,
        "embedded": result.embedded,  # False = indexed as pending (endpoint down)
    }


@mcp.tool()
async def memory_write(
    content: str, metadata: dict | None = None, triple: dict | None = None
) -> dict:
    """Store a discrete FACT with conflict resolution — use when a statement
    has a clear subject and could later be contradicted or updated (owners,
    locations, versions, configurations). The fact is versioned
    bi-temporally and indexed incrementally; a superseded fact's text
    immediately stops being searchable. For free-form context (decisions,
    progress, gotchas), use memory_note instead.

    ALWAYS pass `triple`: `{"subject": ..., "relation": ..., "object": ...}`
    — extract it yourself from `content` (you are better at this than any
    local model, and it makes the write sub-second with no LLM call).
    Conflict detection keys on (subject, relation), so keep them short,
    stable, and reusable: "auth-service" / "owned_by" / "platform team",
    not full sentences. Without `triple`, a local-LLM extraction fallback
    is attempted and errors clearly if that stack isn't installed.

    `metadata` may carry:
      - `project_dir` (str): project this fact belongs to (preferred).
      - `dataset` (str): explicit dataset override.
      - `event_time` (str, ISO-8601): when the fact became TRUE (not when
        you learned it) — feeds the bi-temporal model.
      - `importance` (float): accepted but currently ignored (not yet
        threaded through the write path; stated honestly rather than
        silently dropped).

    Returns the stored triple plus what the resolver did (superseded /
    revived / judged-distinct).
    """
    metadata = metadata or {}
    ds = _dataset(metadata.get("dataset"), metadata.get("project_dir"))
    event_time = metadata.get("event_time")
    if event_time is not None:
        # Validate up front: a malformed timestamp would corrupt bi-temporal
        # version ordering silently. Fail loud and cheap.
        try:
            datetime.fromisoformat(str(event_time))
        except ValueError as exc:
            raise ValueError(
                f"metadata.event_time must be an ISO-8601 timestamp, got {event_time!r}"
            ) from exc

    if triple is not None:
        parts = {}
        for field in ("subject", "relation", "object"):
            value = triple.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"triple.{field} must be a non-empty string, got {value!r}"
                )
            parts[field] = value.strip()
        extraction = "agent"
    else:
        # Optional fallback: the local extraction cascade. Lazy import — the
        # LLM stack (instructor/openai/Ollama) may simply not be installed,
        # and must not be required at server startup.
        try:
            from core import router

            extracted = await router.route_extract_triple(content)
        except ImportError as exc:
            raise ValueError(
                "No `triple` given and the local extraction stack is not "
                "installed. Pass triple={subject, relation, object}."
            ) from exc
        parts = {
            "subject": extracted.subject,
            "relation": extracted.relation,
            "object": extracted.object,
        }
        extraction = "local_llm"

    async with _pipeline_lock, pipeline_lock():
        write = await resolver.write(
            parts["subject"],
            parts["relation"],
            parts["object"],
            content,
            dataset=ds,
            event_time=event_time,
        )
        indexed = True
        if write.changed:
            if write.superseded is not None:
                index.remove_item(ds, "fact", write.superseded.id)
            indexed = index.index_item(ds, "fact", write.fact.id, write.fact.text)

    return {
        "subject": parts["subject"],
        "relation": parts["relation"],
        "object": parts["object"],
        "dataset": ds,
        "extraction": extraction,  # 'agent' (no LLM call) | 'local_llm'
        "changed": write.changed,
        "superseded": _fact_summary(write.superseded),
        "revived": write.revived_from_eviction,
        "distinct": write.distinct,
        "indexed": indexed,  # False = stored, embedding pending (endpoint down)
        "cognify_attempts": 0,  # legacy field: Cognee is no longer on this path
    }


@mcp.tool()
async def memory_context(
    query: str | None = None,
    token_budget: int = 2000,
    project_dir: str | None = None,
    dataset: str | None = None,
) -> dict:
    """Get a compact digest of everything this project's memory currently
    holds — CALL THIS AT THE START OF A TASK, before exploring or asking the
    user, to recover decisions, gotchas, and progress from earlier sessions
    (including anything saved before a context-window compaction).

    The digest is current-facts-only (superseded/evicted content can never
    appear), deduplicated, and fits inside `token_budget` (newest and
    most-accessed first; pass `query` to rank by relevance to what you're
    about to do). The response reports its own size vs the raw history it
    replaced — token figures are ~4 chars/token estimates.

    Cheap and read-only: no LLM, safe to call often.
    """
    ds = _dataset(dataset, project_dir)
    index.backfill_facts(ds)  # migrate pre-pivot facts into the index, idempotent
    return context.build_context(query=query, token_budget=token_budget, dataset=ds)


@mcp.tool()
async def memory_search(
    query: str,
    k: int = 5,
    dataset: str | None = None,
    project_dir: str | None = None,
) -> dict:
    """Search this project's memory (facts AND notes) semantically — use it
    whenever the user references past work ("like we did before", "the bug
    from last week") or you need a specific remembered detail; prefer
    memory_context for broad session-start recall.

    Retrieval records access, which keeps a memory warm and resistant to
    forgetting. Returns `hits` (ranked texts) and index-aligned `hit_facts`
    (`{id, source, subject, relation, object, text}`) — `id` + `source` are
    what memory_forget needs. `ranking` reports how results were ranked:
    `hybrid` (keyword + semantic) or `keyword` (FTS5 only, e.g. when the
    embedding endpoint is down) — search always works.
    """
    ds = _dataset(dataset, project_dir)
    k = max(1, min(int(k), MAX_SEARCH_K))
    index.backfill_facts(ds)

    result = index.search(ds, query, k=k)
    ranking = result.mode
    ranked: list[tuple[str, int, str]] = [(h.kind, h.ref_id, h.text) for h in result.hits]

    facts_by_id = {f.id: f for f in resolver.current_facts(ds)}
    hit_facts = []
    for kind, rid, text in ranked:
        if kind == "fact":
            summary = _fact_summary(facts_by_id.get(rid)) or {
                "id": rid, "subject": None, "relation": None, "object": None
            }
            resolver.record_access(rid, dataset=ds)
        else:
            summary = {"id": rid, "subject": None, "relation": None, "object": None}
            notes.record_access(rid, dataset=ds)
        hit_facts.append({**summary, "source": kind, "text": text})

    return {
        "query": query,
        "dataset": ds,
        "ranking": ranking,
        "hits": [text for _, _, text in ranked],
        "hit_facts": hit_facts,
    }


@mcp.tool()
async def memory_stats(
    dataset: str | None = None, project_dir: str | None = None
) -> dict:
    """Live memory-layer numbers for one project dataset: current fact count
    by tier, note counts by kind, index health (indexed vs pending
    embeddings), the cost ledger (reference pricing, not real spend — see
    `pricing_note`), and saved benchmark results. Read-only and cheap.
    """
    ds = _dataset(dataset, project_dir)
    facts = resolver.current_facts(ds)
    by_tier = {"hot": 0, "warm": 0, "cold": 0}
    for f in facts:
        by_tier[f.tier] = by_tier.get(f.tier, 0) + 1

    current_notes = notes.current_notes(ds)
    by_kind = {k: 0 for k in notes.KINDS}
    for n in current_notes:
        by_kind[n.kind] = by_kind.get(n.kind, 0) + 1

    benchmarks: dict[str, dict] = {}
    if RESULTS_DIR.exists():
        for path in sorted(RESULTS_DIR.glob("*.json")):
            try:
                benchmarks[path.stem] = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue  # a half-written result file shouldn't sink stats

    # The cost ledger lives in the optional local-LLM stack; lazy import so
    # a no-LLM install still gets full stats (with an honest placeholder).
    try:
        from core import router

        cost = router.cost_report()
    except Exception:  # noqa: BLE001 — LLM stack not installed
        cost = {
            "total_calls": 0,
            "pricing_note": (
                "Local LLM stack not installed — no extraction/judge/summary "
                "calls are possible, so the ledger is empty by construction."
            ),
        }

    return {
        "dataset": ds,
        "facts": {"total_current": len(facts), "by_tier": by_tier},
        "notes": {"total_current": len(current_notes), "by_kind": by_kind},
        "index": {
            "indexed": index.indexed_count(ds),
            "pending_embeddings": index.pending_count(ds),
        },
        "cost": cost,
        "benchmarks": benchmarks,
    }


@mcp.tool()
async def memory_forget(
    id: int,
    dataset: str | None = None,
    kind: str = "fact",
    project_dir: str | None = None,
) -> dict:
    """Evict one memory by id — use when the user says something is wrong,
    obsolete, or should not be remembered. `id` and `kind` come from
    memory_search's `hit_facts[*].id` / `.source` (kind: "fact" or "note").

    The item is removed from the search index immediately (a row delete —
    no index rebuild) and kept in SQLite stamped `evicted_at`, so a later
    write of the same thing is detected as a revival (regret signal). A
    miss returns `{found: false}` and touches nothing.
    """
    ds = _dataset(dataset, project_dir)
    async with _pipeline_lock, pipeline_lock():
        if kind == "note":
            found = notes.evict_note(id, dataset=ds)
            remaining = len(notes.current_notes(ds))
        else:
            target = next((f for f in resolver.current_facts(ds) if f.id == id), None)
            found = target is not None
            if found:
                resolver.evict(id, dataset=ds)
                index.remove_item(ds, "fact", id)
            remaining = len(resolver.current_facts(ds))

    return {
        "found": found,
        "evicted_id": id if found else None,
        "kind": kind,
        "remaining_current": remaining,
    }


def main() -> None:
    """Entry point: run the server on stdio (blocks until the client detaches).

    MCP's stdio transport carries the JSON-RPC wire on **stdout**, so anything
    else that writes to stdout corrupts it. Hand the MCP writer a *private*
    duplicate of the real stdout, then point the process's own stdout at
    stderr, both at the fd level (`dup2(2, 1)` catches raw/handler writes to
    fd 1 regardless of any cached stream reference) and the Python level
    (`sys.stdout = sys.stderr` catches `print`). Only the protocol writer
    keeps a path to the real stdout.
    """
    import os
    import sys

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
