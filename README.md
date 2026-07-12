# engram

An [engram](https://en.wikipedia.org/wiki/Engram_(neuropsychology)) is the physical
trace a memory leaves in the brain. This one is a **seamless dev-context memory
layer for coding agents**: project context is captured and recalled automatically
— no "remember this" prompts — via Claude Code hooks and MCP tools, survives
context-window compaction, and is isolated per project.

**100% local. SQLite only. No LLM required.** The calling agent (Claude) is
already a frontier model, so it supplies the structured extractions; engram's
job is durable, versioned, searchable storage. Search works with nothing but
Python installed (SQLite FTS5 / BM25); a local [Ollama](https://ollama.com)
embedding model is an *optional* upgrade that turns search hybrid
(keyword + semantic, reciprocal-rank fusion) — and the response always reports
which mode actually served it.

## Why

Coding agents forget. A session's hard-won context dies at the context-window
boundary: compaction squeezes it out, new sessions start cold, and "just
re-read the codebase" doesn't recover *decisions* ("we chose flock over a
lockfile because…") or *gotchas* ("the sandbox can't write SQLite over the
mount"). Vanilla RAG memory appends everything and never forgets, so
contradictions accumulate and stale facts stay retrievable forever.

engram fixes both ends:

- **Automatic capture/recall** — hooks fire deterministically: `SessionStart`
  injects a budgeted context digest, `PreCompact` saves a session extract
  before the squeeze, `Stop` keeps one rolling session-state note. MCP tools
  cover the deliberate writes (decisions, gotchas, facts).
- **Versioned, not append-only** — a bi-temporal resolver *supersedes* stale
  facts (old versions stay in history, out of the index), tiered
  consolidation decays and evicts cold ones, and near-duplicates are flagged
  at write time.

## Architecture

The serving path is owned code over one SQLite file — no Cognee, no LLM stack,
imported in ~150 ms:

| Layer | File | Role |
|-------|------|------|
| Notes | `core/notes.py` | Raw-text notes (decision/progress/gotcha/reference), keyed upsert, ~ms writes. |
| Index | `core/index.py` | Incremental FTS5 keyword index (always) + vector index (optional), RRF hybrid; `search()` never raises. |
| Context | `core/context.py` | `build_context(query, token_budget)` — budgeted current-only digest that self-measures vs raw history. |
| Projects | `core/projects.py` | Per-project dataset keying (`proj_<slug>_<hash>`); no serving-path operation is global. |
| Embeddings | `core/embeddings.py` | Optional Ollama embed client (stdlib HTTP), 60 s failure-cooldown breaker. |
| Resolver | `core/resolver.py` | Bi-temporal fact versioning; supersede-not-overwrite; near-duplicate write guard. |
| Consolidation | `core/consolidation.py` | Hot/warm/cold tiering, decay eviction, deterministic regret signal. |
| Dedupe | `core/dedupe.py` | Batch near-duplicate sweep; flag-only without an LLM (never auto-merges). |
| MCP server | `mcp_server/server.py` | FastMCP stdio: `memory_note` / `memory_write` / `memory_context` / `memory_search` / `memory_stats` / `memory_forget`. |
| Hooks | `scripts/hooks/` | SessionStart inject / PreCompact save / Stop rolling state — import-light, fail-open. |
| Dashboard | `dashboard/` | FastAPI + vendored Chart.js; live charts from the real memory data. |

Legacy engine, kept for the graph demo and benchmarks only (not on the serving
path): `core/store.py` (Cognee wrapper), `core/ingest.py` / `core/router.py`
(local-LLM extraction cascade — now the *fallback* when the agent doesn't
supply a triple).

## Measured (real runs, this repo's data)

- Note write **p50 27.5 ms / p95 34.2 ms** with live embeddings — vs 3.2 s for
  the pre-pivot rebuild-everything write path (**~116x**). No-Ollama floor:
  p50 ~5 ms. (`scripts/measure_pivot.py`)
- MCP server import **~156 ms** (was seconds when Cognee loaded at startup).
- **29.6% token savings** for the digest vs replaying raw history, measured on
  real accumulated data. Honest caveat: at tiny scale (≤ ~8 items, no
  superseded revisions) the digest's markdown scaffolding can cost slightly
  *more* than raw — savings come from churn, and the metric reports itself
  either way.

## Quickstart

```bash
uv sync                                   # Python 3.12 venv + deps
uv run pytest -q -m "not integration"     # fast suite — no Ollama, no network
uv run --directory . python -m mcp_server.server   # the MCP server (stdio)
```

No `.env`, no models, no Ollama needed for the above. (One caveat: two fast
tests of the *legacy* Cognee path fetch a small HuggingFace tokenizer on
first-ever run; cached forever after. The serving path itself never touches
the network.) Optional extras:

- **Semantic search**: run Ollama with `nomic-embed-text` pulled; the index
  picks it up automatically (pending vectors heal on the next search).
- **Benchmarks / graph demo / local-LLM fallback**: `cp .env.example .env`
  and pull the models it names; `uv run pytest -m integration -q` exercises
  the full stack.

## Wire it into a project

1. **MCP tools**: register the server in the project's `.mcp.json` with
   `MEMORY_PROJECT_DIR` pointing at that project — see
   [`mcp_server/REGISTRATION.md`](mcp_server/REGISTRATION.md).
2. **Automatic layer**: add the three hooks to the project's
   `.claude/settings.json` — see [`docs/HOOKS.md`](docs/HOOKS.md).

This repo dogfoods both (committed `.mcp.json` + `.claude/settings.json`).

## Run the dashboard

```bash
uv run uvicorn dashboard.app:app --host 127.0.0.1 --port 8010
#   → http://127.0.0.1:8010  (charts are live from the real memory data, 4 s poll)
```

## Design notes / honesty

The metrics are deliberately labeled for what they are:

- **Degradation is always reported, never silent.** Search responses say which
  mode served them (`hybrid` / `keyword` / `unavailable`); the context digest
  reports the ranking it actually used; a write that couldn't embed says the
  vector is pending.
- **Token counts are a documented `chars/4` estimate** (`tiktoken`'s vocab
  downloads on first use, which would break the offline guarantee). The
  headline ratio survives any consistent estimator.
- **"$ cost" anywhere in reports is reference cloud pricing, not real spend** —
  everything runs locally; actual spend is $0, and every report carries a
  `pricing_note` saying so.
- **Hooks fail open by design**: any error exits 0. Memory must never block a
  session, a turn, or a compaction. The trade: a hook failure is visible only
  in its stderr, not as a blocked session.
- **Dedupe never auto-merges without a judge.** The no-LLM floor flags
  suspected duplicates (`memory_stats.possible_duplicates`) and asks; a wrong
  merge hides a fact, a flag just asks a question.
- **Local structured-output extraction (the legacy fallback path) is genuinely
  flaky** — documented in [PROGRESS.md](PROGRESS.md)'s hard-won facts rather
  than papered over. This is *why* agent-supplied triples are the primary path.

See [PROGRESS.md](PROGRESS.md) for the session log, locked decisions, and
hard-won facts; [SECURITY.md](SECURITY.md) for the trust model (local,
single-user, no auth by design).
