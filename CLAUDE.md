# engram — context for Claude

Seamless dev-context memory layer for coding agents (pivoted Session 9 from
a manually-invoked fact store; "engram-lite" Session 10): context is
captured and recalled automatically via Claude Code hooks + MCP tools,
survives context-window compaction, and is keyed per-project. Bi-temporal
conflict resolution, tiered consolidation/forgetting, live dashboard.
**100% local/offline, SQLite only** — NO local LLM required: the calling
agent supplies extracted triples; search floor is FTS5 keyword (BM25), and
Ollama embeddings are an *optional* upgrade to hybrid ranking. Never add
cloud calls or new dependencies without asking.

## Read first
- `PROGRESS.md` — session log + hard-won Cognee/Ollama facts (§ "Hard-won
  config facts" saves hours; don't relearn those).
- `docs/HOOKS.md` — how the seamless layer wires into a Claude Code project.
- `SECURITY.md` — trust model (local, no auth, by design).

## Map
| Path | Role |
|---|---|
| `core/index.py` | **Serving path**: owned incremental index — FTS5 keyword (always) + vectors (optional), RRF hybrid; `search()` never raises |
| `core/notes.py` | Fast raw-text notes (decision/progress/gotcha/reference), keyed upsert, ~ms writes |
| `core/context.py` | `build_context(query, token_budget)` — budgeted current-only digest + self-measurement |
| `core/projects.py` | Per-project dataset keying (`proj_<slug>_<hash>`; fallback `main_dataset`) |
| `core/embeddings.py` | Optional Ollama embed client (stdlib, import-light), failure-cooldown breaker, test seam `set_embed_fn` |
| `core/resolver.py` | Bi-temporal fact versioning (SQLite, supersede-not-overwrite) |
| `core/consolidation.py` | Hot/warm/cold tiering, decay eviction, regret signal |
| `core/router.py` | Small→large extraction cascade; conflict judge; cost ledger (in-memory, per-process) |
| `core/ingest.py` | `remember()` = extract → resolver write (serving path uses `sync=False` + incremental index) |
| `core/store.py` | Cognee wrapper — benchmarks/graph demo ONLY, no longer on the serving path |
| `core/locks.py` | Cross-process flock serializing memory writes |
| `mcp_server/server.py` | FastMCP stdio: `memory_note/write/context/search/stats/forget` |
| `scripts/hooks/` | Claude Code hooks: SessionStart inject / PreCompact save / Stop rolling state |
| `dashboard/` | FastAPI + vendored Chart.js; charts are LIVE from `main_dataset` |
| `benchmark/` | Fixed-corpus harness (regenerate: `uv run python -m benchmark.harness`) |

## Commands
```bash
uv run pytest -q -m "not integration"     # fast suite, LLM-free
uv run pytest -m integration -q            # real Ollama, slow
uv run python scripts/measure_pivot.py     # write latency + digest-vs-raw metrics
uv run uvicorn dashboard.app:app --host 127.0.0.1 --port 8010
uv run python scripts/try_memory.py remember "..."   # hand-test the pipeline
```

## Invariants (don't break)
- **Serving path is incremental**: `memory_*` tools must never trigger
  `store.reset_and_load` (it wipes the whole Cognee data root; it survives
  for benchmark/demo paths only). Superseded/evicted content is removed from
  `core/index.py` at write time — that's what keeps it unsearchable.
- **Serving path imports neither Cognee nor the LLM stack at module load**
  (`mcp_server/server.py` starts in ~150ms; extraction/judge are lazy,
  optional fallbacks). `index.search()` never raises for infrastructure
  reasons — embed down → keyword mode; DB unopenable → empty
  `mode="unavailable"`; the mode used is always reported.
- Per-project isolation: nothing on the serving path may touch another
  dataset's rows; destructive helpers must be dataset-scoped.
- Hook scripts (`scripts/hooks/`) stay import-light (never import Cognee)
  and fail-open (exit 0 on any error) — memory must never block a session.
- MCP stdio: **nothing may print to real stdout** in the server process
  (`main()` redirects fd 1 → stderr; keep it that way).
- Token counts are ~4 chars/token *estimates*; `$` costs are reference cloud
  pricing, real spend is $0 — always label them as such (honesty pattern).
- Never commit/push without an explicit ask; one logical change per commit.
- `main_dataset` in `.resolver_data/facts.db` is real user data — never reset.
