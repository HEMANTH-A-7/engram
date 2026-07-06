# engram — context for Claude

Extended memory layer over Cognee: bi-temporal conflict resolution, tiered
consolidation/forgetting, cost-aware LLM routing, MCP server, live dashboard.
**100% local/offline** (Ollama + SQLite + LanceDB). Never add cloud calls or
new dependencies without asking.

## Read first
- `PROGRESS.md` — session log + hard-won Cognee/Ollama facts (§ "Hard-won
  config facts" saves hours; don't relearn those).
- `SECURITY.md` — trust model (local, no auth, by design).

## Map
| Path | Role |
|---|---|
| `core/resolver.py` | Bi-temporal fact versioning (SQLite, supersede-not-overwrite) |
| `core/consolidation.py` | Hot/warm/cold tiering, decay eviction, regret signal |
| `core/router.py` | Small→large extraction cascade; conflict judge; cost ledger (in-memory, per-process) |
| `core/ingest.py` | `remember()` = extract → resolver write → full index resync |
| `core/store.py` | Cognee wrapper; `reset_and_load(graph=False)` = fast embed-only path |
| `core/locks.py` | Cross-process flock for the destructive reset+cognify pipeline |
| `mcp_server/server.py` | FastMCP stdio: `memory_write/search/stats/forget` |
| `dashboard/` | FastAPI + vendored Chart.js; charts are LIVE from `main_dataset` |
| `benchmark/` | Fixed-corpus harness (regenerate: `uv run python -m benchmark.harness`) |

## Commands
```bash
uv run pytest -q -m "not integration"     # fast suite, LLM-free
uv run pytest -m integration -q            # real Ollama, slow
uv run uvicorn dashboard.app:app --host 127.0.0.1 --port 8010
uv run python scripts/try_memory.py remember "..."   # hand-test the pipeline
```

## Invariants (don't break)
- `store.reset_and_load` wipes the **entire** Cognee data root — one active
  dataset at a time; destructive paths must hold both pipeline locks.
- MCP stdio: **nothing may print to real stdout** in the server process
  (`main()` redirects fd 1 → stderr; keep it that way).
- All retrieval is `SearchType.CHUNKS`; the knowledge graph/summaries are
  never queried (that's why embed-only cognify is safe).
- Token counts are ~4 chars/token *estimates*; `$` costs are reference cloud
  pricing, real spend is $0 — always label them as such (honesty pattern).
- Never commit/push without an explicit ask; one logical change per commit.
- `main_dataset` in `.resolver_data/facts.db` is real user data — never reset.
