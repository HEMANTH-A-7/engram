# engram

An [engram](https://en.wikipedia.org/wiki/Engram_(neuropsychology)) is the physical
trace a memory leaves in the brain. This one is an extended memory layer for AI
agents, built on [Cognee](https://github.com/topoteretes/cognee).
Adds bi-temporal conflict resolution, tiered consolidation/forgetting, a cost-aware
extraction router, an MCP server, and a benchmark-driven local dashboard.

**Runs fully offline** (Ollama + SQLite + LanceDB) with zero cloud credentials and no
runtime network calls.

## Why

Vanilla RAG memory appends everything and never forgets, so contradictions
accumulate ("Alice lives in Boston" *and* "Alice lives in Seattle" both stay
retrievable) and cost grows without bound. This layer sits in front of Cognee and
adds the parts an agent actually needs: it *supersedes* stale facts instead of
piling them up, *forgets* cold ones on a decay schedule, and routes extraction to
the cheapest model that can do the job — every decision measured before/after by a
benchmark harness.

## Architecture

Eight components, built bucket-by-bucket (see [PROGRESS.md](PROGRESS.md)):

| Layer | File | Role |
|-------|------|------|
| Baseline | `core/store.py` | Thin async Cognee wrapper (add → cognify → search). |
| Conflict resolution | `core/resolver.py` | Bi-temporal fact versioning in SQLite; supersede-not-overwrite so stale text never reaches the index. |
| Consolidation | `core/consolidation.py` | Hot/warm/cold tiering, decay-based eviction, deterministic regret signal. |
| Cost router | `core/router.py` | Cascade extraction (small → large on failure); conflict-judge + summarize; reference-priced cost report. |
| Ingest | `core/ingest.py` | `remember()` = extract (router) → resolver write → resync Cognee index. |
| MCP server | `mcp_server/server.py` | FastMCP stdio server: `memory_write` / `memory_search` / `memory_stats` / `memory_forget`. |
| Dashboard | `dashboard/` | FastAPI + vendored Chart.js, 5 metric cards, 100% offline. |
| Benchmark | `benchmark/` | Recall@k, conflict accuracy, regret rate, cost/1k, tokens saved, storage growth. |

## Quickstart
```bash
uv sync                          # create venv (Python 3.12) + install deps
cp .env.example .env             # offline Ollama defaults
uv run python scripts/smoke.py   # verify the foundation runs
```

Requires a local [Ollama](https://ollama.com) with `gemma4:latest` (extraction/graph),
`llama3.2:3b` (small-tier cascade), and `nomic-embed-text` (embeddings) pulled.

## Run it

```bash
# Dashboard (use 8010 — stale procs squat on 8000)
uv run uvicorn dashboard.app:app --host 127.0.0.1 --port 8010
#   → http://127.0.0.1:8010

# MCP server (stdio) — register with Claude Code / Codex / Antigravity
uv run --directory . python -m mcp_server.server
#   see mcp_server/REGISTRATION.md for client config

# Tests
uv run pytest -m "not integration" -q   # fast, LLM-free (45 tests)
uv run pytest -m integration -q          # slow, real Ollama
```

## Design notes / honesty

This is a portfolio build; the metrics are deliberately labeled for what they are:

- **Cost is reference cloud pricing, not real spend.** Everything runs locally against
  Ollama — actual spend is $0. `router.cost_report()` prices real token counts against
  published 2026 hosted rates and carries a `pricing_note` saying so.
- **Token counts are a documented `chars/4` estimate** (`tiktoken`'s BPE vocab
  downloads on first use, which would break the offline guarantee).
- **Local structured-output extraction is genuinely flaky.** gemma4 intermittently
  emits null/mistyped fields that fail Pydantic validation inside Cognee's `cognify()`;
  the harness retries with bounds and logs attempts rather than hiding them.
- **Conflict-accuracy card may show a "metric pending" placeholder.** Its backing
  `benchmark/results/conflict.json` is regenerated separately (it goes through the flaky
  `cognify()` path); the dashboard degrades gracefully and the other four cards render
  real data.

See [PROGRESS.md](PROGRESS.md) for status, locked decisions, and hard-won facts.
```
