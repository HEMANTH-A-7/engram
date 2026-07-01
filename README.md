# memory-layer

Extended memory layer for AI agents, built on [Cognee](https://github.com/topoteretes/cognee).
Adds bi-temporal conflict resolution, tiered consolidation/forgetting, a cost-aware
extraction router, an MCP server, and a benchmark-driven local dashboard.

**Runs fully offline** (Ollama + SQLite + LanceDB) with zero cloud credentials.

## Quickstart
```bash
uv sync                          # create venv (Python 3.12) + install deps
cp .env.example .env             # offline Ollama defaults
uv run python scripts/smoke.py   # verify the foundation runs
```

See [PROGRESS.md](PROGRESS.md) for status, decisions, and the build plan.
