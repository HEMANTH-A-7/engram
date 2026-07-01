# PROGRESS — Extended Memory Layer for AI Agents

> Session tracker so any new session gets full context without replaying history.
> Read this first. Update it at every bucket checkpoint.

## What this is
Portfolio project (FAANG SDE1): an extended memory layer over **Cognee** adding
bi-temporal conflict resolution, tiered consolidation/forgetting, a cost-aware
extraction router, an MCP server, and a benchmark-driven local dashboard.
Spec: `/Users/hemanth/Downloads/memory-layer-project-spec.md`.

## Locked decisions
| # | Decision | Choice | Date |
|---|----------|--------|------|
| D1 | LLM backend | Local-first via **Ollama**, behind a pluggable provider seam (`core/config.py`) so the same binary can borrow a host model (Claude Code / Codex / Antigravity) later. Costs computed via a price table, never a live bill. | 2026-07-01 |
| D2 | Env manager | **uv**, Python pinned to **3.12** (Cognee supports 3.10–3.12; system Python is 3.14). | 2026-07-01 |
| D3 | Build order | Minimal **benchmark harness right after baseline**, before the conflict resolver, so every component has clean before/after numbers. | 2026-07-01 |
| D4 | Cost model | Dollar-equivalent cost = tokens × published price table, so metrics stay meaningful while offline. | 2026-07-01 |

## Environment facts
- Ollama models present: `gemma4:latest`, `gemma4-32k` (strong), `llama3.2:3b` (small), `nomic-embed-text` (embeddings, 768 dims).
- Ollama base endpoint `http://localhost:11434`; OpenAI-compat at `/v1`.
- `uv` at `~/.local/bin/uv`. Python pinned **3.12.13**. **Cognee 1.2.2** installed (spec assumed 0.1.x — 1.x adds remember/recall/forget but V1 add/cognify/search still work).
- **Config is unified on Cognee's canonical env names** (`LLM_*`, `EMBEDDING_*`, litellm-style `ollama/<model>`); `core/config.py` reads the same `.env`. Router extras: `MEMORY_MODEL_SMALL/LARGE`.
- Cognee gotcha: for `ollama`, it requires the full trio `LLM_MODEL`+`LLM_ENDPOINT`+`LLM_API_KEY` (and `EMBEDDING_PROVIDER`+`EMBEDDING_MODEL`+`EMBEDDING_DIMENSIONS`) or import fails. `LLM_API_KEY=ollama` is a dummy.
- Verify: `uv run python scripts/smoke.py` → all green offline.

## Bucket status
- [x] **Bucket 0 — Scaffold & env** ✅ (2026-07-01): repo skeleton, uv/py3.12, deps, unified Ollama config, offline smoke test all-green.
- [ ] Bucket 1 — Baseline Cognee pipeline (add → cognify → search)
- [ ] Bucket 2 — Minimal benchmark harness (recall@k, latency) on baseline
- [ ] Bucket 3 — Conflict resolver (bi-temporal fact versioning) + extend harness
- [ ] Bucket 4 — Consolidation / forgetting policy (tiering, regret rate)
- [ ] Bucket 5 — Cost-aware extraction router
- [ ] Bucket 6 — MCP server (stdio) + register with 3 clients
- [ ] Bucket 7 — Local dashboard (FastAPI + Chart.js via Claude design)

## UI / Claude design
Only the dashboard (Bucket 7) needs it. When we reach it, hand off: the JSON data
contract from the FastAPI backend + a layout brief. Nothing needed before then.

## Session log
### Session 1 — 2026-07-01
- Reviewed spec, surfaced offline-vs-cost tension, locked D1–D4.
- **Bucket 0 done**: uv + py3.12 env, Cognee 1.2.2, repo skeleton, unified `.env` on Cognee's scheme, `scripts/smoke.py` green offline. Committed.
- Next: Bucket 1 — baseline `add → cognify → search` end to end against Ollama.
