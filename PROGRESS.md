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
- Verify: `uv run python scripts/smoke.py` → all green offline; `uv run pytest -m integration -q` → baseline passes (~4 min).

### Hard-won Cognee+Ollama config facts (don't relearn these)
- **Endpoints differ by role**: LLM `LLM_ENDPOINT=http://localhost:11434/v1` (instructor uses a raw OpenAI client); embeddings `EMBEDDING_ENDPOINT=http://localhost:11434/api/embed` (native path). `core/config._openai_base()` normalizes either back to `<host>/v1` for our own calls.
- **Bare model names** (no `ollama/` prefix): the native adapters pass the name verbatim and Ollama 404s on the prefix. `LLM_MODEL=gemma4:latest`, `EMBEDDING_MODEL=nomic-embed-text`.
- **Extraction model = gemma4**, NOT llama3.2:3b. 3B models echo the JSON *schema* instead of an instance → validation fail. **→ D5 below.**
- **`LLM_TEMPERATURE=0.3`**: at temp 0, instructor retries are identical, so a single null-for-required-string response never recovers. A small temp makes retries vary and succeed. Tradeoff: extraction is nondeterministic (acceptable; eval set is fixed, average over runs).
- Needs `transformers` (BERT tokenizer `HUGGINGFACE_TOKENIZER=bert-base-uncased`, one-time download) and `COGNEE_SKIP_CONNECTION_TEST=true` (30s pre-flight < Ollama cold start).
- Cognee is graph-provider **ladybug** + vector by default; storage pinned to repo `.cognee_data/` + `.cognee_system/`.
- **Ollama structured-output retries are hardcoded to 2** (`cognee/.../llm/ollama/adapter.py`, not env-configurable), and `cognify()` batches every fact into one pipeline run — one flaky extraction (gemma4 emitting a null `description`) aborts the whole batch. Observed 3/3 attempts needed on an 8-fact benchmark run, so this isn't rare. `benchmark/harness.py` retries the whole ingest+cognify batch (bounded at 3, logged via `cognify_attempts`) to compensate; `core/store.py` (the baseline pipeline) is left vanilla.
- **Raw CHUNKS retrieval latency is high locally**: p50 ≈13s / p95 ≈21s per query on the 8-fact benchmark set, on local Ollama CPU (no GPU). Expected given the hardware — real baseline number, not a bug to chase.

### Open decision to confirm with user
- **D5 (Bucket 5 router)**: spec wants a cheap/small model for routine extraction, but 3B can't do reliable structured graph extraction. Router "cheap path" must be a model that CAN (e.g. a 7B). Revisit at Bucket 5.

## Bucket status
- [x] **Bucket 0 — Scaffold & env** ✅ (2026-07-01): repo skeleton, uv/py3.12, deps, unified Ollama config, offline smoke test all-green.
- [x] **Bucket 1 — Baseline Cognee pipeline** ✅ (2026-07-01): `core/store.py` wrapper (add/cognify/search), `scripts/baseline_demo.py`, `tests/test_baseline.py` integration test PASSES (229s). GRAPH_COMPLETION correctly answers offline.
- [x] **Bucket 2 — Minimal benchmark harness (recall@k, latency) on baseline** ✅ (2026-07-01): `benchmark/eval_set.py` (8 stable-fact cases), `benchmark/harness.py` (recall@{1,3,5} + latency p50/p95, writes `benchmark/results/<label>.json`), `tests/test_benchmark.py` integration test PASSES (914s). Baseline results: recall@1/3/5 = **1.0**, latency p50 **≈13.0s** / p95 **≈20.8s** per query. `cognify()` needed 3/3 retry attempts to succeed (see Hard-won facts above).
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
- **Bucket 1 done**: baseline pipeline working offline. Spent most effort reverse-engineering Cognee 1.2's Ollama config (endpoints, bare model names, extraction model, temperature) — all captured above. Integration test green.
- **Bucket 2 done**: benchmark harness v1 (8 synthetic stable-fact eval cases, recall@{1,3,5} + latency p50/p95). First run failed outright — gemma4 emitted a null `description` field and Cognee's hardcoded 2-attempt retry couldn't recover; added a harness-level bounded retry (3 attempts) around the whole ingest+cognify batch, logged via `cognify_attempts` rather than silently swallowed. Second run succeeded on the 3rd attempt: recall@1/3/5 = 1.0, latency p50 ≈13s / p95 ≈21s (local CPU Ollama — high but real, this is the before/after baseline for later buckets). Integration test green (914s).
- Next: Bucket 3 — conflict resolver (bi-temporal fact versioning: `event_time`/`ingestion_time`, supersede-not-overwrite on contradiction), then extend the eval set with contradiction/update cases so recall@k + a new conflict-resolution-accuracy metric can be compared before/after.
