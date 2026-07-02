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
- **gemma4 echoes the JSON-schema wrapper, not randomly but systematically**: instructor's `json_mode` embeds the target schema as text in the system prompt (it literally contains a `"properties"` key). For our narrow `Triple(subject, relation, object)` schema, gemma4 kept nesting the actual values under a `properties` key (`{"properties": {"subject": "Alice", ...}}`) instead of returning them top-level — and did this on the *first* generation of every one of 3 outer retry attempts (6/6 LLM calls in one observed run), so blind retrying alone never recovered it. Fix: a Pydantic `model_validator(mode="before")` on `Triple` (`core/ingest.py`) that unwraps this exact shape when the nested values are plain strings (the "self-correction" response instructor triggers on its own internal retry usually has real values, just misplaced) — repairs the response instead of burning a retry on a fixable shape. Still correctly rejects the case where `properties.*` are themselves schema-shaped dicts (no real value present).
- **Conflict-case eval sentences must not restate the stale value in the current fact's own text.** Original `CONFLICT_CASES`/`test_ingest.py` used transition sentences like *"Alice moved from Boston to Seattle, so she now lives in Seattle."* as the *update* — but that sentence's raw text (which is legitimately the only thing indexed for the current fact) contains "Boston" as a substring, so any stale-value leak check on retrieved text false-positives even under perfect resolver behavior. Reworded all 3 `CONFLICT_CASES` + the integration test's update sentence to be self-contained (e.g. "Alice now lives in Seattle.") — this is a broader property to keep in mind when adding future eval cases: an update fact's own current-state description must never happen to contain the value it's replacing.

### Open decision to confirm with user
- **D5 (Bucket 5 router)**: spec wants a cheap/small model for routine extraction, but 3B can't do reliable structured graph extraction. Router "cheap path" must be a model that CAN (e.g. a 7B). Revisit at Bucket 5.

## Bucket status
- [x] **Bucket 0 — Scaffold & env** ✅ (2026-07-01): repo skeleton, uv/py3.12, deps, unified Ollama config, offline smoke test all-green.
- [x] **Bucket 1 — Baseline Cognee pipeline** ✅ (2026-07-01): `core/store.py` wrapper (add/cognify/search), `scripts/baseline_demo.py`, `tests/test_baseline.py` integration test PASSES (229s). GRAPH_COMPLETION correctly answers offline.
- [x] **Bucket 2 — Minimal benchmark harness (recall@k, latency) on baseline** ✅ (2026-07-01): `benchmark/eval_set.py` (8 stable-fact cases), `benchmark/harness.py` (recall@{1,3,5} + latency p50/p95, writes `benchmark/results/<label>.json`), `tests/test_benchmark.py` integration test PASSES (914s). Baseline results: recall@1/3/5 = **1.0**, latency p50 **≈13.0s** / p95 **≈20.8s** per query. `cognify()` needed 3/3 retry attempts to succeed (see Hard-won facts above).
- [x] **Bucket 3 — Conflict resolver (bi-temporal fact versioning)** ✅ (2026-07-02): `core/resolver.py` — an owned bi-temporal fact store (SQLite, zero LLM dependency): `event_time`/`ingestion_time`, supersede-not-overwrite via canonical `(subject, relation)` matching, never deletes old rows. `core/ingest.py` — orchestration layer: extracts a single `(subject, relation, object)` triple per fact (narrower schema than Cognee's own multi-node `KnowledgeGraph` extraction), writes it through the resolver, then rebuilds Cognee's index from `resolver.current_facts()` so a superseded fact's text is never searchable (not just outranked). `benchmark/eval_set.py` extended with `CONFLICT_CASES` (3 contradiction scenarios); `benchmark/harness.py` extended with `run_conflict_eval()` comparing resolver-mode vs. naive-cumulative-overwrite on accuracy + stale-leak-rate. 9 fast unit tests (`tests/test_resolver.py`, 0.03s) + 1 integration test (`tests/test_ingest.py`, ~215–250s) PASS. See Hard-won facts below for two real bugs found and fixed during verification.
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

### Session 2 — 2026-07-02
- **Bucket 3 done**: built the resolver as an owned layer in front of Cognee, not a thin wrapper — `core/resolver.py` (bi-temporal SQLite fact store, canonical `(subject, relation)` matching, supersede-not-overwrite, zero LLM dependency, 9/9 unit tests in 0.03s) + `core/ingest.py` (single-triple extraction via instructor+gemma4, writes through the resolver, rebuilds Cognee's index from only the current facts). Extended `benchmark/eval_set.py` (`CONFLICT_CASES`) and `benchmark/harness.py` (`run_conflict_eval()`, resolver-mode vs. naive-overwrite accuracy + stale-leak-rate).
- Debugged the integration test through 5 real-Ollama runs: found gemma4 systematically (not randomly) echoes instructor's injected JSON-schema wrapper for this schema — fixed with a `model_validator` on `Triple` that repairs the recoverable shape instead of just retrying blind (see Hard-won facts above). Then found a second, unrelated bug: the conflict eval sentences themselves leaked the stale value into the *current* fact's own text, causing a false "stale leak" assertion even with a correctly-working resolver — reworded all update sentences to be self-contained.
- `tests/test_ingest.py` integration test now passes (~215–250s); full fast suite (9 tests, no LLM) passes in 1.93s.
- Next: Bucket 4 — consolidation / forgetting policy (tiering, regret rate).
