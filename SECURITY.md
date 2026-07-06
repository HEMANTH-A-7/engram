# Security

## Trust model

This is a **local-first, single-user** memory layer. There is deliberately no
authentication anywhere:

- The **MCP server** speaks stdio only. There is no network listener; anything
  that can spawn the process already runs as your user. Adding a token layer
  here would be security theater.
- The **dashboard** binds `127.0.0.1` only and is read-only (all endpoints are
  GET). It additionally validates the `Host` header (`TrustedHostMiddleware`)
  so a DNS-rebinding page can't read your memory contents through your own
  browser.
- **No cloud calls.** LLM/embedding traffic goes to local Ollama
  (`localhost:11434`). `LLM_API_KEY=ollama` in `.env` is a dummy value Cognee
  requires to be non-empty; it is not a secret.

## What the audit hardened (2026-07-06)

| Control | Where |
|---|---|
| Cross-process `flock` around the destructive reset+cognify pipeline (two MCP client sessions each spawn their own server process; the in-process asyncio lock alone can't serialize them) | `core/locks.py`, used by `mcp_server/server.py` |
| DNS-rebinding guard (Host allowlist: `localhost`, `127.0.0.1`) | `dashboard/app.py` |
| Owner-only permissions on private data (`.resolver_data/` 700, `facts.db` 600, `.cognee_data/` + `.cognee_system/` 700) | `core/resolver.py`, `core/store.py` |
| MCP input validation: `event_time` must be ISO-8601 (fails before any LLM call), `k` clamped to 1–50 | `mcp_server/server.py` |

All SQL is parameterized (`core/resolver.py`); the frontend writes only via
`textContent` (no XSS sink); no `eval`/`exec`/`pickle`/`shell=True` anywhere in
project code; Chart.js is vendored, not CDN-loaded.

## Known accepted risks

- **`diskcache` 5.6.3 (CVE-2025-69872, transitive via Cognee).** No fixed
  release exists. Exploitation requires an attacker who can already write to
  your local cache directory — i.e. same-user filesystem access, at which
  point they own the machine anyway. Re-check when Cognee bumps it.
- **Prompt injection via stored memories.** The layer stores whatever text a
  client writes; that text is later fed to local models for extraction /
  conflict judgment. Outputs are schema-constrained (`Triple`, a 3-way
  verdict), which bounds the blast radius to *wrong memories*, not code
  execution. Inherent to any memory layer; don't feed it untrusted documents.
- **POSIX-only lock.** `core/locks.py` uses `fcntl`; Windows is out of scope
  for this project.

## Reporting

Personal project — open an issue or contact the author directly.
