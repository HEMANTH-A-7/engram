# Registering the memory-layer MCP server

One server binary, three clients, **zero code changes between them** — that's
the spec's definition of done for Bucket 6. Clients differ only in a few lines
of config, all pointing at the same launch command:

```
uv run --directory /Users/hemanth/Documents/Cognee-project python -m mcp_server.server
```

The server speaks MCP over **stdio** (no network, no auth — local process
only). Each client spawns it as a subprocess and talks to it over stdin/stdout.

> Paths below are absolute for this machine. On another checkout, replace
> `/Users/hemanth/Documents/Cognee-project` with that repo's absolute path.

> **Per-project memory (Session 9 pivot):** when registering this server in
> ANOTHER project's `.mcp.json`, add
> `"env": { "MEMORY_PROJECT_DIR": "/path/to/that/project" }` so its tools key
> to that project's own dataset (the server can't trust its cwd —
> `uv run --directory` rewrites it to this repo). Without it, tools fall back
> to the shared `main_dataset`. For the fully automatic layer (SessionStart
> digest injection, PreCompact save, Stop rolling state), see
> [`docs/HOOKS.md`](../docs/HOOKS.md).

---

## 1. Claude Code — `.mcp.json` (committed to the repo)

Already present at the repo root as [`.mcp.json`](../.mcp.json). Claude Code
auto-discovers a project-scoped `.mcp.json`, so anyone who checks out the repo
and opens Claude Code here gets the `memory-layer` server (it'll prompt once to
trust it).

Equivalent CLI form (registers it in your user/global Claude config instead):

```bash
claude mcp add memory-layer --transport stdio \
  -- uv run --directory /Users/hemanth/Documents/Cognee-project python -m mcp_server.server
```

Verify: `claude mcp list` should show `memory-layer` as connected.

---

## 2. Antigravity — `~/.gemini/config/mcp_config.json` (written)

This file has been populated with the entry below. Antigravity reads
`mcpServers` from it on startup:

```json
{
  "mcpServers": {
    "memory-layer": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/Users/hemanth/Documents/Cognee-project",
        "python",
        "-m",
        "mcp_server.server"
      ]
    }
  }
}
```

If you already had other servers in that file, merge this `memory-layer` key
into the existing `mcpServers` object rather than replacing the whole file.

---

## 3. Codex CLI — `~/.codex/config.toml` (paste-ready, not yet installed)

Codex isn't installed on this machine yet, so nothing was written. When you
install it, add this block to `~/.codex/config.toml` — the same binary, no code
changes:

```toml
[mcp_servers.memory_layer]
command = "uv"
args = [
  "run",
  "--directory",
  "/Users/hemanth/Documents/Cognee-project",
  "python",
  "-m",
  "mcp_server.server",
]
```

(TOML table keys can't contain `-`, so the Codex server name is `memory_layer`
with an underscore; the exposed tool names are identical across all clients.)

---

## Tools exposed

| Tool | Signature | Purpose |
|------|-----------|---------|
| `memory_write` | `(content: str, metadata?: dict)` | Extract a triple, version it bi-temporally, reindex. `metadata` may carry `dataset`, `event_time`. |
| `memory_search` | `(query: str, k=5, dataset="main_dataset")` | Ranked retrieval; records access (feeds tiering). Returns `hits` (texts) plus index-aligned `hit_facts` (`{id, subject, relation, object, text}`) — the `id` is what `memory_forget` needs. |
| `memory_stats` | `(dataset="main_dataset")` | Live fact counts by tier + cost report + saved benchmark results. |
| `memory_forget` | `(id: int, dataset="main_dataset")` | Evict a fact by id and drop it from the index. Get the `id` from `memory_search`'s `hit_facts`. |

## Smoke test

```bash
# Starts the server on stdio; it blocks waiting for a client. Ctrl-C to stop.
uv run python -m mcp_server.server
```

A clean start with no traceback means the binary is good; real verification is
`tests/test_mcp_server_integration.py` (starts the server as a subprocess and
lists its tools over a real stdio session).
