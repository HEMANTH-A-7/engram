# Seamless dev-context memory: hooking engram into a Claude Code project

MCP tools are *passive* — Claude decides when to call them. Hooks fire
*deterministically*, which is what makes memory automatic: context is
captured and recalled with no "remember this" prompts, and survives
context-window compaction.

| Hook | Fires | What engram does |
|---|---|---|
| `SessionStart` | new session, resume, **post-compaction continuation** | Injects `memory_context`'s digest for this project as additional context |
| `PreCompact` | just before the window is compacted | Saves a deterministic session extract to memory — nothing important is lost to the squeeze |
| `Stop` | Claude finishes a turn | Upserts ONE rolling "session state" note (keyed per session, supersedes itself) |

All three scripts are fail-open (any error exits 0 — memory can never block
a session), import-light (no Cognee import; a hook run costs ~0.1–0.4 s),
and derive the project from the hook's own `cwd` payload, so each project
gets its own isolated dataset.

## 1. Install the hooks in a project

Add to that project's `.claude/settings.json` (create the file if needed),
replacing `/path/to/engram` with this repo's absolute path:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/engram/.venv/bin/python /path/to/engram/scripts/hooks/session_start.py"
          }
        ]
      }
    ],
    "PreCompact": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/engram/.venv/bin/python /path/to/engram/scripts/hooks/pre_compact.py"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/engram/.venv/bin/python /path/to/engram/scripts/hooks/stop.py"
          }
        ]
      }
    ]
  }
}
```

Use the venv python directly (as above) rather than `uv run` — it skips
uv's resolution step, keeping hook latency minimal. `uv run --directory
/path/to/engram python scripts/hooks/<name>.py` also works; the scripts
never trust their own cwd (they read the project dir from the hook's stdin
JSON), so `--directory`'s cwd rewrite is harmless.

Optional: `MEMORY_CONTEXT_BUDGET` env (tokens, ~4 chars/token estimate)
caps the SessionStart digest; default 2000.

## 2. Register the MCP server in the same project

So Claude can also *actively* write decisions/gotchas and search memory,
add to that project's `.mcp.json`:

```json
{
  "mcpServers": {
    "memory-layer": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/engram", "python", "-m", "mcp_server.server"],
      "env": { "MEMORY_PROJECT_DIR": "/path/to/this/project" }
    }
  }
}
```

`MEMORY_PROJECT_DIR` keys the tools to this project's dataset when Claude
doesn't pass `project_dir` explicitly (the server can't trust its own cwd —
`uv run --directory` rewrites it to the engram repo). Hardcode the project's
absolute path; `.mcp.json` is per-project anyway. Without it, tools fall
back to the shared `main_dataset`.

## 3. What a session then looks like

1. **Session starts** → digest of this project's decisions / gotchas /
   progress / facts is already in context.
2. Claude works; the tool descriptions prompt it to record decisions and
   gotchas via `memory_note` (sub-second) as they happen; each turn end
   updates the rolling session-state note.
3. **Window fills up** → `PreCompact` saves the session extract →
   compaction runs → the continuation's `SessionStart` re-injects the
   digest, which now includes that extract. Nothing important is lost.
4. Next session (tomorrow, next week) starts at step 1 with everything
   still there.

## Design notes (honest constraints)

- The PreCompact/Stop payloads are deterministic *extracts* (recent user
  requests + last assistant state), not LLM summaries — chosen so hooks are
  fast and can never flake; they are labeled as extracts in the note text.
- The Stop hook uses `memory_note`'s keyed upsert, so a 50-turn session
  produces one rolling state note, not 50 notes. Superseded revisions stay
  in SQLite (supersede-not-overwrite), out of the index and the digest.
- **Ollama is optional** (engram-lite). The search floor is SQLite FTS5
  keyword ranking — always on, no extra process. With Ollama running,
  results upgrade to hybrid keyword+semantic; without it, notes still land
  instantly (embeddings marked pending, filled in whenever it's next up)
  and a failure-cooldown breaker keeps the no-Ollama path overhead-free.
  Nothing blocks, nothing is lost.
