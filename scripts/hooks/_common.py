"""Shared plumbing for the Claude Code hook scripts.

Hooks are the deterministic half of "seamless": MCP tools are passive
(Claude chooses to call them), hooks FIRE on every session start / compaction
/ turn end. They must therefore be fast and unbreakable:

- Import-light: only the pivot's serving modules (resolver/notes/index/
  context — SQLite + one optional embed HTTP call). Importing Cognee here
  would cost seconds per hook invocation.
- Fail-open: any exception exits 0 with no output. A memory hiccup must
  never block a session, a compaction, or a turn.
- The project directory comes from the hook's stdin JSON `cwd` (fallback
  $CLAUDE_PROJECT_DIR) — NEVER os.getcwd(), which is wrong whenever the
  hook command is launched via `uv run --directory <engram>`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make the engram repo importable no matter where the hook process starts.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def read_hook_input() -> dict:
    try:
        return json.load(sys.stdin) or {}
    except (json.JSONDecodeError, ValueError):
        return {}


def project_dir(payload: dict) -> str | None:
    return payload.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR")


def _block_text(content) -> str:
    """Extract plain text from a transcript message's content field, which
    may be a string or a list of typed blocks. Unknown shapes -> ''."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p)
    return ""


def transcript_messages(transcript_path: str | None) -> list[tuple[str, str]]:
    """Parse a Claude Code JSONL transcript into (role, text) pairs, in order.
    Defensive: unreadable file or unknown line shapes are skipped, not fatal."""
    if not transcript_path:
        return []
    path = Path(transcript_path)
    if not path.exists():
        return []
    out: list[tuple[str, str]] = []
    try:
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") not in ("user", "assistant"):
                continue
            msg = entry.get("message") or {}
            text = _block_text(msg.get("content"))
            if text.strip():
                out.append((entry["type"], text.strip()))
    except OSError:
        return []
    return out


def clip(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 2].rstrip() + " …"


def session_extract(
    messages: list[tuple[str, str]],
    n_user: int = 8,
    user_chars: int = 300,
    assistant_chars: int = 2000,
) -> str:
    """Deterministic extract of a session: recent user requests + the last
    assistant message. Honest about being an EXTRACT, not a summary — no LLM
    in the hook path (chosen for speed and zero flake risk)."""
    users = [t for role, t in messages if role == "user"][-n_user:]
    assistants = [t for role, t in messages if role == "assistant"]
    parts = []
    if users:
        parts.append("Recent user requests:")
        parts.extend(f"{i}. {clip(u, user_chars)}" for i, u in enumerate(users, 1))
    if assistants:
        parts.append("Last assistant state:")
        parts.append(clip(assistants[-1], assistant_chars))
    return "\n".join(parts)
