"""Per-project dataset keying: memory from one project never pollutes another.

A project is identified by its directory. The dataset name is derived from
the *resolved* path — a readable slug plus a short hash, so two directories
named `api/` in different places never collide, and renam-in-place keeps
the hash honest (a moved project is a new dataset; if that matters, the old
dataset is still intact and inspectable).

Resolution order (first hit wins):

1. Explicit `project_dir` argument (what the Claude Code hooks pass — they
   receive the project cwd in the hook's stdin JSON / $CLAUDE_PROJECT_DIR).
2. `MEMORY_PROJECT_DIR` env var (set per-project in `.mcp.json`'s `env`
   block, see mcp_server/REGISTRATION.md — the MCP server process itself
   can't trust its own cwd because `uv run --directory` rewrites it to the
   engram repo).
3. Fallback: `"main_dataset"` — full backward compatibility with everything
   written before the pivot.

Deliberately NOT `Path.cwd()`: a wrong-but-plausible default that silently
keys every project's memory to the engram repo is worse than an explicit
fallback dataset.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

DEFAULT_DATASET = "main_dataset"


def dataset_for(project_dir: str | os.PathLike | None = None) -> str:
    """Stable dataset name for a project directory (see module docstring)."""
    raw = project_dir or os.environ.get("MEMORY_PROJECT_DIR")
    if not raw:
        return DEFAULT_DATASET
    path = Path(raw).expanduser().resolve()
    slug = re.sub(r"[^a-z0-9]+", "-", path.name.lower()).strip("-") or "project"
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    return f"proj_{slug}_{digest}"
