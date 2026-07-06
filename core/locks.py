"""Cross-process pipeline lock (security-hardening pass).

`mcp_server.server`'s `_pipeline_lock` (an `asyncio.Lock`) serializes
`memory_write`/`memory_forget` *within* one server process -- but every MCP
client session spawns its own server process, and `store.reset_and_load`
wipes the *shared* Cognee data root. Two concurrent sessions (e.g. two Claude
Code windows open on this repo) could interleave reset+cognify destructively,
corrupting the index. This module adds the cross-process half: an exclusive
`flock(2)` on a lock file in the gitignored `.resolver_data/`, acquired in a
worker thread so the event loop never blocks while waiting.

POSIX-only (`fcntl`), matching the project's macOS/Linux local-first scope.
"""

from __future__ import annotations

import asyncio
import fcntl
import os
from contextlib import asynccontextmanager
from pathlib import Path

from core.config import REPO_ROOT

LOCK_PATH = REPO_ROOT / ".resolver_data" / "pipeline.lock"


@asynccontextmanager
async def pipeline_lock(path: Path | None = None):
    """Hold an exclusive cross-process lock around the reset+cognify pipeline.

    Blocks (in a thread, not on the event loop) until any other process
    releases it. Not reentrant within one process -- callers already
    serialize per-process via `mcp_server.server._pipeline_lock`, so this
    only ever guards against *other* processes.
    """
    path = path or LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        await asyncio.to_thread(fcntl.flock, fd, fcntl.LOCK_EX)
        yield
    finally:
        # flock is per open-file-description, so unlocking from the event-loop
        # thread after acquiring in a worker thread is well-defined.
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
