"""Fast tests for the security-hardening pass: cross-process pipeline lock,
private-data file permissions, and MCP tool input validation.

All LLM/Cognee-free: the lock is exercised against a tmp_path lock file with a
real second process (`flock` semantics can't be faked in-process), permissions
against a tmp_path resolver DB, and the validation paths fail before any
pipeline call is reached.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from core import locks, resolver
from mcp_server import server

# Tries to take the lock non-blocking from a *separate* process; exits 42 if
# it's held elsewhere, 0 if it acquired.
_PROBE = (
    "import fcntl, os, sys\n"
    "fd = os.open(sys.argv[1], os.O_CREAT | os.O_RDWR, 0o600)\n"
    "try:\n"
    "    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
    "except BlockingIOError:\n"
    "    sys.exit(42)\n"
    "sys.exit(0)\n"
)


@pytest.mark.asyncio
async def test_pipeline_lock_excludes_other_processes_while_held(tmp_path):
    path = tmp_path / "pipeline.lock"
    async with locks.pipeline_lock(path):
        held = subprocess.run([sys.executable, "-c", _PROBE, str(path)])
        assert held.returncode == 42  # another process must not get it
    released = subprocess.run([sys.executable, "-c", _PROBE, str(path)])
    assert released.returncode == 0  # and must get it once we let go


@pytest.mark.asyncio
async def test_pipeline_lock_file_is_owner_only(tmp_path):
    path = tmp_path / "nested" / "pipeline.lock"  # parent dir auto-created
    async with locks.pipeline_lock(path):
        assert (path.stat().st_mode & 0o777) == 0o600


def test_resolver_store_is_owner_only(tmp_path):
    db = tmp_path / "resolver_data" / "facts.db"
    resolver.configure(db_path=db)
    assert (db.parent.stat().st_mode & 0o777) == 0o700
    assert (db.stat().st_mode & 0o777) == 0o600


@pytest.mark.asyncio
async def test_memory_write_rejects_malformed_event_time(tmp_path):
    resolver.configure(db_path=tmp_path / "facts.db")
    # Must fail loud *before* any LLM/pipeline call -- this test would hang on
    # Ollama otherwise, which is itself the assertion that validation is first.
    with pytest.raises(ValueError, match="event_time"):
        await server.memory_write(
            "Alice lives in Boston.", metadata={"event_time": "not-a-timestamp"}
        )


@pytest.mark.asyncio
async def test_memory_search_clamps_k(tmp_path, monkeypatch):
    resolver.configure(db_path=tmp_path / "facts.db")
    seen: list[int] = []

    def _fake_search(dataset, query, k=5):
        seen.append(k)
        return server.index.SearchResult(hits=[], mode="keyword")

    monkeypatch.setattr(server.index, "search", _fake_search)

    await server.memory_search("anything", k=-3)
    await server.memory_search("anything", k=10_000)
    assert seen == [1, server.MAX_SEARCH_K]
