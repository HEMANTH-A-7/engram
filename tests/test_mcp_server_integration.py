"""Integration tests for mcp_server/server.py -- real Ollama + Cognee.

Marked `integration` — run explicitly with:

    uv run pytest -m integration -q tests/test_mcp_server_integration.py

Two layers of proof:

- **In-process tool flow**: calls the tool coroutines directly through a full
  write -> search -> stats -> forget cycle against the real pipeline. Proves
  the tools do the right thing end to end (a written fact is retrievable, a
  forgotten one stops being retrievable, stats reflect live state).
- **Real stdio round-trip**: spawns the actual server *binary* as a
  subprocess and drives it with an MCP client over stdio, asserting it
  advertises all four tools. Proves "the MCP server works" -- not just that
  the functions work when imported.
"""

from __future__ import annotations

import pytest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from core import resolver, router
from core.config import REPO_ROOT
from mcp_server import server

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_write_search_stats_forget_end_to_end():
    dataset = "test_mcp_dataset"
    resolver.reset(dataset)
    router.reset_ledger()

    written = await server.memory_write(
        "Alice lives in Boston.", metadata={"dataset": dataset}
    )
    assert written["changed"] is True
    assert written["subject"].strip()

    found = await server.memory_search("Where does Alice live?", k=3, dataset=dataset)
    joined = " ".join(found["hits"]).lower()
    assert "boston" in joined, f"written fact not searchable: {found['hits']!r}"

    stats = await server.memory_stats(dataset)
    assert stats["facts"]["total_current"] >= 1
    assert stats["cost"]["total_calls"] >= 1  # the write ran at least one extract

    fact_id = resolver.current_facts(dataset)[0].id
    forgotten = await server.memory_forget(fact_id, dataset=dataset)
    assert forgotten["found"] is True
    assert forgotten["evicted_id"] == fact_id

    after = await server.memory_search("Where does Alice live?", k=3, dataset=dataset)
    joined_after = " ".join(after["hits"]).lower()
    assert "boston" not in joined_after, f"forgotten fact still searchable: {after['hits']!r}"


@pytest.mark.asyncio
async def test_server_advertises_all_four_tools_over_stdio():
    params = StdioServerParameters(
        command="uv",
        args=["run", "--directory", str(REPO_ROOT), "python", "-m", "mcp_server.server"],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()

    names = {t.name for t in result.tools}
    assert {"memory_write", "memory_search", "memory_stats", "memory_forget"} <= names
