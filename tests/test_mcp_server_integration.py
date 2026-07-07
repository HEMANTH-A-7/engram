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

import json
from datetime import timedelta

import pytest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from core import index, resolver, router
from core.config import REPO_ROOT
from mcp_server import server

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_write_search_stats_forget_end_to_end():
    dataset = "test_mcp_dataset"
    resolver.reset(dataset)
    index.remove_dataset(dataset)  # post-pivot: the index persists per-dataset
    router.reset_ledger()

    written = await server.memory_write(
        "Alice lives in Boston.", metadata={"dataset": dataset}
    )
    assert written["changed"] is True
    assert written["subject"].strip()

    found = await server.memory_search("Where does Alice live?", k=3, dataset=dataset)
    joined = " ".join(found["hits"]).lower()
    assert "boston" in joined, f"written fact not searchable: {found['hits']!r}"
    # hit_facts is index-aligned with hits and carries the id a client needs.
    assert len(found["hit_facts"]) == len(found["hits"])
    assert all(hf["text"] == t for hf, t in zip(found["hit_facts"], found["hits"]))

    stats = await server.memory_stats(dataset)
    assert stats["facts"]["total_current"] >= 1
    assert stats["cost"]["total_calls"] >= 1  # the write ran at least one extract

    # Discover the id the way a real client must -- from the search hit itself,
    # not by reaching into the resolver -- then forget it. This is the whole
    # point of hit_facts: memory_search -> memory_forget with no side channel.
    boston_hit = next(hf for hf in found["hit_facts"] if "boston" in hf["text"].lower())
    fact_id = boston_hit["id"]
    assert fact_id is not None, f"hit did not resolve to an id: {boston_hit!r}"
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
    assert {
        "memory_write",
        "memory_search",
        "memory_stats",
        "memory_forget",
        "memory_note",
        "memory_context",
    } <= names


def _tool_payload(result) -> dict:
    """Extract a tool's dict return from a CallToolResult, asserting the wire held.

    The whole point of this round-trip is that the JSON-RPC channel stayed
    clean, so we assert the protocol-level success flag and that the content
    block parses as JSON -- a corrupted wire shows up as either an exception
    before we get here or a malformed/absent content block.
    """
    assert result.isError is False, f"tool call reported an error: {result.content!r}"
    assert result.content, "tool returned no content blocks"
    text = getattr(result.content[0], "text", None)
    assert text is not None, f"expected a text content block, got {result.content[0]!r}"
    return json.loads(text)


@pytest.mark.asyncio
async def test_memory_write_round_trip_keeps_the_wire_clean():
    """A real stdio client must survive memory_write.

    POST-PIVOT NOTE (Session 9): memory_write no longer runs cognify() — the
    serving path is the owned incremental index. The test remains as the
    stdout-wire guard by construction (a real client validates every wire
    line), now exercising the extraction-LLM + incremental-index path. The
    original cognify-era rationale is preserved below for history.

    Original rationale — a real stdio client must survive memory_write, which ran cognify():

    This is the regression test for commit 0332cd7. memory_write is the tool
    that drives Cognee's cognify(), whose DB layer prints chatter ("table
    already exists, skipping creation") to stdout -- the exact stream MCP
    carries JSON-RPC on. Before the fix, a real stdio client rejected that
    chatter with a JSONRPCMessage ValidationError mid-write; the in-process
    tests above and the list_tools handshake never triggered a cognify(), so
    they couldn't catch it. Only a real-client call_tool("memory_write", ...)
    exercises the corruptible path.

    It's slow (~40s) and can hit gemma4's documented cognify flakiness, so we
    use a generous per-call timeout and tolerate a single retry. A genuine
    wire-corruption bug is deterministic (every cognify prints), so it fails
    both attempts; a flake usually clears on the retry. We assert only that
    the protocol survived and the payload shape is valid, never on the exact
    extracted values.

    Honesty caveat (verified 2026-07-03): this is a guard *by construction* --
    a real stdio client parses and validates every line on the wire, so any
    non-JSON-RPC byte written to the server's stdout during a tool call breaks
    it. It is NOT a demonstrated red->green for 0332cd7 in the current stack:
    reverting main() to `mcp.run(transport="stdio")` and re-running this test
    (warm, cold, single write, and full write/search/stats/forget) all stayed
    green, because Cognee 1.2.2 emits its DB chatter to *stderr*, not stdout,
    on these paths. The fix remains correct defense-in-depth; this test catches
    any reintroduction of stdout leakage regardless of which dependency causes
    it.
    """
    dataset = "test_mcp_stdio_write"
    resolver.reset(dataset)
    index.remove_dataset(dataset)

    params = StdioServerParameters(
        command="uv",
        args=["run", "--directory", str(REPO_ROOT), "python", "-m", "mcp_server.server"],
    )

    last_error: Exception | None = None
    for _attempt in range(2):  # tolerate one gemma4 flake; a wire bug fails both
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        "memory_write",
                        {
                            "content": "Alice lives in Boston.",
                            "metadata": {"dataset": dataset},
                        },
                        read_timeout_seconds=timedelta(seconds=180),
                    )
            payload = _tool_payload(result)
            assert isinstance(payload.get("subject"), str) and payload["subject"].strip()
            assert isinstance(payload["changed"], bool)
            return  # clean round-trip through a cognify() -- the wire held
        except Exception as exc:  # noqa: BLE001 -- retry once, then surface it
            last_error = exc
            resolver.reset(dataset)

    raise AssertionError(
        "memory_write stdio round-trip failed on both attempts; "
        f"last error: {last_error!r}"
    )
