"""Fast unit tests for mcp_server/server.py — no Ollama, no Cognee, no LLM.

Since engram-lite (Session 10) the server imports NEITHER Cognee nor the
local-LLM stack at startup, so the entire six-tool surface runs here with
just the fake embedder: memory_write takes the agent-supplied `triple`
(exactly how a real client is instructed to call it), and search works in
keyword or hybrid mode. Every assertion also confirms the return value is
JSON-serializable, since these cross an MCP (JSON-RPC) boundary.
"""

from __future__ import annotations

import json

import pytest

from core import embeddings, notes, resolver
from mcp_server import server
from tests.fake_embed import broken_embed, fake_embed


@pytest.fixture(autouse=True)
def _fresh(tmp_path, monkeypatch):
    resolver.configure(db_path=tmp_path / "facts.db")
    embeddings.set_embed_fn(fake_embed)
    monkeypatch.delenv("MEMORY_PROJECT_DIR", raising=False)
    # Keep the cross-process flock inside tmp too.
    monkeypatch.setattr("core.locks.LOCK_PATH", tmp_path / "pipeline.lock")
    yield
    embeddings.set_embed_fn(None)


def _triple(subject, relation, object_):
    return {"subject": subject, "relation": relation, "object": object_}


# ---------------------------------------------------------------- memory_note

async def test_memory_note_is_written_and_searchable():
    result = await server.memory_note(
        "Chose flock for cross-process locking.", kind="decision", dataset="ds"
    )

    assert result["changed"] is True
    assert result["embedded"] is True
    search = await server.memory_search("what did we choose for locking", dataset="ds")
    assert search["hits"] == ["Chose flock for cross-process locking."]
    assert search["hit_facts"][0]["source"] == "note"
    assert search["hit_facts"][0]["id"] == result["id"]
    json.dumps(result), json.dumps(search)


async def test_memory_note_keyed_upsert():
    first = await server.memory_note("state v1", dataset="ds", key="session:x")
    second = await server.memory_note("state v2", dataset="ds", key="session:x")

    assert second["superseded_note_id"] == first["id"]
    assert [n.text for n in notes.current_notes("ds")] == ["state v2"]
    json.dumps(second)


async def test_memory_note_rejects_bad_kind():
    with pytest.raises(ValueError):
        await server.memory_note("text", kind="musing", dataset="ds")


# --------------------------------------------------------------- memory_write

async def test_memory_write_with_agent_triple_no_llm():
    result = await server.memory_write(
        "Alice lives in Boston.",
        metadata={"dataset": "ds"},
        triple=_triple("Alice", "lives_in", "Boston"),
    )

    assert result["changed"] is True
    assert result["extraction"] == "agent"  # zero LLM calls
    assert result["indexed"] is True
    assert result["cognify_attempts"] == 0
    search = await server.memory_search("Where does Alice live?", dataset="ds")
    assert search["hits"][0] == "Alice lives in Boston."
    json.dumps(result)


async def test_memory_write_supersede_removes_old_text_from_index():
    await server.memory_write(
        "Alice lives in Boston.",
        metadata={"dataset": "ds"},
        triple=_triple("Alice", "lives_in", "Boston"),
    )
    result = await server.memory_write(
        "Alice now lives in Seattle.",
        metadata={"dataset": "ds"},
        triple=_triple("Alice", "lives_in", "Seattle"),
    )

    assert result["superseded"]["object"] == "Boston"
    search = await server.memory_search("Where does Alice live?", dataset="ds", k=10)
    assert "Alice lives in Boston." not in search["hits"]  # never searchable again
    assert "Alice now lives in Seattle." in search["hits"]
    json.dumps(result)


async def test_memory_write_validates_triple_fields():
    for bad in (
        {"subject": "", "relation": "r", "object": "o"},
        {"subject": "s", "relation": None, "object": "o"},
        {"subject": "s", "relation": "r"},
    ):
        with pytest.raises(ValueError, match="triple"):
            await server.memory_write("text", metadata={"dataset": "ds"}, triple=bad)


async def test_memory_write_rejects_malformed_event_time():
    with pytest.raises(ValueError, match="event_time"):
        await server.memory_write(
            "Alice lives in Boston.",
            metadata={"event_time": "not-a-date"},
            triple=_triple("Alice", "lives_in", "Boston"),
        )


# ------------------------------------------------------------- memory_context

async def test_memory_context_digest_and_measurement():
    await server.memory_write(
        "Alice lives in Seattle.",
        metadata={"dataset": "ds"},
        triple=_triple("Alice", "lives_in", "Seattle"),
    )
    await server.memory_note("Gotcha: uv rewrites cwd.", kind="gotcha", dataset="ds")

    result = await server.memory_context(dataset="ds", token_budget=500)

    assert "Seattle" in result["digest"]
    assert "uv rewrites cwd" in result["digest"]
    assert result["digest_tokens"] <= 500
    # NB: on tiny datasets the digest's markdown scaffolding can exceed raw
    # history (documented in test_context.py) — assert only that the
    # measurement is present and sane, not that savings exist at n=2.
    assert result["raw_history_tokens"] > 0
    assert result["included"] == 2
    json.dumps(result)


# -------------------------------------------------------------- memory_search

async def test_memory_search_per_project_isolation():
    await server.memory_note("alpha detail", dataset="proj_a")
    await server.memory_note("beta detail", dataset="proj_b")

    result = await server.memory_search("detail", dataset="proj_a", k=10)

    assert result["hits"] == ["alpha detail"]
    json.dumps(result)


async def test_memory_search_empty_index_returns_empty_lists():
    result = await server.memory_search("anything", dataset="empty_ds")
    assert result["hits"] == [] and result["hit_facts"] == []
    json.dumps(result)


async def test_memory_search_keyword_mode_when_endpoint_down():
    await server.memory_note("remembered thing", dataset="ds")
    embeddings.set_embed_fn(broken_embed)

    result = await server.memory_search("remembered thing", dataset="ds")

    assert result["ranking"] == "keyword"  # honest mode report, no failure
    assert result["hits"] == ["remembered thing"]
    json.dumps(result)


async def test_memory_search_records_access():
    note = await server.memory_note("accessed note", dataset="ds")
    await server.memory_search("accessed note", dataset="ds")

    assert notes.get_note(note["id"], "ds").access_count == 1


# -------------------------------------------------------------- memory_forget

async def test_memory_forget_note_round_trip():
    note = await server.memory_note("to be forgotten", dataset="ds")
    result = await server.memory_forget(id=note["id"], dataset="ds", kind="note")

    assert result["found"] is True
    search = await server.memory_search("to be forgotten", dataset="ds")
    assert search["hits"] == []
    json.dumps(result)


async def test_memory_forget_fact_round_trip():
    written = await server.memory_write(
        "Bob works at Acme.",
        metadata={"dataset": "ds"},
        triple=_triple("Bob", "works_at", "Acme"),
    )
    found = await server.memory_search("Where does Bob work?", dataset="ds")
    fact_id = found["hit_facts"][0]["id"]
    assert found["hit_facts"][0]["source"] == "fact"

    result = await server.memory_forget(id=fact_id, dataset="ds", kind="fact")

    assert result["found"] is True
    after = await server.memory_search("Where does Bob work?", dataset="ds")
    assert after["hits"] == []
    assert written["changed"] is True
    json.dumps(result)


async def test_memory_forget_missing_id_returns_not_found():
    await server.memory_note("keep me", dataset="ds")
    result = await server.memory_forget(id=999_999, dataset="ds")

    assert result["found"] is False and result["evicted_id"] is None
    assert len(notes.current_notes("ds")) == 1
    json.dumps(result)


# --------------------------------------------------------------- memory_stats

async def test_memory_stats_new_shape():
    await server.memory_write(
        "Alice lives in Boston.",
        metadata={"dataset": "ds"},
        triple=_triple("Alice", "lives_in", "Boston"),
    )
    await server.memory_note("a decision", kind="decision", dataset="ds")

    stats = await server.memory_stats(dataset="ds")

    assert stats["facts"]["total_current"] == 1
    assert stats["notes"]["total_current"] == 1
    assert stats["notes"]["by_kind"]["decision"] == 1
    assert stats["index"]["indexed"] == 2
    assert stats["index"]["pending_embeddings"] == 0
    assert "pricing_note" in stats["cost"]  # present with or without the LLM stack
    assert isinstance(stats["benchmarks"], dict)
    json.dumps(stats)


# ------------------------------------------------------------ project keying

async def test_tools_key_dataset_from_project_dir(tmp_path):
    project = tmp_path / "myproj"
    project.mkdir()
    await server.memory_note("project-scoped note", project_dir=str(project))

    from core import projects

    ds = projects.dataset_for(project)
    assert [n.text for n in notes.current_notes(ds)] == ["project-scoped note"]
    # And main_dataset was NOT polluted.
    assert notes.current_notes("main_dataset") == []
