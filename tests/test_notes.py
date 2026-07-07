"""Unit tests for core/notes.py — the fast raw-text write path.

No Ollama, no Cognee (fake embedder). Proves the memory_note contract:
verbatim storage, kind validation, duplicate no-op, keyed upsert
(supersede-not-overwrite), eviction, and the endpoint-down pending path.
"""

from __future__ import annotations

import pytest

from core import embeddings, index, notes, resolver
from tests.fake_embed import broken_embed, fake_embed


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    resolver.configure(db_path=tmp_path / "facts.db")
    embeddings.set_embed_fn(fake_embed)
    yield
    embeddings.set_embed_fn(None)


def test_add_note_stores_verbatim_and_is_searchable():
    result = notes.add_note("Chose SQLite over LanceDB for the index.", kind="decision", dataset="ds")

    assert result.changed is True
    assert result.embedded is True
    assert result.note.kind == "decision"
    hits = index.search("ds", "why did we choose SQLite", k=1).hits
    assert hits[0].text == "Chose SQLite over LanceDB for the index."
    assert hits[0].kind == "note"


def test_invalid_kind_rejected():
    with pytest.raises(ValueError):
        notes.add_note("text", kind="musing", dataset="ds")


def test_empty_text_rejected():
    with pytest.raises(ValueError):
        notes.add_note("   ", kind="progress", dataset="ds")


def test_exact_duplicate_is_idempotent_noop():
    first = notes.add_note("same text", kind="gotcha", dataset="ds")
    second = notes.add_note("same text", kind="gotcha", dataset="ds")

    assert second.changed is False
    assert second.note.id == first.note.id
    assert len(notes.current_notes("ds")) == 1
    assert index.indexed_count("ds") == 1


def test_keyed_upsert_supersedes_previous_note():
    first = notes.add_note("session state v1", kind="progress", dataset="ds", key="session:x")
    second = notes.add_note("session state v2", kind="progress", dataset="ds", key="session:x")

    assert second.superseded_id == first.note.id
    current = notes.current_notes("ds")
    assert [n.text for n in current] == ["session state v2"]
    # Old row kept (supersede-not-overwrite), stamped, out of the index.
    old = notes.get_note(first.note.id, "ds")
    assert old.valid_to is not None
    assert old.superseded_by == second.note.id
    assert [h.text for h in index.search("ds", "session state", k=10).hits] == ["session state v2"]


def test_keys_are_per_dataset():
    notes.add_note("state in a", kind="progress", dataset="proj_a", key="session:x")
    result_b = notes.add_note("state in b", kind="progress", dataset="proj_b", key="session:x")

    assert result_b.superseded_id is None
    assert len(notes.current_notes("proj_a")) == 1
    assert len(notes.current_notes("proj_b")) == 1


def test_evict_note_removes_from_index_keeps_row():
    result = notes.add_note("obsolete detail", kind="reference", dataset="ds")

    assert notes.evict_note(result.note.id, dataset="ds") is True
    assert notes.current_notes("ds") == []
    assert notes.get_note(result.note.id, "ds").evicted_at is not None
    assert index.search("ds", "obsolete detail", k=5).hits == []
    # Second evict / unknown id: honest miss, nothing touched.
    assert notes.evict_note(result.note.id, dataset="ds") is False
    assert notes.evict_note(999_999, dataset="ds") is False


def test_note_lands_even_when_endpoint_down():
    embeddings.set_embed_fn(broken_embed)
    result = notes.add_note("written during outage", kind="progress", dataset="ds")

    assert result.changed is True
    assert result.embedded is False  # stored, embedding pending
    assert [n.text for n in notes.current_notes("ds")] == ["written during outage"]
