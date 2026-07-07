"""Fast dev-context notes: raw text, no LLM extraction, sub-second writes.

The pivot's core insight: most of what matters while developing (decisions,
progress, gotchas, references) is prose, not `(subject, relation, object)`
triples. Forcing it through LLM extraction was both slow (multi-second to
minutes) and lossy. A note is stored verbatim: one SQLite insert plus one
embed call into `core/index.py` — measured in milliseconds-to-subsecond,
never an LLM.

Notes reuse the resolver's supersede-not-overwrite discipline where it makes
sense:

- **Exact duplicate** (same dataset+kind+text, still current) is an
  idempotent no-op — `changed=False`, mirroring `resolver.write`.
- **Keyed upsert**: `add_note(key=...)` supersedes the previous current note
  with the same key instead of accumulating. This is what the Stop hook uses
  to maintain ONE rolling "session state" note per session instead of one
  note per turn.
- Superseded/evicted notes keep their rows (`valid_to`/`evicted_at` stamped,
  same columns as facts) and are removed from the search index immediately —
  never searchable, still inspectable.

The triple path (`ingest.remember` → `resolver.write`) stays for discrete
facts where conflict detection on a canonical key earns its cost.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from core import index, resolver

KINDS = ("decision", "progress", "gotcha", "reference")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset TEXT NOT NULL,
    kind TEXT NOT NULL,
    key TEXT,                       -- optional upsert key (e.g. session id)
    text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT,
    valid_to TEXT,
    superseded_by INTEGER,
    evicted_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_notes_current ON notes (dataset, valid_to);
"""


@dataclass(frozen=True)
class Note:
    id: int
    dataset: str
    kind: str
    key: str | None
    text: str
    created_at: str
    access_count: int = 0
    last_accessed: str | None = None
    valid_to: str | None = None
    superseded_by: int | None = None
    evicted_at: str | None = None


@dataclass(frozen=True)
class NoteResult:
    note: Note
    changed: bool  # False = exact-duplicate no-op
    superseded_id: int | None = None  # keyed upsert replaced this note
    embedded: bool = True  # False = stored, embedding pending (endpoint down)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(resolver.db_path())
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _row_to_note(row: sqlite3.Row) -> Note:
    return Note(**{k: row[k] for k in Note.__dataclass_fields__})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_note(
    text: str,
    kind: str = "progress",
    dataset: str = "main_dataset",
    key: str | None = None,
) -> NoteResult:
    """Store a note verbatim and index it incrementally. Sub-second: no LLM,
    no index rebuild. See module docstring for duplicate/upsert semantics."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    text = text.strip()
    if not text:
        raise ValueError("note text must be non-empty")
    now = _now()

    with _connect() as conn:
        dup = conn.execute(
            "SELECT * FROM notes WHERE dataset = ? AND kind = ? AND text = ? "
            "AND valid_to IS NULL ORDER BY id DESC LIMIT 1",
            (dataset, kind, text),
        ).fetchone()
        if dup is not None:
            return NoteResult(note=_row_to_note(dup), changed=False)

        superseded_id: int | None = None
        if key is not None:
            prev = conn.execute(
                "SELECT id FROM notes WHERE dataset = ? AND key = ? "
                "AND valid_to IS NULL ORDER BY id DESC LIMIT 1",
                (dataset, key),
            ).fetchone()
            superseded_id = prev["id"] if prev else None

        cur = conn.execute(
            "INSERT INTO notes (dataset, kind, key, text, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (dataset, kind, key, text, now),
        )
        new_id = cur.lastrowid
        if superseded_id is not None:
            conn.execute(
                "UPDATE notes SET valid_to = ?, superseded_by = ? WHERE id = ?",
                (now, new_id, superseded_id),
            )
        conn.commit()

    if superseded_id is not None:
        index.remove_item(dataset, "note", superseded_id)
    embedded = index.index_item(dataset, "note", new_id, text)

    note = Note(
        id=new_id, dataset=dataset, kind=kind, key=key, text=text, created_at=now
    )
    return NoteResult(
        note=note, changed=True, superseded_id=superseded_id, embedded=embedded
    )


def current_notes(dataset: str = "main_dataset", kind: str | None = None) -> list[Note]:
    """All currently-valid notes (not superseded, not evicted), oldest first."""
    q = "SELECT * FROM notes WHERE dataset = ? AND valid_to IS NULL"
    args: list = [dataset]
    if kind is not None:
        q += " AND kind = ?"
        args.append(kind)
    with _connect() as conn:
        rows = conn.execute(q + " ORDER BY id", args).fetchall()
    return [_row_to_note(r) for r in rows]


def all_notes(dataset: str = "main_dataset") -> list[Note]:
    """Every note row (current + superseded + evicted) — the raw-history
    baseline for the digest-vs-raw metric, mirroring `resolver.all_versions`."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM notes WHERE dataset = ? ORDER BY id", (dataset,)
        ).fetchall()
    return [_row_to_note(r) for r in rows]


def get_note(note_id: int, dataset: str = "main_dataset") -> Note | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM notes WHERE id = ? AND dataset = ?", (note_id, dataset)
        ).fetchone()
    return _row_to_note(row) if row else None


def record_access(note_id: int, dataset: str = "main_dataset") -> None:
    """Bump access stats — same retrieval-keeps-it-warm signal as facts."""
    with _connect() as conn:
        conn.execute(
            "UPDATE notes SET access_count = access_count + 1, last_accessed = ? "
            "WHERE id = ? AND dataset = ?",
            (_now(), note_id, dataset),
        )
        conn.commit()


def evict_note(note_id: int, dataset: str = "main_dataset") -> bool:
    """Evict a note: stamp valid_to/evicted_at, drop from the index. Returns
    False if the id isn't a current note in this dataset (nothing touched)."""
    now = _now()
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE notes SET valid_to = ?, evicted_at = ? "
            "WHERE id = ? AND dataset = ? AND valid_to IS NULL",
            (now, now, note_id, dataset),
        )
        found = cur.rowcount > 0
        conn.commit()
    if found:
        index.remove_item(dataset, "note", note_id)
    return found
