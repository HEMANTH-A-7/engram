"""Bi-temporal conflict resolver: supersede-not-overwrite fact versioning.

Cognee's own knowledge graph is built by a non-deterministic LLM extraction
pass (`cognify()`) — entity names and edge labels can vary run to run, so
there's no stable key to detect "this new fact contradicts that old edge"
directly on Cognee's graph. This module owns that instead: a small SQLite
table of `(subject, relation, object)` facts, keyed on a deterministic
canonicalization of `(subject, relation)`, versioned bi-temporally
(`event_time` = when the fact became true, `ingestion_time` = when we
learned it). A conflicting write never overwrites the old row — it's marked
`valid_to`/`superseded_by` and kept for history, and a fresh row becomes the
current version.

Pure Python, no LLM calls, fully unit-testable on its own (see
tests/test_resolver.py). `core/ingest.py` is the orchestration layer that
extracts `(subject, relation, object)` from raw text and keeps Cognee's
index in sync with `current_facts()`.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.config import REPO_ROOT

DB_PATH = REPO_ROOT / ".resolver_data" / "facts.db"

_configured = False
_db_path = DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset TEXT NOT NULL,
    canonical_key TEXT NOT NULL,
    subject TEXT NOT NULL,
    relation TEXT NOT NULL,
    object TEXT NOT NULL,
    text TEXT NOT NULL,
    event_time TEXT NOT NULL,
    ingestion_time TEXT NOT NULL,
    valid_to TEXT,
    superseded_by INTEGER
);
CREATE INDEX IF NOT EXISTS idx_facts_current
    ON facts (dataset, canonical_key, valid_to);
"""


@dataclass(frozen=True)
class Fact:
    id: int
    dataset: str
    subject: str
    relation: str
    object: str
    text: str
    event_time: str
    ingestion_time: str
    valid_to: str | None
    superseded_by: int | None


@dataclass(frozen=True)
class WriteResult:
    fact: Fact
    superseded: Fact | None
    changed: bool  # False when the write restates the already-current fact


def configure(db_path: Path | None = None) -> None:
    """Pin the resolver's SQLite store into the repo. Idempotent."""
    global _configured, _db_path
    _db_path = db_path or DB_PATH
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript(_SCHEMA)
    _configured = True


def _ensure_configured() -> None:
    if not _configured:
        configure()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path)
    conn.row_factory = sqlite3.Row
    return conn


def canonical_key(subject: str, relation: str) -> str:
    """Deterministic conflict-detection key: lowercase, whitespace-collapsed."""

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower())

    return f"{norm(subject)}::{norm(relation)}"


def _row_to_fact(row: sqlite3.Row) -> Fact:
    return Fact(
        id=row["id"],
        dataset=row["dataset"],
        subject=row["subject"],
        relation=row["relation"],
        object=row["object"],
        text=row["text"],
        event_time=row["event_time"],
        ingestion_time=row["ingestion_time"],
        valid_to=row["valid_to"],
        superseded_by=row["superseded_by"],
    )


def write(
    subject: str,
    relation: str,
    object_: str,
    text: str,
    dataset: str = "main_dataset",
    event_time: str | None = None,
) -> WriteResult:
    """Record a fact, superseding any conflicting current version.

    A "conflict" is a currently-valid row with the same canonical
    `(subject, relation)` key but a different `object`. The old row is never
    overwritten — it's marked `valid_to`/`superseded_by` and kept for
    history. Restating the same `(subject, relation, object)` is a no-op
    (idempotent): no new version, `changed=False`.
    """
    _ensure_configured()
    now = datetime.now(timezone.utc).isoformat()
    event_time = event_time or now
    key = canonical_key(subject, relation)

    with _connect() as conn:
        cur = conn.execute(
            """
            SELECT * FROM facts
            WHERE dataset = ? AND canonical_key = ? AND valid_to IS NULL
            ORDER BY id DESC LIMIT 1
            """,
            (dataset, key),
        )
        current_row = cur.fetchone()
        current = _row_to_fact(current_row) if current_row else None

        if current is not None and current.object.strip().lower() == object_.strip().lower():
            return WriteResult(fact=current, superseded=None, changed=False)

        cur = conn.execute(
            """
            INSERT INTO facts
                (dataset, canonical_key, subject, relation, object, text,
                 event_time, ingestion_time, valid_to, superseded_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
            """,
            (dataset, key, subject, relation, object_, text, event_time, now),
        )
        new_id = cur.lastrowid
        new_fact = Fact(
            id=new_id,
            dataset=dataset,
            subject=subject,
            relation=relation,
            object=object_,
            text=text,
            event_time=event_time,
            ingestion_time=now,
            valid_to=None,
            superseded_by=None,
        )

        superseded = None
        if current is not None:
            conn.execute(
                "UPDATE facts SET valid_to = ?, superseded_by = ? WHERE id = ?",
                (event_time, new_id, current.id),
            )
            superseded = Fact(
                id=current.id,
                dataset=current.dataset,
                subject=current.subject,
                relation=current.relation,
                object=current.object,
                text=current.text,
                event_time=current.event_time,
                ingestion_time=current.ingestion_time,
                valid_to=event_time,
                superseded_by=new_id,
            )
        conn.commit()

    return WriteResult(fact=new_fact, superseded=superseded, changed=True)


def current_facts(dataset: str = "main_dataset") -> list[Fact]:
    """All currently-valid facts (not superseded) for a dataset, in write order."""
    _ensure_configured()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM facts WHERE dataset = ? AND valid_to IS NULL ORDER BY id",
            (dataset,),
        ).fetchall()
    return [_row_to_fact(r) for r in rows]


def history(subject: str, relation: str, dataset: str = "main_dataset") -> list[Fact]:
    """All versions (current + superseded) of a `(subject, relation)` fact, oldest first."""
    _ensure_configured()
    key = canonical_key(subject, relation)
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM facts WHERE dataset = ? AND canonical_key = ? ORDER BY id",
            (dataset, key),
        ).fetchall()
    return [_row_to_fact(r) for r in rows]


def reset(dataset: str | None = None) -> None:
    """Clear facts. Scoped to one dataset, or everything if omitted."""
    _ensure_configured()
    with _connect() as conn:
        if dataset is None:
            conn.execute("DELETE FROM facts")
        else:
            conn.execute("DELETE FROM facts WHERE dataset = ?", (dataset,))
        conn.commit()
