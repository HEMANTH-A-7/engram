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

Pure Python for the common case (no LLM calls), fully unit-testable on its
own (see tests/test_resolver.py). Since Bucket 5, `write()` makes one LLM
call — `core/router.py`'s `judge_conflict` — but only when a conflict looks
*ambiguous* (old/new object strings overlap); the common clean-contradiction
case stays fully deterministic. `core/ingest.py` is the orchestration layer
that extracts `(subject, relation, object)` from raw text and keeps Cognee's
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
    superseded_by INTEGER,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT,
    tier TEXT NOT NULL DEFAULT 'hot',
    importance REAL NOT NULL DEFAULT 0.5,
    evicted_at TEXT,
    summary TEXT
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
    access_count: int = 0
    last_accessed: str | None = None
    tier: str = "hot"
    importance: float = 0.5
    evicted_at: str | None = None
    summary: str | None = None


@dataclass(frozen=True)
class WriteResult:
    fact: Fact
    superseded: Fact | None
    changed: bool  # False when the write restates the already-current fact
    revived_from_eviction: bool = False  # True when this key was previously evicted
    distinct: bool = False  # True when an ambiguous conflict was judged non-conflicting


_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("access_count", "ALTER TABLE facts ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0"),
    ("last_accessed", "ALTER TABLE facts ADD COLUMN last_accessed TEXT"),
    ("tier", "ALTER TABLE facts ADD COLUMN tier TEXT NOT NULL DEFAULT 'hot'"),
    ("importance", "ALTER TABLE facts ADD COLUMN importance REAL NOT NULL DEFAULT 0.5"),
    ("evicted_at", "ALTER TABLE facts ADD COLUMN evicted_at TEXT"),
    ("summary", "ALTER TABLE facts ADD COLUMN summary TEXT"),
)


def _migrate(conn: sqlite3.Connection) -> None:
    """Add Bucket 4 columns to a facts table created before they existed.

    `CREATE TABLE IF NOT EXISTS` in `_SCHEMA` only applies to brand-new
    databases -- an existing `.resolver_data/facts.db` from before Bucket 4
    keeps its old column set forever unless migrated explicitly.
    """
    existing = {row[1] for row in conn.execute("PRAGMA table_info(facts)")}
    for column, ddl in _MIGRATIONS:
        if column not in existing:
            conn.execute(ddl)


def configure(db_path: Path | None = None) -> None:
    """Pin the resolver's SQLite store into the repo. Idempotent."""
    global _configured, _db_path
    _db_path = db_path or DB_PATH
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    # The DB holds private memory contents -- keep it owner-only rather than
    # the default umask (which leaves it world-readable on a shared machine).
    _db_path.parent.chmod(0o700)
    with _connect() as conn:
        conn.executescript(_SCHEMA)
        _migrate(conn)
        conn.commit()
    _db_path.chmod(0o600)
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
        access_count=row["access_count"],
        last_accessed=row["last_accessed"],
        tier=row["tier"],
        importance=row["importance"],
        evicted_at=row["evicted_at"],
        summary=row["summary"],
    )


def _is_ambiguous(old_object: str, new_object: str) -> bool:
    """Cheap ambiguity proxy: is one object string a substring of the other?

    Catches refinements/typos ("Boston" -> "Boston, MA") that a clean
    contradiction check ("Boston" -> "Seattle") wouldn't flag. This is a
    substring-overlap heuristic, not semantic similarity — most conflicts
    (the common case) have zero string overlap and stay fully deterministic,
    zero LLM calls, matching "routine" vs "ambiguous" from the spec text
    directly.
    """
    a, b = old_object.strip().lower(), new_object.strip().lower()
    return a != b and (a in b or b in a)


async def write(
    subject: str,
    relation: str,
    object_: str,
    text: str,
    dataset: str = "main_dataset",
    event_time: str | None = None,
    importance: float | None = None,
) -> WriteResult:
    """Record a fact, superseding any conflicting current version.

    A "conflict" is a currently-valid row with the same canonical
    `(subject, relation)` key but a different `object`. The old row is never
    overwritten — it's marked `valid_to`/`superseded_by` and kept for
    history. Restating the same `(subject, relation, object)` is a no-op
    (idempotent): no new version, `changed=False`.

    Most conflicts are clean (no string overlap between old/new object) and
    stay fully deterministic — an unconditional supersede, no LLM call. When
    the two object strings overlap (`_is_ambiguous`), that's a proxy for
    "this might be a refinement or restatement, not a real change", and
    `core/router.py`'s `judge_conflict` (Bucket 5, always the large model)
    is asked to decide: `"update"` supersedes as normal, `"same"` is treated
    as the idempotent no-op branch above, `"distinct"` means both values are
    true at once — the new fact is written as a fresh current row without
    closing the old one (`WriteResult.distinct=True`), so both remain
    queryable rather than one incorrectly winning.

    `importance` defaults to a neutral 0.5 when omitted — there's no real
    importance signal wired in yet (Bucket 4), this just leaves the field
    ready for one rather than faking a heuristic.

    If the canonical key was previously evicted (Bucket 4 consolidation)
    rather than superseded, this write is a *revival* — the system forgot
    something that turned out to still matter. Flagged via
    `WriteResult.revived_from_eviction` for the regret-rate metric.
    """
    _ensure_configured()
    now = datetime.now(timezone.utc).isoformat()
    event_time = event_time or now
    key = canonical_key(subject, relation)
    fact_importance = 0.5 if importance is None else importance

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

    distinct = False
    if current is not None and _is_ambiguous(current.object, object_):
        from core import router  # local import: avoids a resolver<->router cycle at module load

        verdict = await router.judge_conflict(current.text, text)
        if verdict == "same":
            return WriteResult(fact=current, superseded=None, changed=False)
        distinct = verdict == "distinct"

    with _connect() as conn:
        revived = False
        if current is None:
            cur = conn.execute(
                """
                SELECT * FROM facts
                WHERE dataset = ? AND canonical_key = ? AND evicted_at IS NOT NULL
                ORDER BY id DESC LIMIT 1
                """,
                (dataset, key),
            )
            revived = cur.fetchone() is not None

        cur = conn.execute(
            """
            INSERT INTO facts
                (dataset, canonical_key, subject, relation, object, text,
                 event_time, ingestion_time, valid_to, superseded_by,
                 access_count, last_accessed, tier, importance, evicted_at, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, 0, NULL, 'hot', ?, NULL, NULL)
            """,
            (dataset, key, subject, relation, object_, text, event_time, now, fact_importance),
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
            access_count=0,
            last_accessed=None,
            tier="hot",
            importance=fact_importance,
            evicted_at=None,
        )

        superseded = None
        if current is not None and not distinct:
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
                access_count=current.access_count,
                last_accessed=current.last_accessed,
                tier=current.tier,
                importance=current.importance,
                evicted_at=current.evicted_at,
            )
        conn.commit()

    return WriteResult(
        fact=new_fact,
        superseded=superseded,
        changed=True,
        revived_from_eviction=revived,
        distinct=distinct,
    )


def current_facts(dataset: str = "main_dataset", tiers: set[str] | None = None) -> list[Fact]:
    """All currently-valid facts (not superseded, not evicted) for a dataset.

    `valid_to IS NULL` already excludes evicted facts for free — `evict()`
    stamps `valid_to` exactly like a supersede does, just with no
    `superseded_by`. `tiers` optionally restricts to a subset (e.g. Bucket
    4's `consolidation.resync_active()` passes `{"hot"}`); omitted means
    all valid facts regardless of tier, so this stays backward-compatible
    with Bucket 3's `core/ingest.py` callsite.
    """
    _ensure_configured()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM facts WHERE dataset = ? AND valid_to IS NULL ORDER BY id",
            (dataset,),
        ).fetchall()
    facts = [_row_to_fact(r) for r in rows]
    if tiers is not None:
        facts = [f for f in facts if f.tier in tiers]
    return facts


def all_versions(dataset: str = "main_dataset") -> list[Fact]:
    """Every fact row for a dataset -- current, superseded, and evicted -- in
    insertion order.

    Read-only counterpart to `current_facts` for the "raw history vs. memory
    layer" dashboard metrics: `current_facts` is what the layer surfaces now,
    while this is the full write history (every version ever recorded). The
    difference between the two is exactly what supersede-not-overwrite saves.
    Ordered by `id` so callers can replay writes chronologically.
    """
    _ensure_configured()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM facts WHERE dataset = ? ORDER BY id",
            (dataset,),
        ).fetchall()
    return [_row_to_fact(r) for r in rows]


def record_access(fact_id: int, dataset: str = "main_dataset", now: str | None = None) -> None:
    """Bump `access_count` and set `last_accessed` for a fact that was retrieved."""
    _ensure_configured()
    now = now or datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE facts SET access_count = access_count + 1, last_accessed = ?
            WHERE id = ? AND dataset = ?
            """,
            (now, fact_id, dataset),
        )
        conn.commit()


def set_tier(fact_id: int, tier: str, dataset: str = "main_dataset") -> None:
    """Set a fact's consolidation tier ('hot' / 'warm' / 'cold')."""
    _ensure_configured()
    with _connect() as conn:
        conn.execute(
            "UPDATE facts SET tier = ? WHERE id = ? AND dataset = ?",
            (tier, fact_id, dataset),
        )
        conn.commit()


def set_summary(fact_id: int, summary: str, dataset: str = "main_dataset") -> None:
    """Persist a fact's LLM-generated summary (Bucket 5 consolidation)."""
    _ensure_configured()
    with _connect() as conn:
        conn.execute(
            "UPDATE facts SET summary = ? WHERE id = ? AND dataset = ?",
            (summary, fact_id, dataset),
        )
        conn.commit()


def evict(fact_id: int, dataset: str = "main_dataset", now: str | None = None) -> None:
    """Physically evict a fact: stamp `valid_to`/`evicted_at`, tier -> 'cold'.

    Reuses the same `valid_to` column supersede uses so `current_facts()`
    excludes it for free, but leaves `superseded_by` NULL and sets
    `evicted_at` so a later write to the same canonical key can be detected
    as a *revival* (see `write()`) rather than mistaken for an ordinary
    fresh fact. The row and its `text` are kept, not deleted — consolidation
    in this project approximates "archival" as index-exclusion, not
    destruction, so evicted facts remain inspectable for debugging/regret
    analysis.
    """
    _ensure_configured()
    now = now or datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE facts SET valid_to = ?, evicted_at = ?, tier = 'cold'
            WHERE id = ? AND dataset = ?
            """,
            (now, now, fact_id, dataset),
        )
        conn.commit()


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
