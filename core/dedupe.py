"""Near-duplicate subject/relation detection over extracted facts.

Why this exists: `resolver.canonical_key` is an *exact* match on normalized
`(subject, relation)`. Agent-supplied extraction is free-form, so the same
real-world attribute can arrive under two relation spellings — the Session 11
pair was `("Hemanth", "prefers", ...)` vs `("Hemanth", "storage_preference",
...)` — and the conflict machinery never fires: both stay current and the
digest contradicts itself.

Design (Session 12, user-ratified):
- D1: detection runs both at write time (`find_near_duplicate`, called from
  `resolver.write` on an exact-key miss) and as an explicit batch pass
  (`sweep`, via scripts/dedupe_pass.py).
- D2: with the optional LLM judge available, a detected pair is resolved like
  any ambiguous conflict (update/same/distinct). WITHOUT the judge — the
  no-LLM floor — a pair is only *flagged* (dupe_flags table, surfaced by
  `memory_stats`), never auto-merged. A wrong merge silently hides a fact;
  a flag just asks the user.
- D3: the similarity floor is deterministic and stdlib-only: same normalized
  subject AND relations share a content-token stem (light suffix stripping;
  "prefers" and "storage_preference" both yield "prefer") or are
  near-identical strings (difflib ratio >= 0.75). Validated against the real
  main_dataset vocabulary: catches the prefers/storage_preference pair,
  flags none of the six unrelated relations.

Fail-open like everything on the serving path: `resolver.write` wraps its
call in try/except — a dedupe bug must never break a write.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from difflib import SequenceMatcher

from core import resolver
from core.resolver import Fact

# Tokens that carry no attribute meaning on their own; matching on these
# would glue "lives_in" to "works_in".
_STOPWORDS = {
    "in", "at", "of", "to", "for", "on", "with", "by", "a", "an", "the",
    "is", "has", "was", "and", "or", "its", "their", "his", "her",
}

# Longest-first so "preferences" strips "ences" -> "prefer", not "s" first.
_SUFFIXES = (
    "ences", "ances", "ments", "tions", "sions",
    "ence", "ance", "ment", "tion", "sion",
    "ings", "ing", "ers", "ies", "es", "ed", "er", "s",
)

_RATIO_FLOOR = 0.8  # 0.75 lets antonym pairs through ("likes"/"dislikes" = 0.77)


def _stem(token: str) -> str:
    """Light suffix stripping — enough to unify inflections of one root
    ("prefers"/"preference" -> "prefer"), deliberately not a real stemmer."""
    for suffix in _SUFFIXES:
        if token.endswith(suffix):
            stripped = token[: -len(suffix)]
            if len(stripped) >= 3:
                return stripped
    return token


def _norm(s: str) -> str:
    import re

    return re.sub(r"\s+", " ", s.strip().lower())


def _content_stems(relation: str) -> set[str]:
    import re

    tokens = [t for t in re.split(r"[^a-z0-9]+", relation.lower()) if t]
    return {_stem(t) for t in tokens if t not in _STOPWORDS and len(t) > 2}


def _stems_match(stems_a: set[str], stems_b: set[str]) -> bool:
    """Equal stems, or prefix-compatible long stems. The light stemmer can
    land on different cut points for one root ("prefers" -> "pref",
    "preference" -> "prefer"), so long stems also match by prefix. Min length
    4 keeps short fragments ("use", "lik") from gluing unrelated relations."""
    for a in stems_a:
        for b in stems_b:
            if a == b:
                return True
            if len(a) >= 4 and len(b) >= 4 and (a.startswith(b) or b.startswith(a)):
                return True
    return False


def relations_similar(a: str, b: str) -> bool:
    """Deterministic floor: do two relation strings plausibly name the same
    attribute? True on a shared content stem or near-identical strings."""
    na, nb = _norm(a), _norm(b)
    if na == nb:
        return True
    if _stems_match(_content_stems(a), _content_stems(b)):
        return True
    return SequenceMatcher(None, na, nb).ratio() >= _RATIO_FLOOR


def find_near_duplicate(dataset: str, subject: str, relation: str) -> Fact | None:
    """The write-time guard's lookup: a currently-valid fact in `dataset`
    with the same normalized subject and a similar-but-not-equal relation.
    Most recent match wins if several. O(current facts) — datasets are small
    and this only runs on an exact-key miss."""
    key = resolver.canonical_key(subject, relation)
    subject_norm = _norm(subject)
    best: Fact | None = None
    for fact in resolver.current_facts(dataset):
        if resolver.canonical_key(fact.subject, fact.relation) == key:
            continue  # exact key is the resolver's own job
        if _norm(fact.subject) != subject_norm:
            continue
        if not relations_similar(fact.relation, relation):
            continue
        if best is None or fact.id > best.id:
            best = fact
    return best


# --- flag persistence (dupe_flags table lives in facts.db, see resolver) ---


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def flag_pair(dataset: str, fact_a: int, fact_b: int, reason: str) -> bool:
    """Record a possible-duplicate pair (idempotent). Returns True if new."""
    a, b = sorted((fact_a, fact_b))
    with resolver._connect() as conn:  # noqa: SLF001 — same-package helper
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO dupe_flags
                (dataset, fact_a, fact_b, reason, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (dataset, a, b, reason, _now()),
        )
        conn.commit()
        return cur.rowcount > 0


def resolve_flag(dataset: str, fact_a: int, fact_b: int, resolution: str) -> None:
    a, b = sorted((fact_a, fact_b))
    with resolver._connect() as conn:  # noqa: SLF001
        conn.execute(
            """
            UPDATE dupe_flags SET resolved_at = ?, resolution = ?
            WHERE dataset = ? AND fact_a = ? AND fact_b = ? AND resolved_at IS NULL
            """,
            (_now(), resolution, dataset, a, b),
        )
        conn.commit()


def unresolved_flags(dataset: str) -> list[dict]:
    """Open flags whose facts are BOTH still current. A flag referencing a
    superseded/evicted fact is auto-resolved ('stale') here — forgetting one
    side of a pair IS the user resolving it, no extra ceremony."""
    with resolver._connect() as conn:  # noqa: SLF001
        rows = conn.execute(
            "SELECT * FROM dupe_flags WHERE dataset = ? AND resolved_at IS NULL",
            (dataset,),
        ).fetchall()
        out: list[dict] = []
        for row in rows:
            pair_rows = conn.execute(
                """
                SELECT id, subject, relation, object FROM facts
                WHERE id IN (?, ?) AND valid_to IS NULL AND evicted_at IS NULL
                """,
                (row["fact_a"], row["fact_b"]),
            ).fetchall()
            if len(pair_rows) < 2:
                conn.execute(
                    "UPDATE dupe_flags SET resolved_at = ?, resolution = 'stale' WHERE id = ?",
                    (_now(), row["id"]),
                )
                continue
            out.append(
                {
                    "fact_a": row["fact_a"],
                    "fact_b": row["fact_b"],
                    "reason": row["reason"],
                    "created_at": row["created_at"],
                    "facts": [
                        {
                            "id": p["id"],
                            "subject": p["subject"],
                            "relation": p["relation"],
                            "object": p["object"],
                        }
                        for p in pair_rows
                    ],
                }
            )
        conn.commit()
    return out


# --- batch pass ---


async def sweep(dataset: str, use_judge: bool = False) -> dict:
    """One dedupe pass over a dataset's current facts.

    Flag-only by default (the no-LLM floor). With `use_judge=True` and the
    optional judge importable, each pair is resolved: 'update'/'same' close
    the OLDER row (supersede-not-overwrite; history keeps it) and remove it
    from the search index; 'distinct' dismisses the flag so it never
    re-surfaces. Judge unavailable -> falls back to flagging, reported
    honestly in `mode`.
    """
    facts = resolver.current_facts(dataset)
    by_subject: dict[str, list[Fact]] = {}
    for f in facts:
        by_subject.setdefault(_norm(f.subject), []).append(f)

    pairs: list[tuple[Fact, Fact]] = []
    for group in by_subject.values():
        group.sort(key=lambda f: f.id)
        for i, older in enumerate(group):
            for newer in group[i + 1 :]:
                same_key = resolver.canonical_key(
                    older.subject, older.relation
                ) == resolver.canonical_key(newer.subject, newer.relation)
                if not same_key and relations_similar(older.relation, newer.relation):
                    pairs.append((older, newer))

    judge = None
    if use_judge:
        try:
            from core import router

            judge = router.judge_conflict
        except Exception:  # noqa: BLE001 — LLM stack not installed
            judge = None

    report = {
        "dataset": dataset,
        "facts_scanned": len(facts),
        "pairs_found": len(pairs),
        "flagged": 0,
        "resolved": [],
        "mode": "judge" if judge else "flag-only",
    }

    for older, newer in pairs:
        verdict = None
        if judge is not None:
            try:
                verdict = await judge(older.text, newer.text)
            except Exception:  # noqa: BLE001 — judge down mid-sweep
                verdict = None
        if verdict in ("update", "same"):
            _close_older(dataset, older, newer)
            resolve_flag(dataset, older.id, newer.id, verdict)
            report["resolved"].append(
                {"older": older.id, "newer": newer.id, "verdict": verdict}
            )
        elif verdict == "distinct":
            flag_pair(dataset, older.id, newer.id, "near-duplicate relation")
            resolve_flag(dataset, older.id, newer.id, "distinct")
            report["resolved"].append(
                {"older": older.id, "newer": newer.id, "verdict": "distinct"}
            )
        else:
            if flag_pair(
                dataset,
                older.id,
                newer.id,
                f"relations look alike: {older.relation!r} vs {newer.relation!r}",
            ):
                report["flagged"] += 1

    return report


def _close_older(dataset: str, older: Fact, newer: Fact) -> None:
    """Supersede `older` by `newer` (bi-temporal close, never a delete) and
    drop it from the search index — the write-time invariant, kept here too."""
    now = _now()
    with resolver._connect() as conn:  # noqa: SLF001
        conn.execute(
            "UPDATE facts SET valid_to = ?, superseded_by = ? WHERE id = ? AND dataset = ?",
            (now, newer.id, older.id, dataset),
        )
        conn.commit()
    try:
        from core import index

        index.remove_item(dataset, "fact", older.id)
    except sqlite3.Error:  # index cleanup is best-effort; FTS resync heals it
        pass
