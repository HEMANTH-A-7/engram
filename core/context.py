"""memory_context: a budgeted, deduplicated, current-only digest of a
project's memory — the payload injected into a Claude Code session.

This is the pivot's headline artifact. Instead of replaying raw history
(every version of every fact, every superseded note — the no-memory-layer
baseline), a session gets ONE compact digest:

- **Current-only by construction**: items come from `resolver.current_facts`
  and `notes.current_notes`, whose `valid_to IS NULL` filter excludes
  superseded and evicted rows for free. Nothing stale can leak in.
- **Deduplicated**: facts are already one-current-per-canonical-key;
  notes/facts are additionally deduped on normalized text.
- **Budgeted**: items are admitted newest/most-relevant-first until the
  token budget is spent. Token counts are the project's documented
  ~4 chars/token *estimate* (see `live_metrics._estimate_tokens`).
- **Ranked**: with a query, relevance comes from the index's hybrid search
  (FTS5 keyword always; fused with semantic when the embed endpoint answers
  — see core/index.py) blended with recency + access frequency (the same
  signals the consolidation tiering uses). The ranking mode actually used
  (`hybrid` / `keyword` / `recency`) is reported, never silently swapped.

The return payload carries the digest AND its own measurement (digest tokens
vs raw-history tokens) so every retrieval doubles as the before/after metric.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from core import index, notes, resolver
from core.live_metrics import _estimate_tokens

DEFAULT_BUDGET = 2000
# One runaway item (e.g. a pre-compaction extract) must not eat the whole
# budget; anything longer is truncated with an ellipsis marker.
MAX_ITEM_TOKENS = 200

RECENCY_HALF_LIFE_DAYS = 14.0  # same intuition as consolidation tiering

_SECTIONS = (  # render order: durable knowledge first
    ("decision", "Decisions"),
    ("gotcha", "Gotchas"),
    ("fact", "Facts"),
    ("progress", "Progress"),
    ("reference", "References"),
)


@dataclass(frozen=True)
class _Item:
    source: str  # 'fact' | 'note'
    id: int
    section: str  # one of _SECTIONS keys
    text: str
    created_at: str
    access_count: int


def _recency(created_or_accessed: str, now: datetime) -> float:
    try:
        then = datetime.fromisoformat(created_or_accessed)
    except ValueError:
        return 0.0
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    age_days = max((now - then).total_seconds() / 86400, 0.0)
    return math.exp(-math.log(2) / RECENCY_HALF_LIFE_DAYS * age_days)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _truncate(text: str, max_tokens: int) -> str:
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 2].rstrip() + " …"


def _gather(dataset: str) -> list[_Item]:
    items: list[_Item] = []
    for f in resolver.current_facts(dataset):
        items.append(
            _Item(
                source="fact",
                id=f.id,
                section="fact",
                text=f.text,
                created_at=f.last_accessed or f.ingestion_time,
                access_count=f.access_count,
            )
        )
    for n in notes.current_notes(dataset):
        items.append(
            _Item(
                source="note",
                id=n.id,
                section=n.kind,
                text=n.text,
                created_at=n.last_accessed or n.created_at,
                access_count=n.access_count,
            )
        )
    return items


def build_context(
    query: str | None = None,
    token_budget: int = DEFAULT_BUDGET,
    dataset: str = "main_dataset",
    max_item_tokens: int = MAX_ITEM_TOKENS,
) -> dict:
    """Build the digest. Returns the digest string plus honest measurement
    (all token figures are ~4 chars/token estimates)."""
    token_budget = max(100, int(token_budget))
    now = datetime.now(timezone.utc)
    items = _gather(dataset)

    # Dedupe on normalized text (facts already unique per canonical key).
    seen: set[str] = set()
    deduped: list[_Item] = []
    for item in items:
        key = _norm(item.text)
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    # Rank. With a query, relevance comes from the index's hybrid search
    # (never raises: worst case keyword-only). Scores are normalized to the
    # top hit since BM25/RRF/cosine live on different scales; the mode used
    # is reported, not silently swapped.
    ranking = "recency"
    rel: dict[tuple[str, int], float] = {}
    if query:
        result = index.search(dataset, query, k=len(deduped) or 1)
        top = max((h.score for h in result.hits), default=0.0)
        if top > 0:
            rel = {(h.kind, h.ref_id): h.score / top for h in result.hits}
        ranking = result.mode  # 'hybrid' | 'keyword'

    max_access = max((i.access_count for i in deduped), default=0)

    def score(item: _Item) -> float:
        recency = _recency(item.created_at, now)
        freq = (
            math.log1p(item.access_count) / math.log1p(max_access)
            if max_access > 0
            else 0.0
        )
        if rel:
            return 0.6 * rel.get((item.source, item.id), 0.0) + 0.25 * recency + 0.15 * freq
        return 0.7 * recency + 0.3 * freq

    ranked = sorted(deduped, key=score, reverse=True)

    # Admit items newest/most-relevant-first until the budget is spent.
    header = f"## Project memory — {dataset} (as of {now.date().isoformat()})\n"
    spent = _estimate_tokens(header)
    included: list[tuple[_Item, str]] = []
    sections_used: set[str] = set()
    for item in ranked:
        line = f"- [{item.created_at[:10]}] {_truncate(item.text, max_item_tokens)}\n"
        cost = _estimate_tokens(line)
        if item.section not in sections_used:
            cost += _estimate_tokens(f"\n### {item.section}\n")
        if spent + cost > token_budget:
            continue  # a shorter later item may still fit
        spent += cost
        sections_used.add(item.section)
        included.append((item, line))

    # Render grouped by section, rank order within each.
    parts = [header]
    for section_key, title in _SECTIONS:
        lines = [line for item, line in included if item.section == section_key]
        if lines:
            parts.append(f"\n### {title}\n")
            parts.extend(lines)
    digest = "".join(parts) if included else ""

    # Measurement: digest vs the no-memory-layer baseline (every fact
    # version + every note row ever written).
    raw_tokens = sum(_estimate_tokens(f.text) for f in resolver.all_versions(dataset))
    raw_tokens += sum(_estimate_tokens(n.text) for n in notes.all_notes(dataset))
    digest_tokens = _estimate_tokens(digest)
    return {
        "digest": digest,
        "dataset": dataset,
        "ranking": ranking,
        "included": len(included),
        "excluded": len(deduped) - len(included),
        "current_items": len(deduped),
        "token_budget": token_budget,
        "digest_tokens": digest_tokens,
        "raw_history_tokens": raw_tokens,
        "pct_of_raw": round(digest_tokens / raw_tokens, 4) if raw_tokens else 0.0,
        "note": (
            "Token figures are ~4 chars/token estimates (offline-safe). "
            "raw_history_tokens = every fact version + note ever written "
            "(the no-memory-layer baseline); the digest is current-only, "
            "deduplicated, and budget-capped."
        ),
    }
