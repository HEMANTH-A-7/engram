"""Consolidation / forgetting policy: tiered storage over the resolver's facts.

Scores every current fact on recency + frequency + importance, buckets it
into hot/warm/cold, and evicts anything that scores far enough into cold.
Built on top of `core/resolver.py` (owns the SQLite schema and the
`access_count`/`tier`/`evicted_at` columns this module reads and writes)
and `core/store.py` (Cognee's index, rebuilt from only the hot tier).

Two honest simplifications vs. a "real" implementation, worth stating up
front rather than letting them surprise a reader later:

- **"Compression" here means index exclusion, not summarization.** Warm and
  cold facts keep their full text in `resolver.db` — nothing is deleted
  until eviction — but are excluded from Cognee's active search index.
  True compression would need another LLM pass over the text; out of scope
  for this bucket given the project's offline/cost-aware design.
- **"Background job" here means an explicitly-invoked function, not a
  daemon.** This project has no persistent service until the MCP server
  (Bucket 6) / dashboard (Bucket 7). `run_consolidation_pass()` is the
  payload a real scheduler would call periodically; for now callers
  (harness, tests, demo scripts) invoke it directly.

Regret (a fact evicted, then later needed again) is measured via
`resolver.write()`'s `revived_from_eviction` flag, not by trying to infer
it from search-ranking noise — see `core/resolver.py`'s `evict()`/`write()`
for how that's detected deterministically from the resolver's own history.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from cognee.modules.search.types import SearchType

from core import resolver, store

# Scoring weights (sum to 1.0) and thresholds. Tunable; defaults favor
# recency, matching the intuition that a memory system should prioritize
# "what's been relevant lately" over raw hit count or a static importance.
WEIGHT_RECENCY = 0.5
WEIGHT_FREQUENCY = 0.3
WEIGHT_IMPORTANCE = 0.2

RECENCY_HALF_LIFE_DAYS = 14.0

HOT_THRESHOLD = 0.55
WARM_THRESHOLD = 0.25
# Stricter than WARM_THRESHOLD (a subset of "cold"). Must exceed
# WEIGHT_IMPORTANCE * 0.5 (the score floor a fully-decayed, never-accessed,
# default-importance fact settles at as recency/frequency -> 0) or nothing
# with the default importance would ever be evictable.
EVICT_THRESHOLD = 0.15

TIERS = ("hot", "warm", "cold")


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def _recency_score(fact: resolver.Fact, now: datetime) -> float:
    reference = fact.last_accessed or fact.ingestion_time
    age_days = max((now - _parse(reference)).total_seconds() / 86400, 0.0)
    return math.exp(-math.log(2) / RECENCY_HALF_LIFE_DAYS * age_days)


def _frequency_score(fact: resolver.Fact, max_access_count: int) -> float:
    if max_access_count <= 0:
        return 0.0
    return math.log1p(fact.access_count) / math.log1p(max_access_count)


def score_fact(fact: resolver.Fact, now: datetime, max_access_count: int) -> float:
    """Combined 0-1 score: weighted recency + frequency + importance."""
    recency = _recency_score(fact, now)
    frequency = _frequency_score(fact, max_access_count)
    return (
        WEIGHT_RECENCY * recency
        + WEIGHT_FREQUENCY * frequency
        + WEIGHT_IMPORTANCE * fact.importance
    )


def assign_tier(score: float) -> str:
    """Map a 0-1 score to a tier. Eviction is a stricter check on top of 'cold'."""
    if score >= HOT_THRESHOLD:
        return "hot"
    if score >= WARM_THRESHOLD:
        return "warm"
    return "cold"


@dataclass(frozen=True)
class ConsolidationReport:
    dataset: str
    tier_counts: dict[str, int]
    evicted_ids: tuple[int, ...] = field(default_factory=tuple)

    @property
    def evicted_count(self) -> int:
        return len(self.evicted_ids)


def run_consolidation_pass(dataset: str = "main_dataset", now: datetime | None = None) -> ConsolidationReport:
    """Rescore every current fact, reassign its tier, evict what scores under EVICT_THRESHOLD.

    Stateless and idempotent: every pass recomputes tier from scratch off
    `access_count`/`last_accessed`/`importance` rather than tracking "how
    long has this been cold" separately, so running it twice in a row with
    no activity in between produces the same result.
    """
    now = now or datetime.now(timezone.utc)
    facts = resolver.current_facts(dataset)
    max_access_count = max((f.access_count for f in facts), default=0)

    tier_counts = {t: 0 for t in TIERS}
    evicted_ids: list[int] = []

    for fact in facts:
        score = score_fact(fact, now, max_access_count)
        tier = assign_tier(score)
        if tier == "cold" and score < EVICT_THRESHOLD:
            resolver.evict(fact.id, dataset=dataset, now=now.isoformat())
            evicted_ids.append(fact.id)
            tier_counts["cold"] += 1
        else:
            resolver.set_tier(fact.id, tier, dataset=dataset)
            tier_counts[tier] += 1

    return ConsolidationReport(
        dataset=dataset, tier_counts=tier_counts, evicted_ids=tuple(evicted_ids)
    )


async def resync_active(dataset: str = "main_dataset") -> int:
    """Rebuild Cognee's index from only the hot-tier facts.

    Distinct from `core/ingest.py`'s `resync()` (all current facts,
    Bucket 3 behavior) -- this is the consolidation-aware version that
    excludes warm/cold facts from the active search index while leaving
    their text intact in `resolver.db`.
    """
    texts = [f.text for f in resolver.current_facts(dataset, tiers={"hot"})]
    return await store.reset_and_load(dataset, texts)


def _hit_text(hit) -> str:
    if isinstance(hit, dict):
        return str(hit.get("text") or hit.get("name") or hit)
    return str(getattr(hit, "text", None) or hit)


def record_access_from_hits(dataset: str, hit_texts: list[str]) -> int:
    """Match search hit text back to resolver facts by exact text, bump access stats.

    Relies on `resync_active`/`ingest.resync` indexing exactly `Fact.text`
    (1:1), so an exact-text match reliably identifies which fact a hit
    came from without needing Cognee to carry resolver IDs through its
    own pipeline.
    """
    by_text = {f.text: f for f in resolver.current_facts(dataset)}
    recorded = 0
    seen_ids: set[int] = set()
    for text in hit_texts:
        fact = by_text.get(text)
        if fact is not None and fact.id not in seen_ids:
            resolver.record_access(fact.id, dataset=dataset)
            seen_ids.add(fact.id)
            recorded += 1
    return recorded


async def search(query: str, dataset: str = "main_dataset", k: int = 5):
    """Search + record access, so retrieval feeds tiering for free.

    Thin convenience wrapper mirroring `core/ingest.py`'s `remember()`
    pattern (extract -> write -> resync in one call) -- here it's
    search -> record access in one call, so callers don't have to
    remember to call `record_access_from_hits` manually.
    """
    hits = await store.search(query, k=k, query_type=SearchType.CHUNKS, dataset=dataset)
    record_access_from_hits(dataset, [_hit_text(h) for h in hits])
    return hits
