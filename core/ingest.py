"""Orchestration layer: raw fact text -> resolver -> Cognee's search index.

Extracts a single `(subject, relation, object)` triple per fact via
`core/router.py`'s cost-aware cascade (Bucket 5: cheap model first, escalate
to the reliable model only on failure), hands it to `core/resolver.py` for
bi-temporal versioning, then rebuilds Cognee's index from
`resolver.current_facts()` so a superseded fact's text can never remain
searchable.

POST-PIVOT NOTE (Session 9): the MCP serving path calls
`remember(sync=False)` and maintains `core/index.py` incrementally instead —
`resync()`/`sync=True` (full `store.reset_and_load()`, which wipes Cognee's
whole configured data root) survives ONLY for the Cognee-based benchmark and
demo paths, where one dataset at a time is still the rule.
"""

from __future__ import annotations

from dataclasses import dataclass

from core import resolver, router, store
from core.router import Triple


@dataclass(frozen=True)
class IngestResult:
    triple: Triple
    write: resolver.WriteResult
    cognify_attempts: int


async def resync(dataset: str = "main_dataset") -> int:
    """Rebuild Cognee's index from `resolver.current_facts(dataset)`."""
    texts = [f.text for f in resolver.current_facts(dataset)]
    return await store.reset_and_load(dataset, texts)


async def remember(
    text: str,
    dataset: str = "main_dataset",
    event_time: str | None = None,
    sync: bool = True,
) -> IngestResult:
    """Extract a fact, version it bi-temporally, and resync Cognee's index.

    `sync=False` skips the resync (extraction + resolver write only) — for
    callers batching many writes before a single `resync()`, since each
    resync is a full reset+cognify pass. The default stays `True` so a
    caller writing one fact at a time always gets an up-to-date index.
    """
    triple = await router.route_extract_triple(text)
    write_result = await resolver.write(
        triple.subject,
        triple.relation,
        triple.object,
        text,
        dataset=dataset,
        event_time=event_time,
    )
    cognify_attempts = await resync(dataset) if sync else 0
    return IngestResult(triple=triple, write=write_result, cognify_attempts=cognify_attempts)
