"""Orchestration layer: raw fact text -> resolver -> Cognee's search index.

Extracts a single `(subject, relation, object)` triple per fact (a much
narrower schema than Cognee's own multi-node `KnowledgeGraph` extraction, and
so a smaller failure surface for the same class of gemma4 structured-output
flakiness hit in Bucket 2), hands it to `core/resolver.py` for bi-temporal
versioning, then rebuilds Cognee's index from `resolver.current_facts()` so a
superseded fact's text can never remain searchable.

Every `remember()` call does a full `store.reset_and_load()` rather than an
incremental add. That's intentionally not efficient — it guarantees
correctness (no stale text ever lingers in the index) for the single-dataset
scope this bucket targets. `store.reset()` itself wipes Cognee's whole
configured data root, not just one dataset, so — like the Bucket 1/2
baseline — callers should treat one dataset as active at a time.
"""

from __future__ import annotations

from dataclasses import dataclass

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, model_validator

from core import resolver, store
from core.config import LLM, bare_model

MAX_EXTRACT_ATTEMPTS = 3

_TRIPLE_SYSTEM_PROMPT = (
    "Extract the single (subject, relation, object) triple this fact expresses. "
    "Use a concise canonical entity name for `subject` (e.g. 'Alice', not 'my "
    "friend Alice'). `relation` is a short snake_case predicate (e.g. "
    "'lives_in', 'favorite_color'). `object` is the value, kept as concise as "
    "possible. If this fact updates a subject/relation stated elsewhere, "
    "extract only the new value here — do not merge it with any old value."
)

_client: AsyncOpenAI | None = None
_aclient = None


class Triple(BaseModel):
    subject: str = Field(..., description="Canonical entity name the fact is about.")
    relation: str = Field(..., description="Short snake_case predicate.")
    object: str = Field(..., description="The value of the relation.")

    @model_validator(mode="before")
    @classmethod
    def _unwrap_schema_echo(cls, data):
        """Recover from gemma4 echoing the injected JSON-schema shape.

        instructor's json_mode prompt embeds this schema (as text) so the
        model can match its format — but gemma4 sometimes pattern-matches
        the shown schema's `properties` wrapper too literally and nests the
        actual values under it (`{"properties": {"subject": "Alice", ...}}`)
        instead of returning them at the top level. When that happens with
        plain string values (not another round of schema-shaped dicts), the
        real triple is still fully present — just misplaced — so unwrap it
        rather than burning a retry on a fixable shape.
        """
        if isinstance(data, dict) and "subject" not in data and isinstance(
            data.get("properties"), dict
        ):
            props = data["properties"]
            if all(isinstance(props.get(k), str) for k in ("subject", "relation", "object")):
                return props
        return data


def _ensure_client() -> None:
    global _client, _aclient
    if _aclient is not None:
        return
    _client = AsyncOpenAI(base_url=LLM.openai_base(), api_key=LLM.api_key)
    _aclient = instructor.from_openai(_client, mode=instructor.Mode("json_mode"))


async def _extract_triple(text: str) -> Triple:
    """Extract a (subject, relation, object) triple, retrying on flakiness.

    Bounded at MAX_EXTRACT_ATTEMPTS full attempts, mirroring the harness's
    cognify-retry discipline (Bucket 2) — a narrower schema than Cognee's own
    KnowledgeGraph extraction, but still an LLM call, so still retried rather
    than trusted blind.
    """
    _ensure_client()
    last_error: Exception | None = None
    for attempt in range(1, MAX_EXTRACT_ATTEMPTS + 1):
        try:
            return await _aclient.chat.completions.create(
                model=bare_model(LLM.model),
                messages=[
                    {"role": "system", "content": _TRIPLE_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                max_retries=2,
                response_model=Triple,
            )
        except Exception as exc:  # noqa: BLE001 - LLM structured-output flakiness
            last_error = exc
            print(f"  triple extraction attempt {attempt}/{MAX_EXTRACT_ATTEMPTS} failed: {exc}")
    raise RuntimeError(
        f"triple extraction failed after {MAX_EXTRACT_ATTEMPTS} attempts"
    ) from last_error


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
    triple = await _extract_triple(text)
    write_result = resolver.write(
        triple.subject,
        triple.relation,
        triple.object,
        text,
        dataset=dataset,
        event_time=event_time,
    )
    cognify_attempts = await resync(dataset) if sync else 0
    return IngestResult(triple=triple, write=write_result, cognify_attempts=cognify_attempts)
