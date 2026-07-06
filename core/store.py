"""Thin async wrapper around Cognee's baseline pipeline (add -> cognify -> search).

This is the *control group*: vanilla Cognee, no conflict resolution, no
consolidation. Every later component benchmarks against this. Storage is pinned
into the repo (`.cognee_data` / `.cognee_system`) and everything runs offline
against Ollama via the shared `.env` config.
"""

from __future__ import annotations

from pathlib import Path

import cognee
from cognee.api.v1.cognify.cognify import extract_graph_and_summarize, get_default_tasks
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.cognify.rollback import cognify_rollback_handler
from cognee.modules.pipelines import run_pipeline
from cognee.modules.pipelines.layers.pipeline_execution_mode import get_pipeline_executor
from cognee.modules.search.types import SearchType
from cognee.shared.data_models import KnowledgeGraph, Node as _CogneeNode
from pydantic import BaseModel, Field, field_validator

from core.config import REPO_ROOT

DATA_DIR = REPO_ROOT / ".cognee_data"
SYSTEM_DIR = REPO_ROOT / ".cognee_system"

_configured = False


class LenientSummary(BaseModel):
    """Drop-in for Cognee's ``SummarizedContent`` with an *optional* ``summary``.

    Cognee 1.2 fuses graph extraction and chunk summarization into a single
    ``cognify()`` task run under one ``asyncio.gather`` (see
    ``tasks/graph/extract_graph_and_summarize.py``). Cognee's stock
    ``SummarizedContent`` makes ``summary`` a *required* field, so whenever the
    local Ollama model returns summary JSON missing that field -- a recurring
    flake on small quantized models -- the pydantic ``missing`` validation error
    propagates up through both gather layers and aborts the **entire** cognify
    batch, not just the summary. That is the single largest source of the
    "cognify keeps failing / needs 3 attempts" slowness in this project.

    We search with ``SearchType.CHUNKS`` (pure retrieval over raw chunk text)
    and never read these summaries, so defaulting ``summary`` to ``""`` is
    safe: a flaky response validates as an empty summary instead of exploding,
    and ``instructor`` stops burning its (hardcoded) retries on a shape that
    retrying rarely fixes. The field *description* is kept verbatim so the
    happy-path prompt to the model is unchanged.
    """

    summary: str = Field(
        default="",
        description=(
            "One leading sentence stating what the input is about, "
            "followed by a bulleted list of self-contained facts."
        ),
    )
    description: str = Field(default="", description="Unused; kept for backwards compatibility.")


class _LenientNode(_CogneeNode):
    """Cognee ``Node`` with every string field optional and ``None`` coerced to ``""``.

    Same failure shape as the summary flake: Cognee's stock ``Node`` makes
    ``description`` (and ``label``) *required* strings, but the local model
    routinely emits ``"description": null`` -- which fails validation and forces
    ``instructor`` into its (hardcoded) retries, adding latency and log noise,
    and risking a whole-batch abort if the retries are exhausted. Defaulting the
    fields and coercing ``None -> ""`` lets the first response validate. We
    retrieve with ``SearchType.CHUNKS`` (raw-chunk vectors), so the knowledge
    graph's node text is never queried and empty descriptions are harmless here.
    """

    description: str = ""
    label: str = ""

    @field_validator("id", "name", "type", "description", "label", mode="before")
    @classmethod
    def _none_to_empty(cls, value):
        return "" if value is None else value


class LenientKnowledgeGraph(KnowledgeGraph):
    """``KnowledgeGraph`` subclass using :class:`_LenientNode`.

    Subclassing (rather than a parallel model) is deliberate: Cognee gates its
    graph post-processing on ``issubclass(graph_model, KnowledgeGraph)``
    (``tasks/graph/extract_graph_from_data.py``), so a subclass stays on the
    same well-trodden code path while swapping in lenient nodes.
    """

    summary: str = ""
    description: str = ""
    nodes: list[_LenientNode] = Field(default_factory=list)


def configure(data_dir: Path | None = None, system_dir: Path | None = None) -> None:
    """Pin Cognee's storage into the repo and harden summarization. Idempotent."""
    global _configured
    data_dir = data_dir or DATA_DIR
    system_dir = system_dir or SYSTEM_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    system_dir.mkdir(parents=True, exist_ok=True)
    # Both roots hold private memory text -- owner-only, not umask-default.
    data_dir.chmod(0o700)
    system_dir.chmod(0o700)
    cognee.config.data_root_directory(str(data_dir))
    cognee.config.system_root_directory(str(system_dir))
    # Swap Cognee's required-field summarization model for the lenient one so a
    # flaky local-model summary can't abort the whole cognify batch. The config
    # is an lru_cache'd singleton and summarize_text reads it per call, so this
    # one assignment covers every cognify path (demo, tests, MCP server).
    get_cognify_config().summarization_model = LenientSummary
    _configured = True


def _ensure_configured() -> None:
    if not _configured:
        configure()


async def add(text: str, dataset: str = "main_dataset") -> None:
    """Ingest a raw text fact into a dataset (not yet cognified)."""
    _ensure_configured()
    await cognee.add(text, dataset_name=dataset)


async def cognify(dataset: str = "main_dataset") -> None:
    """Run extraction: build the knowledge graph + embeddings for a dataset.

    Uses :class:`LenientKnowledgeGraph` so a local-model node with a ``null``
    description validates on the first pass instead of forcing instructor
    retries (or aborting the batch). Combined with the lenient summarization
    model installed in ``configure()``, both required-field-vs-null flakes on
    the cognify path are neutralized.
    """
    _ensure_configured()
    await cognee.cognify(datasets=[dataset], graph_model=LenientKnowledgeGraph)


def _embed_only_tasks(tasks: list):
    """Drop the slow ``extract_graph_and_summarize`` task from a default task list.

    Everything else (classify -> chunk -> ``add_data_points``) is kept, so raw
    ``DocumentChunk``s are still embedded and persisted for CHUNKS retrieval.
    Split out from :func:`cognify_embed_only` so the filter is unit-testable
    without paying the LLM pipeline.
    """
    return [t for t in tasks if t.executable is not extract_graph_and_summarize]


async def cognify_embed_only(dataset: str = "main_dataset") -> None:
    """Cognify for CHUNKS retrieval only, skipping the slow graph+summarize task.

    Cognee's public ``cognify()`` runs a fixed 5-task pipeline; exactly one task,
    ``extract_graph_and_summarize``, is expensive (two gemma4 LLM passes, minutes
    per batch on CPU Ollama) -- and every retrieval path in this layer uses
    ``SearchType.CHUNKS``, which never touches the graph or summaries that task
    produces. So it is pure dead weight for writes: dropping it is the single
    biggest write-latency win (proven spike: ~1.3s vs. minutes).

    ``cognee.cognify()`` exposes no ``tasks=`` parameter, so we rebuild the
    default task list, filter out that one task, and drive Cognee's *internal*
    ``run_pipeline`` ourselves via the same executor ``cognify()`` uses. This is
    deliberate deeper coupling into Cognee internals (``get_default_tasks`` /
    ``run_pipeline`` / ``get_pipeline_executor``) in exchange for the speedup;
    the remaining tasks (classify -> chunk -> ``add_data_points``) still embed
    and persist the raw ``DocumentChunk``s, which is what CHUNKS search returns.
    ``cognify()`` is kept for the GRAPH_COMPLETION baseline demo.
    """
    _ensure_configured()
    tasks = await get_default_tasks(graph_model=LenientKnowledgeGraph)
    kept = _embed_only_tasks(tasks)
    executor = get_pipeline_executor(run_in_background=False)
    await executor(
        pipeline=run_pipeline,
        tasks=kept,
        user=None,
        datasets=[dataset],
        vector_db_config=None,
        graph_db_config=None,
        incremental_loading=True,
        use_pipeline_cache=False,
        pipeline_name="cognify_pipeline",
        data_per_batch=20,
        rollback_handler=cognify_rollback_handler,
    )


async def search(
    query: str,
    k: int = 5,
    query_type: SearchType = SearchType.CHUNKS,
    dataset: str = "main_dataset",
):
    """Query the memory. Defaults to CHUNKS (ranked retrieval hits, no LLM)."""
    _ensure_configured()
    return await cognee.search(
        query_text=query,
        query_type=query_type,
        top_k=k,
        datasets=[dataset],
    )


async def reset() -> None:
    """Wipe all data + system state. Used by tests and demos for a clean slate."""
    _ensure_configured()
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


async def reset_and_load(
    dataset: str, texts: list[str], max_attempts: int = 3, graph: bool = False
) -> int:
    """Reset, add every text, and cognify, retrying the whole batch on flakiness.

    Cognee's Ollama adapter hardcodes 2 structured-output retries and batches
    every fact into one `cognify()` run, so a single flaky extraction (e.g.
    gemma4 emitting a null field) aborts the whole batch. Retrying the full
    reset+add+cognify cycle absorbs that. Returns the number of attempts taken
    (1 = succeeded first try).

    ``graph`` defaults to ``False``: the memory-layer write/resync paths only
    ever retrieve with ``SearchType.CHUNKS``, so they take the fast
    :func:`cognify_embed_only` path (no graph+summarize LLM passes). Callers that
    need the full knowledge graph (e.g. the GRAPH_COMPLETION baseline demo) pass
    ``graph=True`` to run the complete :func:`cognify` pipeline.
    """
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        await reset()
        for text in texts:
            await add(text, dataset=dataset)
        try:
            if graph:
                await cognify(dataset=dataset)
            else:
                await cognify_embed_only(dataset=dataset)
            return attempt
        except Exception as exc:  # noqa: BLE001 - LLM structured-output flakiness
            last_error = exc
            print(f"  cognify attempt {attempt}/{max_attempts} failed: {exc}")
    raise RuntimeError(f"cognify failed after {max_attempts} attempts") from last_error
