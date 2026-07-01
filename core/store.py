"""Thin async wrapper around Cognee's baseline pipeline (add -> cognify -> search).

This is the *control group*: vanilla Cognee, no conflict resolution, no
consolidation. Every later component benchmarks against this. Storage is pinned
into the repo (`.cognee_data` / `.cognee_system`) and everything runs offline
against Ollama via the shared `.env` config.
"""

from __future__ import annotations

from pathlib import Path

import cognee
from cognee.modules.search.types import SearchType

from core.config import REPO_ROOT

DATA_DIR = REPO_ROOT / ".cognee_data"
SYSTEM_DIR = REPO_ROOT / ".cognee_system"

_configured = False


def configure(data_dir: Path | None = None, system_dir: Path | None = None) -> None:
    """Pin Cognee's storage into the repo. Idempotent."""
    global _configured
    data_dir = data_dir or DATA_DIR
    system_dir = system_dir or SYSTEM_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    system_dir.mkdir(parents=True, exist_ok=True)
    cognee.config.data_root_directory(str(data_dir))
    cognee.config.system_root_directory(str(system_dir))
    _configured = True


def _ensure_configured() -> None:
    if not _configured:
        configure()


async def add(text: str, dataset: str = "main_dataset") -> None:
    """Ingest a raw text fact into a dataset (not yet cognified)."""
    _ensure_configured()
    await cognee.add(text, dataset_name=dataset)


async def cognify(dataset: str = "main_dataset") -> None:
    """Run extraction: build the knowledge graph + embeddings for a dataset."""
    _ensure_configured()
    await cognee.cognify(datasets=[dataset])


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
