"""Fast wiring test for the embed-only cognify path -- no Ollama, no cognify.

Writes in this memory layer only ever retrieve with ``SearchType.CHUNKS``, so
``core.store.cognify_embed_only`` drops Cognee's slow
``extract_graph_and_summarize`` task (two gemma4 LLM passes per batch) while
keeping chunking + ``add_data_points`` (embedding + persistence). These tests
pin that task selection against Cognee's real default task list without paying
the slow/flaky LLM pipeline. Building the task list (``get_default_tasks``) only
constructs ``Task`` objects around function references -- no network, no LLM.
"""

from __future__ import annotations

from cognee.api.v1.cognify.cognify import extract_graph_and_summarize, get_default_tasks

from core import store


async def test_default_tasks_contain_both_graph_and_embed():
    """Guards the premise: Cognee's default pipeline has both the slow graph
    task and the embed task, so filtering one out is a meaningful change."""
    store.configure()
    tasks = await get_default_tasks(graph_model=store.LenientKnowledgeGraph)
    names = {t.executable.__name__ for t in tasks}
    assert "extract_graph_and_summarize" in names
    assert "add_data_points" in names
    assert "extract_chunks_from_documents" in names


async def test_embed_only_drops_graph_keeps_chunking_and_embedding():
    """The kept task list excludes the slow graph+summarize LLM task but keeps
    chunking and ``add_data_points``, so CHUNKS retrieval still has vectors."""
    store.configure()
    tasks = await get_default_tasks(graph_model=store.LenientKnowledgeGraph)
    kept = store._embed_only_tasks(tasks)
    kept_names = {t.executable.__name__ for t in kept}

    assert "extract_graph_and_summarize" not in kept_names
    assert "add_data_points" in kept_names
    assert "extract_chunks_from_documents" in kept_names
    assert "classify_documents" in kept_names
    # Exactly the one expensive task is removed; nothing else is dropped.
    assert len(kept) == len(tasks) - 1


def test_embed_only_filter_removes_only_the_graph_task():
    """Unit-level filter check with fakes -- no cognee task construction at all."""

    class _Task:
        def __init__(self, executable):
            self.executable = executable

    def _keep_a():
        pass

    def _keep_b():
        pass

    tasks = [_Task(_keep_a), _Task(extract_graph_and_summarize), _Task(_keep_b)]
    kept = store._embed_only_tasks(tasks)
    assert [t.executable for t in kept] == [_keep_a, _keep_b]
