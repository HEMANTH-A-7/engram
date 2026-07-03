"""Fast regression test for the cognify summarization-flakiness fix -- no
Ollama, no cognify.

Root cause: Cognee's stock ``SummarizedContent`` makes ``summary`` a required
field, so a local-model summary response that omits it raises a pydantic
``missing`` error which -- because graph extraction and summarization share one
``asyncio.gather`` inside a single cognify task -- aborts the entire cognify
batch. ``core.store`` swaps in a lenient model and installs it via
``configure()``. These tests pin both halves deterministically, without paying
the slow/flaky LLM path.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cognee.modules.cognify.config import get_cognify_config
from cognee.shared.data_models import KnowledgeGraph, Node, SummarizedContent

from core import store


def test_stock_model_rejects_missing_summary():
    """Guards the premise: the stock model is exactly what blows up the batch."""
    with pytest.raises(ValidationError):
        SummarizedContent.model_validate({"description": "x"})


def test_lenient_model_accepts_missing_summary():
    """The flaky shape (no ``summary``) now validates as an empty summary."""
    parsed = store.LenientSummary.model_validate({"description": "x"})
    assert parsed.summary == ""
    # And a well-formed response still round-trips unchanged.
    full = store.LenientSummary.model_validate({"summary": "s", "description": "d"})
    assert full.summary == "s"


def test_configure_installs_lenient_summarization_model():
    """configure() must point Cognee's cognify config at the lenient model, so
    every cognify path (demo, tests, MCP server) inherits the hardening."""
    store.configure()
    assert get_cognify_config().summarization_model is store.LenientSummary


def test_stock_node_rejects_null_description():
    """Guards the premise for the graph-extraction flake."""
    with pytest.raises(ValidationError):
        Node.model_validate(
            {"id": "a", "name": "a", "type": "T", "description": None, "label": "L"}
        )


def test_lenient_node_coerces_null_description_to_empty():
    """The flaky node shape (``description: null``) validates as empty."""
    node = store._LenientNode.model_validate(
        {"id": "a", "name": "a", "type": "T", "description": None, "label": None}
    )
    assert node.description == ""
    assert node.label == ""


def test_lenient_graph_is_a_knowledgegraph_subclass():
    """Cognee gates graph post-processing on issubclass(_, KnowledgeGraph), so
    the lenient graph must stay a subclass to keep that code path."""
    assert issubclass(store.LenientKnowledgeGraph, KnowledgeGraph)
    # A whole graph with a null node description now validates end-to-end.
    graph = store.LenientKnowledgeGraph.model_validate(
        {
            "summary": "s",
            "description": "d",
            "nodes": [{"id": "a", "name": "a", "type": "T", "description": None, "label": "L"}],
            "edges": [],
        }
    )
    assert graph.nodes[0].description == ""
