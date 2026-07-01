"""Integration test for the Bucket 2 benchmark harness.

Runs the real pipeline against Ollama (offline), so it's slow. Marked
`integration` — run explicitly with:

    uv run pytest -m integration -q

Asserts recall@5 clears a sane floor and latency numbers are well-formed,
without pinning exact values (extraction is nondeterministic — see
PROGRESS.md's LLM_TEMPERATURE note).
"""

from __future__ import annotations

import pytest

from benchmark import harness

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_baseline_harness_runs_and_recalls():
    result = await harness.run(label="test_baseline")

    assert result["n_cases"] > 0
    assert result["recall"]["recall@5"] >= 0.75

    latency = result["latency"]
    assert latency["p50_ms"] > 0
    assert latency["p95_ms"] >= latency["p50_ms"]
