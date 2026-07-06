"""Fast tests for the Bucket 7 dashboard -- no Ollama, no Cognee cognify.

Two halves:
- FastAPI endpoints via `TestClient` (the app only reads saved JSON + live
  resolver/router state; no LLM). Asserts the three routes respond and carry
  the expected shape.
- The two new harness metrics (`run_tokens_eval`, `run_storage_growth_eval`)
  against an isolated `tmp_path` resolver DB. They're LLM-free (they call
  `resolver.write` on pre-labeled triples), so they belong in the fast suite
  and their invariants are checked here.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from benchmark import harness
from core import resolver
from dashboard.app import app

client = TestClient(app)


def test_index_serves_html():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Extended Memory Layer" in resp.text


def test_api_results_returns_dict():
    resp = client.get("/api/results")
    assert resp.status_code == 200
    assert isinstance(resp.json(), dict)


def test_api_stats_has_expected_keys():
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert {"facts", "cost", "benchmarks"} <= set(body)
    assert {"total_current", "by_tier"} <= set(body["facts"])


def test_api_live_has_expected_keys():
    # Everything the charts render, all live from the dataset (no benchmark JSON).
    resp = client.get("/api/live")
    assert resp.status_code == 200
    body = resp.json()
    assert {"facts", "tokens", "storage", "revisions"} <= set(body)
    assert {"raw_history_tokens", "extended_tokens", "pct_saved"} <= set(body["tokens"])
    assert {"revisions", "evicted", "by_tier"} <= set(body["revisions"])
    assert "series" in body["storage"]


def test_dashboard_polls_live_endpoint():
    # The whole dashboard refreshes on a timer against /api/live. We can't run
    # the JS here, so assert the served app.js wires up the poll: a setInterval
    # that calls the live loader, which fetches /api/live.
    resp = client.get("/static/app.js")
    assert resp.status_code == 200
    body = resp.text
    assert "setInterval(loadLive" in body
    assert 'fetch("/api/live")' in body


def test_estimate_tokens_grows_with_length():
    assert harness._estimate_tokens("") == 0
    assert harness._estimate_tokens("a short fact") < harness._estimate_tokens(
        "a considerably longer fact with substantially more characters in it"
    )


@pytest.mark.asyncio
async def test_run_tokens_eval_saves_no_more_than_it_started_with(tmp_path):
    resolver.configure(db_path=tmp_path / "facts.db")
    result = await harness.run_tokens_eval()

    assert result["extended_tokens"] <= result["raw_history_tokens"]
    assert result["tokens_saved"] == result["raw_history_tokens"] - result["extended_tokens"]
    assert 0.0 <= result["pct_saved"] <= 1.0
    # The pre-labeled set has 3 in-place updates, so some facts are superseded:
    # the memory layer must retain strictly fewer facts than were written.
    assert result["n_current_facts"] < result["n_writes"]


@pytest.mark.asyncio
async def test_run_storage_growth_series_is_consistent(tmp_path):
    resolver.configure(db_path=tmp_path / "facts.db")
    result = await harness.run_storage_growth_eval()

    series = result["series"]
    assert len(series) == result["n_writes"]
    for i, point in enumerate(series, start=1):
        assert point["facts_ingested"] == i
        assert point["without"] == i  # raw history keeps every write
        assert point["with_"] <= point["without"]  # versioning never stores more
    # raw-history line is strictly increasing; versioned line is non-decreasing.
    assert [p["without"] for p in series] == sorted(p["without"] for p in series)
