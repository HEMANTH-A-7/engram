"""Bucket 0 smoke test: prove the offline foundation runs.

Checks, all offline:
  1. Cognee imports.
  2. Ollama returns an embedding vector (via the configured embedding model).
  3. Ollama returns an LLM completion (via the configured small model).

Run: `uv run python scripts/smoke.py`
"""

from __future__ import annotations

import sys

import httpx

# Make `core` importable when run as a script.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from core.config import EMBEDDING, LLM, bare_model  # noqa: E402


def check_cognee_import() -> str:
    import cognee  # noqa: F401

    return getattr(cognee, "__version__", "unknown")


def check_embedding() -> int:
    resp = httpx.post(
        f"{EMBEDDING.openai_base()}/embeddings",
        json={"model": bare_model(EMBEDDING.model), "input": "hello memory layer"},
        timeout=60,
    )
    resp.raise_for_status()
    vec = resp.json()["data"][0]["embedding"]
    return len(vec)


def check_llm() -> str:
    resp = httpx.post(
        f"{LLM.openai_base()}/chat/completions",
        json={
            "model": bare_model(LLM.model_small),
            "messages": [{"role": "user", "content": "Reply with exactly one word: ok"}],
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def main() -> int:
    ok = True

    try:
        version = check_cognee_import()
        print(f"[PASS] cognee import OK (version={version})")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"[FAIL] cognee import: {exc}")

    try:
        dims = check_embedding()
        print(f"[PASS] embedding OK ({EMBEDDING.model}, {dims} dims)")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"[FAIL] embedding ({EMBEDDING.model}): {exc}")

    try:
        reply = check_llm()
        print(f"[PASS] LLM completion OK ({LLM.model_small} -> {reply!r})")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"[FAIL] LLM completion ({LLM.model_small}): {exc}")

    print("\nBucket 0 smoke:", "ALL GREEN ✅ (offline)" if ok else "FAILURES ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
