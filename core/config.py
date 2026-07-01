"""Central configuration + LLM/embedding provider seam.

We reuse Cognee's canonical env-var names (LLM_PROVIDER, LLM_MODEL, LLM_ENDPOINT,
LLM_API_KEY, EMBEDDING_*) so Cognee and our own code read ONE config. Defaults are
an offline Ollama setup, so the repo runs with zero cloud credentials. The provider
indirection is the seam we extend later so the same binary can borrow a host model
(Claude Code / Codex / Antigravity) without code changes.

Models are stored litellm-style (e.g. "ollama/llama3.2:3b"). `bare_model()` strips
the provider prefix and `openai_base()` yields the OpenAI-compatible URL for direct
HTTP calls (Ollama serves one at `<endpoint>/v1`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Minimal .env loader (avoids a hard dependency for one small file)."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        # allow inline comments after the value
        value = value.split("#", 1)[0].strip()
        os.environ.setdefault(key.strip(), value)


_load_dotenv()


def bare_model(model: str) -> str:
    """Strip a litellm provider prefix: 'ollama/llama3.2:3b' -> 'llama3.2:3b'."""
    return model.split("/", 1)[1] if "/" in model else model


def _openai_base(endpoint: str) -> str:
    """Reduce any Ollama endpoint to its OpenAI-compatible base (`<host>/v1`)."""
    base = endpoint.rstrip("/")
    for suffix in ("/api/embeddings", "/api/embed", "/api", "/v1"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return f"{base.rstrip('/')}/v1"


@dataclass(frozen=True)
class LLMConfig:
    provider: str = os.environ.get("LLM_PROVIDER", "ollama")
    endpoint: str = os.environ.get("LLM_ENDPOINT", "http://localhost:11434")
    api_key: str = os.environ.get("LLM_API_KEY", "ollama")
    # Cognee's default extraction model + our router's two paths (Bucket 5).
    model: str = os.environ.get("LLM_MODEL", "ollama/llama3.2:3b")
    model_small: str = os.environ.get("MEMORY_MODEL_SMALL", "ollama/llama3.2:3b")
    model_large: str = os.environ.get("MEMORY_MODEL_LARGE", "ollama/gemma4:latest")

    def openai_base(self) -> str:
        """OpenAI-compatible base URL for our own direct HTTP calls.

        Cognee needs provider-specific suffixes (LLM `/v1`, embedding `/api/embed`),
        so strip any known suffix back to the host root and append `/v1`.
        """
        return _openai_base(self.endpoint)


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = os.environ.get("EMBEDDING_PROVIDER", "ollama")
    endpoint: str = os.environ.get("EMBEDDING_ENDPOINT", "http://localhost:11434")
    model: str = os.environ.get("EMBEDDING_MODEL", "ollama/nomic-embed-text")
    dimensions: int = int(os.environ.get("EMBEDDING_DIMENSIONS", "768"))

    def openai_base(self) -> str:
        return _openai_base(self.endpoint)


LLM = LLMConfig()
EMBEDDING = EmbeddingConfig()
