"""Cost-aware LLM router: cascade extraction, conflict judgment, summarization.

Bucket 5. Three call sites, one shared ledger:

- `route_extract_triple` -- the "routine fact extraction" path. Empirical
  testing this bucket (10 sample facts, real Ollama, production's actual
  retry policy) found `llama3.2:3b` succeeded 5/10 (avg 4.3s/call) while
  `gemma4:latest` succeeded 7/10 but averaged 38.4s/call (~9x slower) --
  gemma4 is *more reliable and much slower*, not simply "better". Neither
  model is reliable enough alone even on this narrow single-triple schema,
  so this is a **cascade**, not a static small/large split: try the cheap
  model first, escalate to the reliable-but-slow model only on failure.
  Most facts never need to escalate, so most extraction stays cheap.
- `judge_conflict` -- the "ambiguous conflict resolution" path. Always the
  large model, no cascade: ambiguity judgment isn't something worth
  retry-cheap on, and it's called rarely (only when `core/resolver.py`'s
  substring-overlap heuristic flags a conflict as possibly-not-clean).
- `summarize_fact` -- the "consolidation summarization" path. Always the
  large model, called once per fact the first time it drops to the cold
  tier (see `core/consolidation.py`).

Every call, success or failure, is logged to `_ledger` as a `CallRecord` --
a failed cheap-model attempt still cost tokens, so `cost_report()` counts
it rather than only counting what eventually succeeded.

**Cost is reference pricing, not real spend.** This project runs 100%
locally against Ollama; actual spend is $0. `_PRICE_PER_M_TOKENS` prices
real prompt/completion token counts against published 2026 cloud rates for
comparably-sized open models, so "$ per 1,000 memories" means "what this
would cost against a hosted model of this size" -- not an invented number,
but also not a real bill. Sources: Llama 3.2 3B ~$0.10/M
(https://pricepertoken.com/pricing-page/model/meta-llama-llama-3.2-3b),
Llama 3.1 8B-class ~$0.20/M (Together AI, via
https://www.aipricing.guru/together-pricing/).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, model_validator

from core.config import LLM, bare_model

MAX_EXTRACT_ATTEMPTS = 3  # outer attempts on model_large, after the 2-attempt small-model phase

_PRICE_PER_M_TOKENS = {"small": 0.10, "large": 0.20}

_TRIPLE_SYSTEM_PROMPT = (
    "Extract the single (subject, relation, object) triple this fact expresses. "
    "Use a concise canonical entity name for `subject` (e.g. 'Alice', not 'my "
    "friend Alice'). `relation` is a short snake_case predicate (e.g. "
    "'lives_in', 'favorite_color'). `object` is the value, kept as concise as "
    "possible. If this fact updates a subject/relation stated elsewhere, "
    "extract only the new value here -- do not merge it with any old value."
)

_CONFLICT_SYSTEM_PROMPT = (
    "Two fact statements share the same subject and relation but give "
    "different values. Decide the relationship between them: 'update' if the "
    "new statement is a genuine change or correction (the old value is no "
    "longer true), 'same' if the new statement restates the same underlying "
    "value in different words (not a real change), or 'distinct' if both "
    "values can be true at the same time (they don't actually conflict)."
)

_SUMMARIZE_SYSTEM_PROMPT = (
    "Summarize the following fact in one concise sentence, preserving its "
    "key subject, relation, and value. Output only the summary sentence, "
    "nothing else."
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
        """Recover from gemma4/llama3.2 echoing the injected JSON-schema shape.

        instructor's json_mode prompt embeds this schema (as text) so the
        model can match its format -- but these models sometimes pattern-match
        the shown schema's `properties` wrapper too literally and nest the
        actual values under it (`{"properties": {"subject": "Alice", ...}}`)
        instead of returning them at the top level. When that happens with
        plain string values (not another round of schema-shaped dicts), the
        real triple is still fully present -- just misplaced -- so unwrap it
        rather than burning a retry on a fixable shape.
        """
        if isinstance(data, dict) and "subject" not in data and isinstance(
            data.get("properties"), dict
        ):
            props = data["properties"]
            if all(isinstance(props.get(k), str) for k in ("subject", "relation", "object")):
                return props
        return data


class _ConflictVerdict(BaseModel):
    verdict: Literal["update", "same", "distinct"]


@dataclass(frozen=True)
class CallRecord:
    task: Literal["extract", "conflict_judge", "summarize"]
    model_tier: Literal["small", "large"]
    prompt_tokens: int
    completion_tokens: int
    attempt: int
    succeeded: bool
    latency_s: float


_ledger: list[CallRecord] = []


def reset_ledger() -> None:
    _ledger.clear()


def _record(**kwargs) -> None:
    _ledger.append(CallRecord(**kwargs))


def _ensure_client() -> None:
    global _client, _aclient
    if _aclient is not None:
        return
    _client = AsyncOpenAI(base_url=LLM.openai_base(), api_key=LLM.api_key)
    _aclient = instructor.from_openai(_client, mode=instructor.Mode("json_mode"))


def _usage_tokens(completion) -> tuple[int, int]:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return 0, 0
    return getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0


async def route_extract_triple(text: str) -> Triple:
    """Cascade extraction: cheap model first, escalate to the reliable model on failure.

    2 outer attempts on `model_small` (max_retries=1 each -- the small
    model's own failures are usually schema-shape flakes instructor's retry
    already fixes; two independent full attempts catches the rest without
    paying the slow model's latency), then up to `MAX_EXTRACT_ATTEMPTS`
    further attempts on `model_large` (max_retries=2 each, matching this
    project's pre-router retry policy) if the small model never succeeds.
    Every attempt is logged to `_ledger` regardless of outcome.
    """
    _ensure_client()
    last_error: Exception | None = None
    attempt = 0

    for _ in range(2):
        attempt += 1
        tier, model, max_retries = "small", bare_model(LLM.model_small), 1
        start = time.perf_counter()
        try:
            triple, completion = await _aclient.chat.completions.create_with_completion(
                model=model,
                messages=[
                    {"role": "system", "content": _TRIPLE_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                max_retries=max_retries,
                response_model=Triple,
            )
            prompt_tokens, completion_tokens = _usage_tokens(completion)
            _record(
                task="extract", model_tier=tier, prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens, attempt=attempt, succeeded=True,
                latency_s=time.perf_counter() - start,
            )
            return triple
        except Exception as exc:  # noqa: BLE001 - LLM structured-output flakiness
            last_error = exc
            _record(
                task="extract", model_tier=tier, prompt_tokens=0, completion_tokens=0,
                attempt=attempt, succeeded=False, latency_s=time.perf_counter() - start,
            )
            print(f"  [router] small-model extraction attempt {attempt} failed: {exc}")

    for _ in range(MAX_EXTRACT_ATTEMPTS):
        attempt += 1
        tier, model, max_retries = "large", bare_model(LLM.model_large), 2
        start = time.perf_counter()
        try:
            triple, completion = await _aclient.chat.completions.create_with_completion(
                model=model,
                messages=[
                    {"role": "system", "content": _TRIPLE_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                max_retries=max_retries,
                response_model=Triple,
            )
            prompt_tokens, completion_tokens = _usage_tokens(completion)
            _record(
                task="extract", model_tier=tier, prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens, attempt=attempt, succeeded=True,
                latency_s=time.perf_counter() - start,
            )
            return triple
        except Exception as exc:  # noqa: BLE001 - LLM structured-output flakiness
            last_error = exc
            _record(
                task="extract", model_tier=tier, prompt_tokens=0, completion_tokens=0,
                attempt=attempt, succeeded=False, latency_s=time.perf_counter() - start,
            )
            print(f"  [router] large-model extraction attempt {attempt} failed: {exc}")

    raise RuntimeError(f"triple extraction failed after {attempt} attempts") from last_error


async def judge_conflict(old_text: str, new_text: str) -> Literal["update", "same", "distinct"]:
    """Ask the large model to resolve an ambiguous conflict. Always model_large, no cascade."""
    _ensure_client()
    start = time.perf_counter()
    try:
        result, completion = await _aclient.chat.completions.create_with_completion(
            model=bare_model(LLM.model_large),
            messages=[
                {"role": "system", "content": _CONFLICT_SYSTEM_PROMPT},
                {"role": "user", "content": f"OLD: {old_text}\nNEW: {new_text}"},
            ],
            max_retries=2,
            response_model=_ConflictVerdict,
        )
        prompt_tokens, completion_tokens = _usage_tokens(completion)
        _record(
            task="conflict_judge", model_tier="large", prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, attempt=1, succeeded=True,
            latency_s=time.perf_counter() - start,
        )
        return result.verdict
    except Exception:
        _record(
            task="conflict_judge", model_tier="large", prompt_tokens=0, completion_tokens=0,
            attempt=1, succeeded=False, latency_s=time.perf_counter() - start,
        )
        raise


async def summarize_fact(text: str) -> str:
    """Ask the large model for a one-sentence summary. Always model_large, plain text output."""
    _ensure_client()
    start = time.perf_counter()
    try:
        completion = await _client.chat.completions.create(
            model=bare_model(LLM.model_large),
            messages=[
                {"role": "system", "content": _SUMMARIZE_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )
        summary = (completion.choices[0].message.content or "").strip()
        prompt_tokens, completion_tokens = _usage_tokens(completion)
        _record(
            task="summarize", model_tier="large", prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, attempt=1, succeeded=True,
            latency_s=time.perf_counter() - start,
        )
        return summary
    except Exception:
        _record(
            task="summarize", model_tier="large", prompt_tokens=0, completion_tokens=0,
            attempt=1, succeeded=False, latency_s=time.perf_counter() - start,
        )
        raise


def cost_report() -> dict:
    """Aggregate `_ledger` into per-task/per-tier stats and a reference $/1000-memories figure."""
    by_task: dict[str, dict[str, int]] = {}
    by_tier: dict[str, dict[str, int]] = {
        "small": {"calls": 0, "successes": 0, "prompt_tokens": 0, "completion_tokens": 0},
        "large": {"calls": 0, "successes": 0, "prompt_tokens": 0, "completion_tokens": 0},
    }
    total_cost = 0.0

    for r in _ledger:
        task_stats = by_task.setdefault(r.task, {"calls": 0, "successes": 0})
        task_stats["calls"] += 1
        task_stats["successes"] += int(r.succeeded)

        tier_stats = by_tier[r.model_tier]
        tier_stats["calls"] += 1
        tier_stats["successes"] += int(r.succeeded)
        tier_stats["prompt_tokens"] += r.prompt_tokens
        tier_stats["completion_tokens"] += r.completion_tokens

        total_cost += (
            (r.prompt_tokens + r.completion_tokens) / 1_000_000 * _PRICE_PER_M_TOKENS[r.model_tier]
        )

    # route_extract_triple returns on first success, so exactly one succeeded
    # "extract" record exists per distinct fact actually stored.
    successful_extracts = by_task.get("extract", {}).get("successes", 0)
    usd_per_1000 = (
        round(total_cost / successful_extracts * 1000, 4) if successful_extracts else None
    )

    return {
        "total_calls": len(_ledger),
        "by_task": by_task,
        "by_tier": by_tier,
        "total_cost_usd": round(total_cost, 6),
        "successful_extracts": successful_extracts,
        "usd_per_1000_memories": usd_per_1000,
        "pricing_note": (
            "Reference cloud pricing, not real spend -- this project runs "
            "100% locally against Ollama (actual cost: $0). $0.10/M tokens "
            "for the small (~3B) tier, $0.20/M for the large (~8-9B) tier, "
            "based on 2026 Together AI / comparable published rates for "
            "similarly-sized open models."
        ),
    }
