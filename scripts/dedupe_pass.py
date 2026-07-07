#!/usr/bin/env python3
"""Batch near-duplicate pass over one dataset's current facts.

Flag-only by default (the no-LLM floor — pairs land in dupe_flags and show
up in memory_stats.possible_duplicates; resolve one by forgetting a side).
`--resolve` asks the optional LLM judge to settle each pair instead:
'update'/'same' supersede the older row (kept in history), 'distinct'
dismisses the pair for good. Judge unavailable -> falls back to flagging,
and the report says so (mode: flag-only).

Usage:
    uv run python scripts/dedupe_pass.py                        # main_dataset, flag-only
    uv run python scripts/dedupe_pass.py --dataset proj_x_y     # another dataset
    uv run python scripts/dedupe_pass.py --resolve              # judge-resolved (needs Ollama)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import dedupe  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="main_dataset")
    parser.add_argument(
        "--resolve",
        action="store_true",
        help="resolve pairs with the LLM judge instead of only flagging",
    )
    args = parser.parse_args()

    report = asyncio.run(dedupe.sweep(args.dataset, use_judge=args.resolve))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
