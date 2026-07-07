#!/usr/bin/env python3
"""SessionStart hook: inject this project's memory digest into the new session.

Fires when a Claude Code session starts (including the post-compaction
continuation — `source: "compact"` — which is exactly the moment the
PreCompact-saved extract needs to come back). Prints the digest as
`additionalContext`, so the session begins already knowing the project's
decisions, gotchas, and progress with no explicit recall prompt.

Budget: MEMORY_CONTEXT_BUDGET env (tokens, ~4 chars/token estimate),
default 2000. Empty memory -> no output at all (no noise in fresh projects).
Fail-open: any error exits 0 silently; memory must never block a session.
"""

from __future__ import annotations

import json
import os
import sys

from _common import project_dir, read_hook_input  # noqa: E402


def main() -> None:
    payload = read_hook_input()
    from core import index, projects  # after _common fixed sys.path
    from core.context import build_context

    dataset = projects.dataset_for(project_dir(payload))
    index.backfill_facts(dataset)  # idempotent pre-pivot migration
    budget = int(os.environ.get("MEMORY_CONTEXT_BUDGET", "2000"))
    result = build_context(token_budget=budget, dataset=dataset)
    if not result["digest"]:
        return
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": result["digest"],
                }
            }
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 — fail-open by design (see docstring)
        sys.exit(0)
