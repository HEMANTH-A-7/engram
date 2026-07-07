#!/usr/bin/env python3
"""PreCompact hook: persist a session extract BEFORE compaction squeezes it.

This is the pivot's core safety property: when the context window exhausts,
whatever mattered is already on disk. Compaction then proceeds; the
follow-up SessionStart (source "compact") re-injects the digest, so the
continuation picks up where the full window left off.

The saved note is a deterministic EXTRACT (recent user requests + last
assistant state), not an LLM summary — chosen for speed and zero flake risk
inside a hook, and labeled honestly as such. Keyed per session, so repeated
compactions in one session update one note instead of accumulating.
Fail-open: any error exits 0; memory must never block compaction.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

from _common import project_dir, read_hook_input, session_extract, transcript_messages


def main() -> None:
    payload = read_hook_input()
    from core import projects
    from core.notes import add_note

    messages = transcript_messages(payload.get("transcript_path"))
    extract = session_extract(messages)
    if not extract:
        return
    session_id = payload.get("session_id", "unknown")
    trigger = payload.get("trigger", "auto")
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    text = (
        f"[pre-compaction extract | session {session_id} | {trigger} | {stamp}]\n"
        f"{extract}"
    )
    add_note(
        text,
        kind="progress",
        dataset=projects.dataset_for(project_dir(payload)),
        key=f"compact:{session_id}",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 — fail-open by design (see docstring)
        sys.exit(0)
