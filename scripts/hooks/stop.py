#!/usr/bin/env python3
"""Stop hook: persist the turn's outcome as a rolling per-session state note.

Fires when Claude finishes responding. Upserts ONE note per session (keyed
`session:<id>`), so a long session maintains a single continuously-updated
"where we are" record instead of a note per turn — the keyed-supersede path
in core/notes.py exists for exactly this. Combined with PreCompact (full
extract before the window is squeezed) and SessionStart (re-inject), nothing
important is lost between turns, compactions, or sessions.

Deterministic extract, no LLM (speed + zero flake risk in a hook).
Fail-open: any error exits 0; memory must never block a turn.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

from _common import clip, project_dir, read_hook_input, transcript_messages


def main() -> None:
    payload = read_hook_input()
    if payload.get("stop_hook_active"):
        return  # re-entry guard
    from core import projects
    from core.notes import add_note

    messages = transcript_messages(payload.get("transcript_path"))
    last_user = next((t for role, t in reversed(messages) if role == "user"), "")
    last_assistant = next((t for role, t in reversed(messages) if role == "assistant"), "")
    if not last_assistant:
        return
    session_id = payload.get("session_id", "unknown")
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    parts = [f"[session state | session {session_id} | updated {stamp}]"]
    if last_user:
        parts.append(f"Working on: {clip(last_user, 300)}")
    parts.append(f"Latest state: {clip(last_assistant, 1500)}")
    add_note(
        "\n".join(parts),
        kind="progress",
        dataset=projects.dataset_for(project_dir(payload)),
        key=f"session:{session_id}",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 — fail-open by design (see docstring)
        sys.exit(0)
