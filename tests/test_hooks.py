"""Tests for the Claude Code hook scripts (scripts/hooks/) — real subprocesses.

Each test runs the actual script the way Claude Code would: JSON payload on
stdin, MEMORY_DB_PATH pointing the whole stack at a tmp SQLite file. No
Ollama is required — notes land with pending embeddings (the honest
degradation path), and the SessionStart digest ranks by recency when no
query is involved. Fail-open behavior (exit 0, no output, DB untouched on
garbage input) is part of the contract: a memory bug must never block a
session, a compaction, or a turn.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from core import notes, projects, resolver

HOOKS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "hooks"


def _run_hook(script: str, payload: dict, db_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(HOOKS_DIR / script)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": "/usr/bin:/bin", "MEMORY_DB_PATH": str(db_path)},
    )


def _write_transcript(path: Path, messages: list[tuple[str, str]]) -> None:
    lines = [
        json.dumps(
            {"type": role, "message": {"role": role, "content": [{"type": "text", "text": text}]}}
        )
        for role, text in messages
    ]
    path.write_text("\n".join(lines))


@pytest.fixture()
def env(tmp_path):
    db = tmp_path / "facts.db"
    resolver.configure(db_path=db)  # test process reads the same file the hooks write
    project = tmp_path / "someproject"
    project.mkdir()
    return db, project, projects.dataset_for(project)


def test_stop_hook_upserts_one_rolling_session_note(env, tmp_path):
    db, project, dataset = env
    transcript = tmp_path / "t.jsonl"
    payload = {"cwd": str(project), "session_id": "s1", "transcript_path": str(transcript)}

    _write_transcript(transcript, [("user", "fix the login bug"), ("assistant", "Fixed: the token check was inverted.")])
    proc = _run_hook("stop.py", payload, db)
    assert proc.returncode == 0

    _write_transcript(transcript, [("user", "fix the login bug"), ("assistant", "Also added a regression test.")])
    proc = _run_hook("stop.py", payload, db)
    assert proc.returncode == 0

    current = notes.current_notes(dataset)
    assert len(current) == 1  # rolling upsert, not accumulation
    assert "regression test" in current[0].text
    assert "session s1" in current[0].text
    assert len(notes.all_notes(dataset)) == 2  # superseded revision kept


def test_stop_hook_reentry_guard_writes_nothing(env, tmp_path):
    db, project, dataset = env
    transcript = tmp_path / "t.jsonl"
    _write_transcript(transcript, [("assistant", "something")])
    proc = _run_hook(
        "stop.py",
        {"cwd": str(project), "session_id": "s1", "transcript_path": str(transcript), "stop_hook_active": True},
        db,
    )
    assert proc.returncode == 0
    assert notes.current_notes(dataset) == []


def test_pre_compact_saves_labeled_session_extract(env, tmp_path):
    db, project, dataset = env
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            ("user", "pivot the project to a dev-context layer"),
            ("assistant", "Built memory_note and the incremental index."),
            ("user", "now add per-project datasets"),
            ("assistant", "Done: datasets keyed by project dir; single-active-dataset constraint removed."),
        ],
    )
    proc = _run_hook(
        "pre_compact.py",
        {"cwd": str(project), "session_id": "s9", "transcript_path": str(transcript), "trigger": "auto"},
        db,
    )
    assert proc.returncode == 0

    current = notes.current_notes(dataset)
    assert len(current) == 1
    text = current[0].text
    assert "[pre-compaction extract" in text  # labeled as an extract, honestly
    assert "per-project datasets" in text  # recent user asks captured
    assert "single-active-dataset constraint removed" in text  # last assistant state


def test_session_start_injects_digest_of_saved_memory(env, tmp_path):
    db, project, dataset = env
    notes.add_note("Chose flock for cross-process locking.", kind="decision", dataset=dataset)

    proc = _run_hook("session_start.py", {"cwd": str(project), "session_id": "s2"}, db)

    assert proc.returncode == 0
    out = json.loads(proc.stdout)
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert out["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "flock" in ctx
    assert "Decisions" in ctx


def test_session_start_empty_memory_prints_nothing(env):
    db, project, _ = env
    proc = _run_hook("session_start.py", {"cwd": str(project)}, db)
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


def test_hooks_fail_open_on_garbage_input(env):
    db, _, dataset = env
    for script in ("session_start.py", "pre_compact.py", "stop.py"):
        proc = subprocess.run(
            [sys.executable, str(HOOKS_DIR / script)],
            input="not json at all {{{",
            capture_output=True,
            text=True,
            timeout=60,
            env={"PATH": "/usr/bin:/bin", "MEMORY_DB_PATH": str(db)},
        )
        assert proc.returncode == 0, script
        assert proc.stdout.strip() == "", script
    assert notes.current_notes(dataset) == []
