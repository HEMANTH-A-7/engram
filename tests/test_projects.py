"""Unit tests for core/projects.py — per-project dataset keying."""

from __future__ import annotations

from core import projects


def test_no_project_falls_back_to_main_dataset(monkeypatch):
    monkeypatch.delenv("MEMORY_PROJECT_DIR", raising=False)
    assert projects.dataset_for(None) == "main_dataset"


def test_explicit_dir_beats_env(monkeypatch, tmp_path):
    a = tmp_path / "proj-a"
    b = tmp_path / "proj-b"
    a.mkdir(), b.mkdir()
    monkeypatch.setenv("MEMORY_PROJECT_DIR", str(b))

    assert projects.dataset_for(str(a)) == projects.dataset_for(a)
    assert projects.dataset_for(str(a)) != projects.dataset_for(None)


def test_env_var_used_when_no_explicit_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_PROJECT_DIR", str(tmp_path))
    assert projects.dataset_for(None) == projects.dataset_for(tmp_path)


def test_same_basename_different_paths_do_not_collide(tmp_path):
    a = tmp_path / "one" / "api"
    b = tmp_path / "two" / "api"
    a.mkdir(parents=True), b.mkdir(parents=True)

    assert projects.dataset_for(a) != projects.dataset_for(b)
    # Both stay readable: slug survives, hash disambiguates.
    assert projects.dataset_for(a).startswith("proj_api_")
    assert projects.dataset_for(b).startswith("proj_api_")


def test_dataset_name_is_stable_and_sanitized(tmp_path):
    weird = tmp_path / "My Cool Project!!"
    weird.mkdir()
    name = projects.dataset_for(weird)

    assert name == projects.dataset_for(weird)  # stable across calls
    assert name.startswith("proj_my-cool-project_")
    assert " " not in name and "!" not in name
