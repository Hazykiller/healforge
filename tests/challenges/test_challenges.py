from __future__ import annotations

from pathlib import Path
import pytest

from app.ai import AIEngine
from app.runner import detect_project, detect_project_profile, _dockerfile_for
from tests.challenges.definitions import CHALLENGES, Challenge


@pytest.mark.parametrize("challenge", CHALLENGES, ids=lambda c: c.id)
def test_challenge_detection(tmp_path: Path, challenge: Challenge):
    """Verify that detect_project correctly identifies the language and test command."""
    for rel_path, content in challenge.files.items():
        full_path = tmp_path / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")

    profile = detect_project_profile(tmp_path)
    assert profile.language == challenge.expected_language
    assert profile.test_command == challenge.expected_test_command

    lang, cmd = detect_project(tmp_path)
    assert lang == challenge.expected_language
    assert cmd == challenge.expected_test_command


@pytest.mark.parametrize("challenge", CHALLENGES, ids=lambda c: c.id)
def test_challenge_dockerfile_generation(tmp_path: Path, challenge: Challenge):
    """Verify that _dockerfile_for produces a valid Dockerfile for each challenge."""
    for rel_path, content in challenge.files.items():
        full_path = tmp_path / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")

    profile = detect_project_profile(tmp_path)
    dockerfile = _dockerfile_for(profile)

    assert f"FROM {profile.docker_image}" in dockerfile
    assert "WORKDIR /opt/healforge-repo" in dockerfile
    assert "COPY" in dockerfile
    assert challenge.expected_test_command in dockerfile


@pytest.mark.parametrize("challenge", CHALLENGES, ids=lambda c: c.id)
def test_challenge_sample_edit_generates_diff(challenge: Challenge):
    """Verify that AIEngine._apply_edits produces valid patch and touches correct file."""
    engine = object.__new__(AIEngine)
    patch, touched = engine._apply_edits(
        [challenge.sample_edit],
        dict(challenge.files),
    )

    expected_file = challenge.sample_edit["file"]
    assert expected_file in touched
    assert f"--- a/{expected_file}" in patch
    assert f"+++ b/{expected_file}" in patch
    assert "+" in patch
