from pathlib import Path

from app.runner import (
    _copy_build_context,
    _dockerfile_for,
    detect_project,
    detect_project_profile,
    validate_patch_paths,
)


def test_detect_python_pytest(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    assert detect_project(tmp_path) == ("python", "pytest -q")


def test_detect_project_inside_workspace_directory(tmp_path: Path):
    root = tmp_path / "workspace" / "session" / "attempt-1"
    root.mkdir(parents=True)
    (root / "calculator.py").write_text("def add(a, b):\n    return a + b\n")
    (root / "test_calculator.py").write_text("from calculator import add\n")
    assert detect_project(root) == ("python", "pytest -q")


def test_detect_node_with_lockfile(tmp_path: Path):
    (tmp_path / "package.json").write_text('{"scripts":{"test":"jest"}}')
    (tmp_path / "package-lock.json").write_text("{}")
    language, command = detect_project(tmp_path)
    assert language == "node"
    assert command == "npm test"


def test_detect_java(tmp_path: Path):
    (tmp_path / "pom.xml").write_text("<project></project>")
    assert detect_project(tmp_path) == ("java", "mvn test -q")


def test_gradle_sandbox_uses_system_gradle_without_wrapper(tmp_path: Path):
    (tmp_path / "build.gradle").write_text("plugins { id 'java' }")
    profile = detect_project_profile(tmp_path)
    assert profile.test_command == "gradle test"
    assert profile.docker_image == "gradle:8.10-jdk21"
    assert "gradle dependencies --no-daemon" in _dockerfile_for(profile)


def test_gradle_sandbox_uses_wrapper_when_present(tmp_path: Path):
    (tmp_path / "build.gradle").write_text("plugins { id 'java' }")
    (tmp_path / "gradlew").write_text("#!/bin/sh\n")
    profile = detect_project_profile(tmp_path)
    assert profile.test_command == "./gradlew test --no-daemon"
    assert "./gradlew dependencies --no-daemon" in _dockerfile_for(profile)


def test_detect_go(tmp_path: Path):
    (tmp_path / "go.mod").write_text("module example.com/x\ngo 1.25\n")
    assert detect_project(tmp_path) == ("go", "go test ./...")


def test_detect_rust(tmp_path: Path):
    (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\nversion='0.1.0'\n")
    assert detect_project(tmp_path) == ("rust", "cargo test")


def test_detect_cmake(tmp_path: Path):
    (tmp_path / "CMakeLists.txt").write_text("project(x)")
    assert detect_project(tmp_path) == ("cpp", "ctest --test-dir build --output-on-failure")


def test_patch_path_security():
    validate_patch_paths("--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-a\n+b\n")

    import pytest
    with pytest.raises(RuntimeError):
        validate_patch_paths("--- a/../secret\n+++ b/../secret\n@@ -1 +1 @@\n-a\n+b\n")


def test_build_context_excludes_secrets(tmp_path: Path):
    (tmp_path / "app.py").write_text("print('ok')")
    (tmp_path / ".env").write_text("SECRET=should-not-copy")
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "secret.txt").write_text("bad")
    destination = tmp_path / "out"
    _copy_build_context(tmp_path, destination)

    assert (destination / "app.py").exists()
    assert not (destination / ".env").exists()
    assert not (destination / "workspace").exists()


def test_python_dockerfile_contains_offline_verification_command(tmp_path: Path):
    (tmp_path / "requirements.txt").write_text("pytest\n")
    profile = detect_project_profile(tmp_path)
    dockerfile = _dockerfile_for(profile)
    assert "requirements.txt" in dockerfile
    assert "pytest -q" in dockerfile


def test_docker_test_reports_missing_docker(monkeypatch, tmp_path):
    from app import runner

    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    monkeypatch.setattr(runner, "_docker_available", lambda: False)
    result = runner.docker_test(tmp_path, "python", "pytest -q")
    assert result.passed is False
    assert "Docker is required" in result.output


def test_plain_python_dockerfile_does_not_assume_package_metadata(tmp_path: Path):
    (tmp_path / "calculator.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "test_calculator.py").write_text("def test_add(): assert True\n")
    profile = detect_project_profile(tmp_path)
    dockerfile = _dockerfile_for(profile)
    assert "RUN true" in dockerfile
    assert "pip install --no-cache-dir -e ." not in dockerfile
    assert "pytest -q" in dockerfile


def test_build_context_excludes_private_keys_and_symlinks(tmp_path: Path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "app.py").write_text("print('ok')")
    (source / "private.key").write_text("secret")
    target = tmp_path / "outside.txt"
    target.write_text("outside")
    try:
        (source / "escape.txt").symlink_to(target)
    except OSError:
        return
    _copy_build_context(source, destination)
    assert (destination / "app.py").exists()
    assert not (destination / "private.key").exists()
    assert not (destination / "escape.txt").exists()
