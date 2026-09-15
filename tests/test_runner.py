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
    assert detect_project(tmp_path) == ("python", "python -m pytest -q -p no:cacheprovider")


def test_detect_project_inside_workspace_directory(tmp_path: Path):
    root = tmp_path / "workspace" / "session" / "attempt-1"
    root.mkdir(parents=True)
    (root / "calculator.py").write_text("def add(a, b):\n    return a + b\n")
    (root / "test_calculator.py").write_text("from calculator import add\n")
    assert detect_project(root) == ("python", "python -m pytest -q -p no:cacheprovider")


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


def test_plain_python_project_dockerfile_does_not_require_editable_install(tmp_path: Path):
    (tmp_path / "calculator.py").write_text("def add(a, b): return a + b\n")
    (tmp_path / "test_calculator.py").write_text("def test_add(): assert True\n")
    profile = detect_project_profile(tmp_path)
    dockerfile = _dockerfile_for(profile)
    assert "RUN true" in dockerfile
    assert "pip install --no-cache-dir -e ." not in dockerfile


def test_python_dockerfile_contains_verification_command(tmp_path: Path):
    (tmp_path / "requirements.txt").write_text("pytest\n")
    profile = detect_project_profile(tmp_path)
    dockerfile = _dockerfile_for(profile)
    assert "requirements.txt" in dockerfile
    assert "pytest -q" in dockerfile


def test_build_context_rejects_symlink_escape(tmp_path):
    from app.runner import _copy_build_context
    import pytest

    source = tmp_path / "source"
    destination = tmp_path / "out"
    source.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("do not copy")
    link = source / "escape.txt"

    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is unavailable in this environment")

    with pytest.raises(RuntimeError):
        _copy_build_context(source, destination)


def test_docker_test_reports_missing_docker(monkeypatch, tmp_path):
    from app import runner

    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    monkeypatch.setattr(runner, "_docker_available", lambda: False)
    result = runner.docker_test(tmp_path, "python", "pytest -q")
    assert result.passed is False
    assert "Docker is required" in result.output


def test_dockerfile_runs_code_from_isolated_writable_runtime_copy(tmp_path: Path):
    (tmp_path / "calculator.py").write_text("def add(a, b): return a + b\\n")
    (tmp_path / "test_calculator.py").write_text("def test_add(): assert True\\n")
    profile = detect_project_profile(tmp_path)
    dockerfile = _dockerfile_for(profile)
    assert "WORKDIR /opt/healforge-repo" in dockerfile
    assert "cp -a /opt/healforge-repo/. /work/" in dockerfile
    assert "cd /work" in dockerfile


def test_recovery_restarts_from_clean_state_without_attempt1_modifications(tmp_path: Path):
    from app.runner import prepare_attempt_workspace, apply_patch

    session_dir = tmp_path / "session_1"
    session_dir.mkdir()
    base_repo = session_dir / "base_repo"
    base_repo.mkdir()
    (base_repo / ".git").mkdir()
    (base_repo / "calc.py").write_text("def op(): return 'ORIGINAL'\n")

    # Attempt 1: starts from clean state, then patch X -> Y is applied
    a1_root = prepare_attempt_workspace("http://fake.git", "sha1", session_dir, 1)
    assert (a1_root / "calc.py").read_text() == "def op(): return 'ORIGINAL'\n"

    patch1_file = session_dir / "repair-1.patch"
    patch1_file.write_text(
        "--- a/calc.py\n+++ b/calc.py\n@@ -1 +1 @@\n-def op(): return 'ORIGINAL'\n+def op(): return 'ATTEMPT_1'\n"
    )
    applied1, _ = apply_patch(a1_root, patch1_file)
    assert applied1 is True
    assert (a1_root / "calc.py").read_text() == "def op(): return 'ATTEMPT_1'\n"

    # Attempt 2: MUST start from clean ORIGINAL state, NOT contaminated with ATTEMPT_1!
    a2_root = prepare_attempt_workspace("http://fake.git", "sha1", session_dir, 2)
    assert (a2_root / "calc.py").read_text() == "def op(): return 'ORIGINAL'\n"
    assert "ATTEMPT_1" not in (a2_root / "calc.py").read_text()

    # Apply patch X -> Z on attempt 2
    patch2_file = session_dir / "repair-2.patch"
    patch2_file.write_text(
        "--- a/calc.py\n+++ b/calc.py\n@@ -1 +1 @@\n-def op(): return 'ORIGINAL'\n+def op(): return 'ATTEMPT_2'\n"
    )
    applied2, _ = apply_patch(a2_root, patch2_file)
    assert applied2 is True
    assert (a2_root / "calc.py").read_text() == "def op(): return 'ATTEMPT_2'\n"

    # Base snapshot MUST remain untouched
    assert (base_repo / "calc.py").read_text() == "def op(): return 'ORIGINAL'\n"


def test_git_clone_called_at_most_once_per_session(tmp_path: Path):
    from app.runner import prepare_attempt_workspace

    session_dir = tmp_path / "session_2"
    clone_count = 0

    def mock_checkout(url, sha, dest, pr_number=None):
        nonlocal clone_count
        clone_count += 1
        dest.mkdir(parents=True, exist_ok=True)
        (dest / ".git").mkdir()
        (dest / "file.txt").write_text("pristine")

    # Run 3 recovery attempts
    for attempt in range(1, 4):
        root = prepare_attempt_workspace(
            "http://fake.git",
            "sha123",
            session_dir,
            attempt,
            checkout_fn=mock_checkout,
        )
        assert (root / "file.txt").read_text() == "pristine"

    # Git network clone occurred exactly ONCE!
    assert clone_count == 1


def test_dependency_fingerprint_ignores_pure_code_changes(tmp_path: Path):
    from app.runner import _dependency_fingerprint, detect_project_profile

    (tmp_path / "requirements.txt").write_text("pytest==8.0.0\n")
    (tmp_path / "main.py").write_text("x = 1\n")
    profile = detect_project_profile(tmp_path)

    fp1 = _dependency_fingerprint(tmp_path, profile)

    # Pure source code edit
    (tmp_path / "main.py").write_text("x = 9999\n")
    fp2 = _dependency_fingerprint(tmp_path, profile)

    # Base image fingerprint is invariant under source code changes
    assert fp1 == fp2

    # Manifest change changes fingerprint
    (tmp_path / "requirements.txt").write_text("pytest==8.1.0\n")
    fp3 = _dependency_fingerprint(tmp_path, profile)
    assert fp3 != fp1


def test_classify_failure_categories():
    from app.runner import RunResult, classify_failure

    # Success
    assert classify_failure(RunResult(True, "cmd", 0, "OK")) == "SUCCESS"

    # Category A: Patch failure
    assert classify_failure(RunResult(False, "pytest", 1, "FAILED test_calc.py::test_add - AssertionError")) == "PATCH_FAILURE"

    # Category B: Environment failure
    assert classify_failure(RunResult(False, "pytest", 1, "ModuleNotFoundError: No module named 'numpy'")) == "ENVIRONMENT_FAILURE"
    assert classify_failure(RunResult(False, "pytest", 1, "ImportError: cannot import name 'c_internal'")) == "ENVIRONMENT_FAILURE"
    assert classify_failure(RunResult(False, "pytest", 1, "ENVIRONMENT NOT READY: Pre-flight environment validation failed.")) == "ENVIRONMENT_FAILURE"

    # Category C: Timeout
    assert classify_failure(RunResult(False, "pytest", 124, "Process timed out after 180 seconds.")) == "TIMEOUT"

    # Category D: Compilation / Syntax failure
    assert classify_failure(RunResult(False, "pytest", 1, "SyntaxError: invalid syntax")) == "COMPILATION_FAILURE"
    assert classify_failure(RunResult(False, "make", 2, "fatal error: ft2build.h: No such file or directory")) == "COMPILATION_FAILURE"

    # Category E: Infrastructure failure
    assert classify_failure(RunResult(False, "pytest", 127, "Cannot connect to the Docker daemon")) == "INFRASTRUCTURE_FAILURE"


def test_python_setup_dockerfile_generates_c_and_editable_install(tmp_path: Path):
    from app.runner import detect_project_profile, _dockerfile_for

    (tmp_path / "setup.py").write_text("from setuptools import setup; setup(name='pkg')\n")
    (tmp_path / "pyproject.toml").write_text("[build-system]\nrequires=['setuptools']\n")
    profile = detect_project_profile(tmp_path)
    dockerfile = _dockerfile_for(profile, tmp_path)

    # Must have generic build tools — no matplotlib-specific packages
    assert "build-essential" in dockerfile
    assert "setuptools" in dockerfile
    assert "pip install" in dockerfile
    # Must NOT have any matplotlib-specific packages
    assert "numpy<2" not in dockerfile
    assert "contourpy" not in dockerfile
    assert "pybind11" not in dockerfile
    assert "libfreetype" not in dockerfile

