from pathlib import Path

from app.runner import detect_project


def test_detect_python_project(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    assert detect_project(tmp_path) == ("python", "python -m unittest discover -v")


def test_detect_python_pytest_project(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "pytest.ini").write_text("[pytest]\ntestpaths = tests\n")
    assert detect_project(tmp_path) == ("python", "python -m pytest -q -p no:cacheprovider")


def test_detect_node_project(tmp_path: Path):
    (tmp_path / "package.json").write_text("{}")
    assert detect_project(tmp_path) == ("node", "npm test")


def test_detect_node_locked_project(tmp_path: Path):
    (tmp_path / "package.json").write_text('{"scripts":{"test":"jest"}}')
    (tmp_path / "package-lock.json").write_text('{}')
    assert detect_project(tmp_path) == ("node", "npm test")


def test_detect_go_project(tmp_path: Path):
    (tmp_path / "go.mod").write_text("module example.com/test\n\ngo 1.25\n")
    assert detect_project(tmp_path) == ("go", "go test ./...")


def test_detect_rust_project(tmp_path: Path):
    (tmp_path / "Cargo.toml").write_text('[package]\nname="example"\nversion="0.1.0"\n')
    assert detect_project(tmp_path) == ("rust", "cargo test")


def test_detect_java_maven_project(tmp_path: Path):
    (tmp_path / "pom.xml").write_text("<project><modelVersion>4.0.0</modelVersion></project>")
    assert detect_project(tmp_path) == ("java", "mvn test -q")


def test_detect_java_gradle_project(tmp_path: Path):
    (tmp_path / "build.gradle").write_text("plugins { id 'java' }")
    assert detect_project(tmp_path) == ("java", "gradle test")


def test_detect_java_gradle_wrapper_project(tmp_path: Path):
    (tmp_path / "build.gradle.kts").write_text("plugins { java }")
    (tmp_path / "gradlew").write_text("#!/bin/sh\n")
    assert detect_project(tmp_path) == ("java", "./gradlew test --no-daemon")


def test_detect_cpp_cmake_project(tmp_path: Path):
    (tmp_path / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.20)\nproject(example)\n")
    assert detect_project(tmp_path) == ("cpp", "ctest --test-dir build --output-on-failure")
