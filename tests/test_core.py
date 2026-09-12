from pathlib import Path

from app.runner import detect_project


def test_detect_python_project(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='x'\n"
    )

    language, command = detect_project(tmp_path)

    assert language == "python"
    assert command == "python -m unittest discover -v"


def test_detect_python_pytest_project(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "x"

[project.optional-dependencies]
test = ["pytest"]
"""
    )

    (tmp_path / "pytest.ini").write_text(
        "[pytest]\ntestpaths = tests\n"
    )

    language, command = detect_project(tmp_path)

    assert language == "python"
    assert command == "pytest -q"


def test_detect_node_project(tmp_path: Path):
    (tmp_path / "package.json").write_text("{}")

    language, command = detect_project(tmp_path)

    assert language == "node"
    assert command == "npm test"


def test_detect_node_jest_project(tmp_path: Path):
    (tmp_path / "package.json").write_text(
        """
{
  "scripts": {
    "test": "jest"
  },
  "devDependencies": {
    "jest": "^30.0.0"
  }
}
"""
    )

    language, command = detect_project(tmp_path)

    assert language == "node"
    assert command == "npm test"


def test_detect_go_project(tmp_path: Path):
    (tmp_path / "go.mod").write_text(
        "module example.com/test\n\ngo 1.25\n"
    )

    language, command = detect_project(tmp_path)

    assert language == "go"
    assert command == "go test ./..."


def test_detect_rust_project(tmp_path: Path):
    (tmp_path / "Cargo.toml").write_text(
        """
[package]
name = "example"
version = "0.1.0"
edition = "2024"
"""
    )

    language, command = detect_project(tmp_path)

    assert language == "rust"
    assert command == "cargo test"


def test_detect_java_maven_project(tmp_path: Path):
    (tmp_path / "pom.xml").write_text(
        """
<project>
    <modelVersion>4.0.0</modelVersion>
</project>
"""
    )

    language, command = detect_project(tmp_path)

    assert language == "java"
    assert command == "mvn test -q"


def test_detect_cpp_cmake_project(tmp_path: Path):
    (tmp_path / "CMakeLists.txt").write_text(
        """
cmake_minimum_required(VERSION 3.20)
project(example)
"""
    )

    language, command = detect_project(tmp_path)

    assert language == "cpp"
    assert command == "ctest --output-on-failure"