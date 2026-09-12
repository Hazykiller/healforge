from pathlib import Path
from app.github import parse_pr_url
from app.runner import detect_project

def test_parse_pr_url():
    ref = parse_pr_url("https://github.com/octocat/Hello-World/pull/7")
    assert ref.owner == "octocat"
    assert ref.repo == "Hello-World"
    assert ref.number == 7

def test_detect_python_project(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    assert detect_project(tmp_path) == ("python", "pytest -q")

def test_detect_node_project(tmp_path: Path):
    (tmp_path / "package.json").write_text("{}")
    assert detect_project(tmp_path) == ("node", "npm test -- --runInBand")
