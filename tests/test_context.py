from app.context import build_context, is_sensitive_path


def test_context_contains_diagnostic_evidence():
    context = build_context(
        {
            "title": "Fix request handling",
            "body": "",
            "head": {"sha": "abc"},
            "base": {"ref": "main"},
        },
        [{
            "filename": "api.py",
            "status": "modified",
            "additions": 2,
            "deletions": 1,
            "patch": "@@ -1 +1 @@",
        }],
        [{
            "name": "pytest",
            "status": "completed",
            "conclusion": "failure",
            "output": {"summary": "AssertionError"},
        }],
        {
            "api.py": "from service import process",
            "service.py": "def process(x): return x",
            "tests/test_api.py": "def test_api(): assert True",
            "pyproject.toml": "[project]\nname='x'",
        },
        ["api.py", "service.py", "tests/test_api.py", "pyproject.toml"],
    )

    assert "AssertionError" in context
    assert "service.py" in context
    assert "pyproject.toml" in context
    assert "DIAGNOSTIC RULES" in context


def test_sensitive_files_are_excluded():
    assert is_sensitive_path(".env")
    assert is_sensitive_path("secrets/id_rsa")
    assert is_sensitive_path("cert/server.pem")
    assert not is_sensitive_path("src/app.py")
