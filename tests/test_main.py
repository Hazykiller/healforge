from fastapi.testclient import TestClient

import app.main as main
from app.runner import RunResult


class FakeGitHub:
    def __init__(self, token, timeout=30):
        pass

    async def pull_request(self, ref):
        return {
            "title": "Fix calculator",
            "body": "Repair addition",
            "number": 1,
            "html_url": "https://github.com/test/repo/pull/1",
            "head": {"sha": "abc123"},
            "base": {"ref": "main"},
        }

    async def files(self, ref):
        return [{
            "filename": "calculator.py",
            "status": "modified",
            "additions": 1,
            "deletions": 1,
            "patch": "@@ -1 +1 @@",
        }]

    async def checks(self, ref, sha):
        return {"check_runs": []}

    async def tree(self, ref, sha):
        return [
            {"path": "calculator.py", "type": "blob"},
            {"path": "test_calculator.py", "type": "blob"},
            {"path": "README.md", "type": "blob"},
        ]

    async def content(self, ref, path, sha):
        return {
            "calculator.py": "def add(a,b):\n    return a-b\n",
            "test_calculator.py": "def test_add():\n    assert add(2,3)==5\n",
        }.get(path, "")


class FakeAI:
    def diagnose(self, context):
        assert "calculator.py" in context
        return {
            "summary": "addition uses subtraction",
            "root_cause": "wrong operator",
            "confidence": 0.99,
            "affected_files": ["calculator.py"],
            "evidence": ["test failure"],
            "repair_strategy": "replace subtraction with addition",
            "risk_notes": [],
        }


def test_inspect_builds_realistic_session(monkeypatch):
    monkeypatch.setattr(main, "GitHubClient", FakeGitHub)
    client = TestClient(main.app)
    main.SESSIONS.clear()

    response = client.post(
        "/api/inspect",
        json={"pr_url": "https://github.com/test/repo/pull/1"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["evidence_files"] >= 1
    assert data["context_chars"] > 0
    assert data["files"][0]["path"] == "calculator.py"


def test_analyze_returns_normalized_diagnosis(monkeypatch):
    monkeypatch.setattr(main, "GitHubClient", FakeGitHub)
    monkeypatch.setattr(main, "AIEngine", FakeAI)
    client = TestClient(main.app)
    main.SESSIONS.clear()

    inspect = client.post(
        "/api/inspect",
        json={"pr_url": "https://github.com/test/repo/pull/1"},
    ).json()

    response = client.post(
        "/api/analyze",
        json={"session_id": inspect["session_id"]},
    )

    assert response.status_code == 200
    assert response.json()["confidence"] == 0.99


def test_repair_attempt_two_receives_verification_feedback(monkeypatch):
    monkeypatch.setattr(main, "GitHubClient", FakeGitHub)

    class RetryAI:
        feedback_seen = ""

        def diagnose(self, context):
            return {
                "summary": "bug",
                "root_cause": "wrong operator",
                "confidence": 0.9,
                "affected_files": ["calculator.py"],
                "evidence": ["failure"],
                "repair_strategy": "change operator",
                "risk_notes": [],
            }

        def generate_patch(self, context, diagnosis, contents, verification_feedback=""):
            RetryAI.feedback_seen = verification_feedback
            return {
                "patch": "--- a/calculator.py\n+++ b/calculator.py\n@@ -1 +1 @@\n-a\n+b\n",
                "explanation": "fix",
                "touched_files": ["calculator.py"],
                "confidence": 0.9,
            }

    monkeypatch.setattr(main, "AIEngine", RetryAI)
    client = TestClient(main.app)
    main.SESSIONS.clear()

    inspect = client.post(
        "/api/inspect",
        json={"pr_url": "https://github.com/test/repo/pull/1"},
    ).json()
    sid = inspect["session_id"]

    assert client.post("/api/analyze", json={"session_id": sid}).status_code == 200
    main.SESSIONS[sid]["verifications"].append({"passed": False, "output": "pytest failed: AssertionError"})

    response = client.post(
        "/api/repair",
        json={"session_id": sid, "attempt": 2},
    )
    assert response.status_code == 200
    assert "AssertionError" in RetryAI.feedback_seen


def test_verify_records_sandbox_result(monkeypatch, tmp_path):
    client = TestClient(main.app)
    main.SESSIONS.clear()
    session_id = "verify-session"
    main.SESSIONS[session_id] = {
        "ref": main.RepoRef("test", "repo", 1),
        "pr": {"head": {"sha": "abc123"}},
        "repairs": [{
            "attempt": 1,
            "patch": "--- a/calculator.py\n+++ b/calculator.py\n@@ -1 +1 @@\n-a\n+b\n",
        }],
        "verifications": [],
    }

    def fake_checkout(_url, _sha, root):
        root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(main, "checkout", fake_checkout)
    monkeypatch.setattr(main, "apply_patch", lambda *_: (True, ""))
    monkeypatch.setattr(main, "detect_project", lambda _: ("python", "pytest -q"))
    monkeypatch.setattr(main, "docker_test", lambda *_: RunResult(True, "pytest -q", 0, "1 passed"))

    response = client.post("/api/verify", json={"session_id": session_id, "attempt": 1})
    assert response.status_code == 200
    assert response.json()["passed"] is True
    assert main.SESSIONS[session_id]["verification"]["output"] == "1 passed"


def test_verify_rejects_sensitive_patch_before_checkout(monkeypatch):
    client = TestClient(main.app)
    main.SESSIONS.clear()
    session_id = "unsafe-verify"
    main.SESSIONS[session_id] = {
        "repairs": [{"attempt": 1, "patch": "--- a/.env\n+++ b/.env\n@@ -1 +1 @@\n-a\n+b\n"}],
        "verifications": [],
    }
    monkeypatch.setattr(main, "checkout", lambda *_: (_ for _ in ()).throw(AssertionError("must not checkout")))

    response = client.post("/api/verify", json={"session_id": session_id, "attempt": 1})
    assert response.status_code == 400
    assert "safety policy" in response.json()["detail"]



def test_verify_requires_exact_repair_attempt(monkeypatch):
    client = TestClient(main.app)
    main.SESSIONS.clear()
    session_id = "attempt-check"
    main.SESSIONS[session_id] = {
        "repairs": [{
            "attempt": 1,
            "patch": "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-a\n+b\n",
        }],
        "verifications": [],
    }

    monkeypatch.setattr(
        main,
        "checkout",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not checkout")),
    )

    response = client.post(
        "/api/verify",
        json={"session_id": session_id, "attempt": 2},
    )

    assert response.status_code == 400
    assert "attempt 2" in response.json()["detail"]


def test_verify_internal_failure_is_server_error(monkeypatch):
    client = TestClient(main.app)
    main.SESSIONS.clear()
    session_id = "verify-error"
    main.SESSIONS[session_id] = {
        "ref": main.RepoRef("test", "repo", 1),
        "pr": {"head": {"sha": "abc123"}},
        "repairs": [{
            "attempt": 1,
            "patch": "--- a/a.py\\n+++ b/a.py\\n@@ -1 +1 @@\\n-a\\n+b\\n",
        }],
        "verifications": [],
    }

    monkeypatch.setattr(
        main,
        "checkout",
        lambda *_: (_ for _ in ()).throw(RuntimeError("sandbox setup broke")),
    )

    response = client.post(
        "/api/verify",
        json={"session_id": session_id, "attempt": 1},
    )

    assert response.status_code == 500
    assert "verification error" in response.json()["detail"].lower()
