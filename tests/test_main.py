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

    def fake_checkout(_url, _sha, root, **_kw):
        root.mkdir(parents=True, exist_ok=True)

    from app.runner import ProjectProfile
    from app.verifier import VerificationResult
    fake_profile = ProjectProfile(
        language="python",
        framework="pytest",
        package_manager="pip",
        test_command="pytest -q",
        docker_image="python:3.11-slim",
        confidence=0.99,
        evidence=["mocked"],
    )

    class FakeVerifier:
        def verify(self, root, profile, command="", patch="", files=None):
            return VerificationResult(
                status="VERIFIED",
                passed=True,
                exit_code=0,
                output="1 passed",
                duration_ms=100,
                verifier_type="docker",
                category="SUCCESS",
                command=command or profile.test_command,
            )


    monkeypatch.setattr(main, "checkout", fake_checkout)
    monkeypatch.setattr(main, "apply_patch", lambda *_: (True, ""))
    monkeypatch.setattr(main, "detect_project", lambda _: ("python", "pytest -q"))
    monkeypatch.setattr(main, "detect_project_profile", lambda _: fake_profile)
    monkeypatch.setattr(main, "get_verifier", lambda: FakeVerifier())

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

    from app.verifier import LocalDockerVerifier

    class BreakingVerifier:
        """A verifier that reaches prepare_attempt_workspace but then crashes."""
        def verify(self, root, profile, command="", patch="", files=None):
            raise RuntimeError("sandbox setup broke")

    monkeypatch.setattr(main, "get_verifier", lambda: BreakingVerifier())
    monkeypatch.setattr(main, "apply_patch", lambda *_: (True, ""))
    monkeypatch.setattr(main, "detect_project", lambda _: ("python", "pytest -q"))

    from app.runner import ProjectProfile
    fake_profile = ProjectProfile(
        language="python", framework="pytest", package_manager="pip",
        test_command="pytest -q", docker_image="python:3.11-slim",
        confidence=0.99, evidence=["mocked"],
    )
    monkeypatch.setattr(main, "detect_project_profile", lambda _: fake_profile)

    def fake_checkout(_url, _sha, dest, **_kw):
        dest.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(main, "checkout", fake_checkout)

    response = client.post(
        "/api/verify",
        json={"session_id": session_id, "attempt": 1},
    )

    assert response.status_code == 500
    assert "verification error" in response.json()["detail"].lower()


def test_health_reports_ai_status_without_calling_model(monkeypatch):
    client = TestClient(main.app)
    main.record_ai_success()
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert "ai_status" in data
    assert data["ai_status"] == "AVAILABLE"


def test_analyze_handles_quota_exhausted_error(monkeypatch):
    import time
    from app.ai import AIQuotaExhaustedError

    future_reset = str(int(time.time() + 3600))

    class QuotaFailingAI:
        def diagnose(self, context):
            raise AIQuotaExhaustedError("OpenRouter free-model daily quota is exhausted.", reset_timestamp=future_reset)

    monkeypatch.setattr(main, "AIEngine", QuotaFailingAI)
    client = TestClient(main.app)
    main.SESSIONS.clear()
    main.record_ai_success()  # reset cache
    main.SESSIONS["test-session"] = {"context": "failure evidence"}

    response = client.post("/api/analyze", json={"session_id": "test-session"})
    assert response.status_code == 429
    detail = response.json()["detail"]
    assert detail["error"] == "AI_QUOTA_EXHAUSTED"
    assert detail["reset_timestamp"] == future_reset

    # Verify health endpoint now reflects the exhausted quota
    health_res = client.get("/api/health")
    assert health_res.json()["ai_status"] == "DAILY_QUOTA_EXHAUSTED"

    # Subsequent request is short-circuited without calling model again
    short_circuit_res = client.post("/api/analyze", json={"session_id": "test-session"})
    assert short_circuit_res.status_code == 429
    assert "avoid wasting quota" in short_circuit_res.json()["detail"]["message"].lower()

    # Reset cache back to available for subsequent test isolation
    main.record_ai_success()

