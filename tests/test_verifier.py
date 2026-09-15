from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import httpx

from app.config import settings
from app.runner import ProjectProfile, RunResult
from app.verifier import (
    BaseVerifier,
    LocalDockerVerifier,
    RemoteSandboxVerifier,
    UnavailableVerifier,
    VerificationResult,
    get_verifier,
)


def _dummy_profile(language: str = "python") -> ProjectProfile:
    return ProjectProfile(
        language=language,
        framework="pytest",
        package_manager="pip",
        test_command="pytest -v",
        docker_image="python:3.12-slim",
        confidence=1.0,
        evidence=["test evidence"],
    )


@pytest.fixture
def patch_settings():
    saved = {}
    def _apply(**kwargs):
        for k, v in kwargs.items():
            if k not in saved:
                saved[k] = getattr(settings, k)
            object.__setattr__(settings, k, v)
    yield _apply
    for k, v in saved.items():
        object.__setattr__(settings, k, v)


# ---------------------------------------------------------------------------
# Provider Selection Tests
# ---------------------------------------------------------------------------

def test_verifier_selection_local_docker_available(patch_settings, monkeypatch):
    patch_settings(sandbox_provider="auto", is_vercel=False)
    monkeypatch.setattr("app.verifier._docker_available", lambda: True)

    verifier = get_verifier()
    assert isinstance(verifier, LocalDockerVerifier)


def test_verifier_selection_docker_unavailable_remote_configured(patch_settings, monkeypatch):
    patch_settings(
        sandbox_provider="auto",
        is_vercel=False,
        sandbox_url="https://sandbox.healforge.internal/execute",
    )
    monkeypatch.setattr("app.verifier._docker_available", lambda: False)

    verifier = get_verifier()
    assert isinstance(verifier, RemoteSandboxVerifier)
    assert verifier.endpoint == "https://sandbox.healforge.internal/execute"


def test_verifier_selection_both_unavailable(patch_settings, monkeypatch):
    patch_settings(sandbox_provider="auto", is_vercel=False, sandbox_url="")
    monkeypatch.setattr("app.verifier._docker_available", lambda: False)

    verifier = get_verifier()
    assert isinstance(verifier, UnavailableVerifier)
    res = verifier.verify(Path("/tmp"), _dummy_profile())
    assert res.status == "SANDBOX_UNAVAILABLE"
    assert res.passed is False
    assert "SANDBOX_UNAVAILABLE" in res.status


def test_verifier_selection_vercel_environment(patch_settings, monkeypatch):
    patch_settings(
        sandbox_provider="auto",
        is_vercel=True,
        sandbox_url="https://cloud-sandbox.internal/run",
    )
    # Even if local docker happened to return True, Vercel must use remote
    monkeypatch.setattr("app.verifier._docker_available", lambda: True)

    verifier = get_verifier()
    assert isinstance(verifier, RemoteSandboxVerifier)


def test_verifier_selection_vercel_without_remote(patch_settings, monkeypatch):
    patch_settings(sandbox_provider="auto", is_vercel=True, sandbox_url="")

    verifier = get_verifier()
    assert isinstance(verifier, UnavailableVerifier)
    res = verifier.verify(Path("/tmp"), _dummy_profile())
    assert res.status == "SANDBOX_UNAVAILABLE"
    assert res.passed is False


# ---------------------------------------------------------------------------
# Remote Sandbox Execution Tests (Mocked API)
# ---------------------------------------------------------------------------

def test_remote_verifier_success(monkeypatch):
    def mock_post(url, headers=None, json=None):
        return httpx.Response(
            200,
            json={
                "passed": True,
                "exit_code": 0,
                "output": "2 passed in 0.05s",
                "category": "SUCCESS",
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kwargs: mock_post(url, **kwargs))

    verifier = RemoteSandboxVerifier(endpoint="https://api.sandbox.com/run", token="secret-token-123")
    res = verifier.verify(Path("/tmp"), _dummy_profile(), command="pytest", patch="diff")

    assert res.status == "VERIFIED"
    assert res.passed is True
    assert res.exit_code == 0
    assert "2 passed" in res.output
    assert res.verifier_type == "remote"


def test_remote_verifier_test_failed(monkeypatch):
    def mock_post(url, headers=None, json=None):
        return httpx.Response(
            200,
            json={
                "passed": False,
                "exit_code": 1,
                "output": "FAILED tests/test_calc.py::test_add - AssertionError",
                "category": "PATCH_FAILURE",
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kwargs: mock_post(url, **kwargs))

    verifier = RemoteSandboxVerifier(endpoint="https://api.sandbox.com/run")
    res = verifier.verify(Path("/tmp"), _dummy_profile())

    assert res.status == "TEST_FAILED"
    assert res.passed is False
    assert res.exit_code == 1
    assert "AssertionError" in res.output


def test_remote_verifier_timeout(monkeypatch):
    def mock_post(url, headers=None, json=None):
        raise httpx.TimeoutException("Remote connection timed out")

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kwargs: mock_post(url, **kwargs))

    verifier = RemoteSandboxVerifier(endpoint="https://api.sandbox.com/run")
    res = verifier.verify(Path("/tmp"), _dummy_profile())

    assert res.status == "TIMEOUT"
    assert res.passed is False
    assert res.exit_code == 124


def test_remote_verifier_unsupported_language(monkeypatch):
    def mock_post(url, headers=None, json=None):
        return httpx.Response(
            422,
            json={"message": "Unsupported language environment: cobol"},
            headers={"content-type": "application/json"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kwargs: mock_post(url, **kwargs))

    verifier = RemoteSandboxVerifier(endpoint="https://api.sandbox.com/run")
    res = verifier.verify(Path("/tmp"), _dummy_profile("cobol"))

    assert res.status == "UNSUPPORTED"
    assert res.passed is False


def test_remote_verifier_redacts_secrets(monkeypatch):
    fake_token = "SANDBOX_SECRET_KEY_xyz987"
    def mock_post(url, headers=None, json=None):
        raise RuntimeError(f"Connection failed to {url} with Bearer {fake_token}")

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kwargs: mock_post(url, **kwargs))

    verifier = RemoteSandboxVerifier(endpoint="https://api.sandbox.com/run", token=fake_token)
    res = verifier.verify(Path("/tmp"), _dummy_profile())

    assert fake_token not in res.output
    assert "REDACTED" in res.output or fake_token not in res.output


# ---------------------------------------------------------------------------
# Local Docker Verifier Tests (Mocked docker_test)
# ---------------------------------------------------------------------------

def test_local_docker_verifier_success(monkeypatch):
    dummy_run = RunResult(
        passed=True,
        command="pytest -v",
        exit_code=0,
        output="test_add PASSED",
        category="SUCCESS",
    )
    monkeypatch.setattr("app.verifier.docker_test", lambda root, lang, cmd: dummy_run)

    verifier = LocalDockerVerifier()
    res = verifier.verify(Path("/tmp"), _dummy_profile())

    assert res.status == "VERIFIED"
    assert res.passed is True
    assert res.exit_code == 0
    assert res.verifier_type == "docker"


def test_local_docker_verifier_failure(monkeypatch):
    dummy_run = RunResult(
        passed=False,
        command="pytest -v",
        exit_code=1,
        output="test_add FAILED: AssertionError: 35 == 12",
        category="PATCH_FAILURE",
    )
    monkeypatch.setattr("app.verifier.docker_test", lambda root, lang, cmd: dummy_run)

    verifier = LocalDockerVerifier()
    res = verifier.verify(Path("/tmp"), _dummy_profile())

    assert res.status == "TEST_FAILED"
    assert res.passed is False
    assert res.exit_code == 1


def test_vercel_health_endpoint_reports_remote(patch_settings, monkeypatch):
    import tempfile
    from fastapi.testclient import TestClient
    from app import main

    patch_settings(
        sandbox_provider="auto",
        is_vercel=True,
        sandbox_url="https://sandbox.api.healforge/run",
    )
    client = TestClient(main.app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_vercel"] is True
    assert data["verifier_type"] == "remote"
    assert data["verifier_status"] == "REMOTE_SANDBOX_CONFIGURED"


def test_vercel_health_endpoint_reports_unavailable_if_no_sandbox(patch_settings, monkeypatch):
    from fastapi.testclient import TestClient
    from app import main

    patch_settings(
        sandbox_provider="auto",
        is_vercel=True,
        sandbox_url="",
    )
    client = TestClient(main.app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_vercel"] is True
    assert data["verifier_type"] == "none"
    assert data["verifier_status"] == "SANDBOX_UNAVAILABLE"


def test_vercel_workspace_root_resolves_to_tmp(patch_settings):
    import tempfile
    from app.main import _workspace_root

    patch_settings(is_vercel=True)
    ws = _workspace_root()
    assert str(tempfile.gettempdir()).lower() in str(ws).lower()

