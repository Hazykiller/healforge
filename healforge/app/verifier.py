from __future__ import annotations

import abc
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .ai import _redact_secrets
from .config import settings
from .runner import (
    ProjectProfile,
    RunResult,
    classify_failure,
    detect_project_profile,
    docker_test,
    _docker_available,
)


@dataclass
class VerificationResult:
    status: str  # VERIFIED, TEST_FAILED, BUILD_FAILED, UNSUPPORTED, SANDBOX_UNAVAILABLE, TIMEOUT, SECURITY_BLOCKED
    passed: bool
    exit_code: int
    output: str
    duration_ms: int
    verifier_type: str  # "docker", "remote", "none"
    category: str
    command: str = ""
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "exit_code": self.exit_code,
            "output": self.output,
            "duration_ms": self.duration_ms,
            "verifier_type": self.verifier_type,
            "category": self.category,
            "command": self.command,
            "evidence": self.evidence,
        }


class BaseVerifier(abc.ABC):
    @abc.abstractmethod
    def verify(
        self,
        root: Path,
        profile: ProjectProfile,
        command: str = "",
        patch: str = "",
        files: dict[str, str] | None = None,
    ) -> VerificationResult:
        raise NotImplementedError


class LocalDockerVerifier(BaseVerifier):
    """
    Hardened local Docker sandbox verification.
    Preserves all isolation invariants: --network none, --cap-drop ALL,
    --read-only, executable tmpfs, non-destructive overlay mounts.
    """

    def verify(
        self,
        root: Path,
        profile: ProjectProfile,
        command: str = "",
        patch: str = "",
        files: dict[str, str] | None = None,
    ) -> VerificationResult:
        t_start = time.perf_counter()
        test_cmd = command.strip() if command and command.strip() else profile.test_command

        try:
            import app.main as main_mod
            import app.runner as runner_mod
            if getattr(main_mod, "docker_test", None) is not getattr(runner_mod, "docker_test", None):
                test_fn = main_mod.docker_test
            else:
                test_fn = docker_test
        except Exception:
            test_fn = docker_test

        run: RunResult = test_fn(root, profile.language, test_cmd)
        duration_ms = int((time.perf_counter() - t_start) * 1000)

        category = getattr(run, "category", "") or classify_failure(run)
        if run.passed:
            status = "VERIFIED"
        elif category == "TIMEOUT":
            status = "TIMEOUT"
        elif category in {"ENVIRONMENT_FAILURE", "INFRASTRUCTURE_FAILURE"}:
            status = "BUILD_FAILED"
        else:
            status = "TEST_FAILED"

        return VerificationResult(
            status=status,
            passed=run.passed,
            exit_code=run.exit_code,
            output=_redact_secrets(run.output),
            duration_ms=duration_ms,
            verifier_type="docker",
            category=category,
            command=test_cmd,
            evidence=list(profile.evidence),
        )


class RemoteSandboxVerifier(BaseVerifier):
    """
    Remote cloud sandbox verification.
    Executes repository test commands within an isolated remote container environment,
    enabling full verification on serverless platforms (Vercel) with zero local Docker daemon.
    """

    def __init__(self, endpoint: str | None = None, token: str | None = None):
        self.endpoint = (endpoint or settings.sandbox_url).strip()
        self.token = (token or settings.sandbox_token).strip()

    def is_configured(self) -> bool:
        return bool(self.endpoint)

    def verify(
        self,
        root: Path,
        profile: ProjectProfile,
        command: str = "",
        patch: str = "",
        files: dict[str, str] | None = None,
    ) -> VerificationResult:
        t_start = time.perf_counter()
        test_cmd = command.strip() if command and command.strip() else profile.test_command

        if not self.is_configured():
            duration_ms = int((time.perf_counter() - t_start) * 1000)
            return VerificationResult(
                status="SANDBOX_UNAVAILABLE",
                passed=False,
                exit_code=1,
                output=(
                    "Remote sandbox is not configured (SANDBOX_URL missing). "
                    "In production serverless mode, repository code cannot be executed locally. "
                    "Configure SANDBOX_URL or run in an environment with Docker available."
                ),
                duration_ms=duration_ms,
                verifier_type="none",
                category="ENVIRONMENT_FAILURE",
                command=test_cmd,
            )

        # Collect source files to transmit if not explicitly provided
        payload_files: dict[str, str] = {}
        if files:
            payload_files = dict(files)
        elif root.exists():
            max_bytes = getattr(settings, "max_sandbox_file_bytes", 1_000_000)
            for path in root.rglob("*"):
                if path.is_file() and ".git" not in path.parts:
                    try:
                        rel = path.relative_to(root).as_posix()
                        if path.stat().st_size <= max_bytes:
                            payload_files[rel] = path.read_text(encoding="utf-8", errors="replace")
                    except Exception:
                        continue

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent": "HEALFORGE-Remote-Verifier/1.0",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        timeout_sec = float(getattr(settings, "sandbox_timeout_seconds", 120))
        payload = {
            "language": profile.language,
            "framework": profile.framework,
            "package_manager": profile.package_manager,
            "command": test_cmd,
            "patch": patch,
            "files": payload_files,
            "timeout_seconds": int(timeout_sec),
        }

        try:
            with httpx.Client(timeout=timeout_sec) as client:
                resp = client.post(self.endpoint, headers=headers, json=payload)
        except httpx.TimeoutException:
            duration_ms = int((time.perf_counter() - t_start) * 1000)
            return VerificationResult(
                status="TIMEOUT",
                passed=False,
                exit_code=124,
                output=f"Remote sandbox execution timed out after {int(timeout_sec)}s.",
                duration_ms=duration_ms,
                verifier_type="remote",
                category="TIMEOUT",
                command=test_cmd,
            )
        except Exception as exc:
            duration_ms = int((time.perf_counter() - t_start) * 1000)
            return VerificationResult(
                status="SANDBOX_UNAVAILABLE",
                passed=False,
                exit_code=1,
                output=f"Failed to connect to remote sandbox service: {_redact_secrets(str(exc))}",
                duration_ms=duration_ms,
                verifier_type="remote",
                category="INFRASTRUCTURE_FAILURE",
                command=test_cmd,
            )

        duration_ms = int((time.perf_counter() - t_start) * 1000)
        if resp.status_code == 401 or resp.status_code == 403:
            return VerificationResult(
                status="SANDBOX_UNAVAILABLE",
                passed=False,
                exit_code=1,
                output="Remote sandbox authentication failed: invalid or unauthorized SANDBOX_TOKEN.",
                duration_ms=duration_ms,
                verifier_type="remote",
                category="ENVIRONMENT_FAILURE",
                command=test_cmd,
            )
        elif resp.status_code == 422 or resp.status_code == 400:
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            msg = data.get("message") or resp.text[:400]
            status = "UNSUPPORTED" if "unsupported" in msg.lower() else "BUILD_FAILED"
            return VerificationResult(
                status=status,
                passed=False,
                exit_code=1,
                output=f"Remote sandbox rejected execution request: {_redact_secrets(msg)}",
                duration_ms=duration_ms,
                verifier_type="remote",
                category="ENVIRONMENT_FAILURE",
                command=test_cmd,
            )
        elif resp.status_code != 200:
            return VerificationResult(
                status="SANDBOX_UNAVAILABLE",
                passed=False,
                exit_code=1,
                output=f"Remote sandbox returned HTTP {resp.status_code}: {_redact_secrets(resp.text[:300])}",
                duration_ms=duration_ms,
                verifier_type="remote",
                category="INFRASTRUCTURE_FAILURE",
                command=test_cmd,
            )

        try:
            res_data = resp.json()
        except Exception as exc:
            return VerificationResult(
                status="BUILD_FAILED",
                passed=False,
                exit_code=1,
                output=f"Invalid JSON received from remote sandbox: {exc}",
                duration_ms=duration_ms,
                verifier_type="remote",
                category="INFRASTRUCTURE_FAILURE",
                command=test_cmd,
            )

        passed = bool(res_data.get("passed", False))
        exit_code = int(res_data.get("exit_code", 0 if passed else 1))
        output = _redact_secrets(str(res_data.get("output", "")))
        cat = str(res_data.get("category", "SUCCESS" if passed else "PATCH_FAILURE"))
        status = "VERIFIED" if passed else ("TIMEOUT" if cat == "TIMEOUT" else "TEST_FAILED")

        return VerificationResult(
            status=status,
            passed=passed,
            exit_code=exit_code,
            output=output,
            duration_ms=duration_ms,
            verifier_type="remote",
            category=cat,
            command=test_cmd,
            evidence=list(profile.evidence),
        )


class UnavailableVerifier(BaseVerifier):
    """
    Safe refusal when neither local Docker nor remote sandbox execution is available.
    Strictly prevents executing arbitrary untrusted code directly on the host.
    """

    def __init__(self, reason: str = ""):
        self.reason = reason or (
            "No isolated sandbox execution environment is available. "
            "Local Docker is not running and no remote sandbox (SANDBOX_URL) is configured. "
            "Untrusted repository code cannot be safely verified."
        )

    def verify(
        self,
        root: Path,
        profile: ProjectProfile,
        command: str = "",
        patch: str = "",
        files: dict[str, str] | None = None,
    ) -> VerificationResult:
        return VerificationResult(
            status="SANDBOX_UNAVAILABLE",
            passed=False,
            exit_code=1,
            output=self.reason,
            duration_ms=0,
            verifier_type="none",
            category="ENVIRONMENT_FAILURE",
            command=command or profile.test_command,
            evidence=list(profile.evidence),
        )


def get_verifier() -> BaseVerifier:
    """
    Determine the appropriate verification provider based on deployment mode and configuration:
    1. If SANDBOX_PROVIDER == 'remote' -> RemoteSandboxVerifier
    2. If SANDBOX_PROVIDER == 'docker' -> LocalDockerVerifier (if available) or UnavailableVerifier
    3. If SANDBOX_PROVIDER == 'auto' (default):
       - If running on Vercel / serverless: RemoteSandboxVerifier if configured, else UnavailableVerifier
       - If local Docker is running and available: LocalDockerVerifier
       - If SANDBOX_URL is set: RemoteSandboxVerifier
       - Else: UnavailableVerifier
    """
    provider_mode = getattr(settings, "sandbox_provider", "auto").strip().lower()
    is_vercel = getattr(settings, "is_vercel", False) or bool(os.getenv("VERCEL"))
    remote_url = getattr(settings, "sandbox_url", "").strip()

    if provider_mode == "remote":
        if remote_url:
            return RemoteSandboxVerifier()
        return UnavailableVerifier("SANDBOX_PROVIDER is set to 'remote' but SANDBOX_URL is not configured.")

    if provider_mode == "docker":
        if _docker_available():
            return LocalDockerVerifier()
        return UnavailableVerifier("SANDBOX_PROVIDER is set to 'docker' but local Docker daemon is not available.")

    # 'auto' mode
    if is_vercel:
        # In serverless environment, local Docker can never be used
        if remote_url:
            return RemoteSandboxVerifier()
        return UnavailableVerifier(
            "Running in serverless/cloud environment without local Docker. "
            "Configure SANDBOX_URL to enable remote container verification."
        )

    # Local environment
    if _docker_available():
        return LocalDockerVerifier()

    if remote_url:
        return RemoteSandboxVerifier()

    return UnavailableVerifier(
        "Docker Desktop is not running locally and no remote sandbox (SANDBOX_URL) is configured. "
        "Start Docker Desktop or set SANDBOX_URL to enable sandbox verification."
    )
