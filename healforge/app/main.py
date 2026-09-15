import asyncio
import json
import logging
import os
import re
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .ai import AIEngine, AIQuotaExhaustedError, AIModelRateLimitError
from .config import settings
from .context import build_context, is_sensitive_path
from .security import is_safe_repo_path, normalize_repo_path
from .github import GitHubClient, RepoRef, parse_pr_url
from .runner import (
    apply_patch,
    checkout,
    classify_failure,
    detect_project,
    detect_project_profile,
    docker_test,
    prepare_attempt_workspace,
    summarize_verification_failure,
    validate_patch_paths,
    ProjectProfile,
)
from .schemas import AnalyzeRequest, InspectRequest, PublishRequest, RepairRequest
from .verifier import get_verifier, VerificationResult, UnavailableVerifier


app = FastAPI(title="HEALFORGE", version="2.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
SESSIONS: dict[str, dict] = {}
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Lightweight AI status tracking (Directive 10)
# Tracks availability without calling the model or wasting quota on health checks.
_AI_STATUS_CACHE: dict[str, Any] = {
    "status": "AVAILABLE",
    "reset_timestamp": None,
    "remedy_hint": None,
    "last_checked_at": 0.0,
    "message": None,
}


def record_ai_success() -> None:
    _AI_STATUS_CACHE["status"] = "AVAILABLE"
    _AI_STATUS_CACHE["reset_timestamp"] = None
    _AI_STATUS_CACHE["remedy_hint"] = None
    _AI_STATUS_CACHE["last_checked_at"] = time.time()
    _AI_STATUS_CACHE["message"] = None


def record_ai_quota_exhausted(
    reset_ts: str | None = None,
    hint: str | None = None,
    message: str | None = None,
) -> None:
    _AI_STATUS_CACHE["status"] = "DAILY_QUOTA_EXHAUSTED"
    _AI_STATUS_CACHE["reset_timestamp"] = reset_ts
    _AI_STATUS_CACHE["remedy_hint"] = hint
    _AI_STATUS_CACHE["last_checked_at"] = time.time()
    provider = getattr(settings, "ai_provider", "AI").capitalize()
    _AI_STATUS_CACHE["message"] = message or f"{provider} daily quota is exhausted."


def record_ai_temporarily_unavailable(message: str | None = None) -> None:
    _AI_STATUS_CACHE["status"] = "TEMPORARILY_UNAVAILABLE"
    _AI_STATUS_CACHE["last_checked_at"] = time.time()
    _AI_STATUS_CACHE["message"] = message or "AI service is temporarily unavailable."


def check_ai_quota_status() -> None:
    """Raise HTTP 429 if the daily quota is known to be currently exhausted without expiring."""
    if _AI_STATUS_CACHE["status"] == "DAILY_QUOTA_EXHAUSTED":
        reset_ts = _AI_STATUS_CACHE.get("reset_timestamp")
        if reset_ts:
            try:
                if float(reset_ts) <= time.time():
                    _AI_STATUS_CACHE["status"] = "AVAILABLE"
                    return
            except (ValueError, TypeError):
                pass
        elif time.time() - _AI_STATUS_CACHE.get("last_checked_at", 0) > 3600:
            _AI_STATUS_CACHE["status"] = "AVAILABLE"
            return

        provider = getattr(settings, "ai_provider", "openrouter").capitalize()
        raise HTTPException(
            429,
            detail={
                "error": "AI_QUOTA_EXHAUSTED",
                "message": f"{provider} free-model daily quota is exhausted. No repair request was attempted further to avoid wasting quota.",
                "reset_timestamp": _AI_STATUS_CACHE.get("reset_timestamp"),
                "remedy_hint": _AI_STATUS_CACHE.get("remedy_hint"),
            },
        )


def _workspace_root() -> Path:
    if getattr(settings, "is_vercel", False) or bool(os.getenv("VERCEL")):
        root = Path(tempfile.gettempdir()) / "healforge_workspace"
        root.mkdir(parents=True, exist_ok=True)
        return root

    configured = Path(settings.host_workspace)
    root = (PROJECT_ROOT / configured).resolve() if not configured.is_absolute() else configured.resolve()
    try:
        root.relative_to(PROJECT_ROOT)
        root.mkdir(parents=True, exist_ok=True)
        test_file = root / ".write_test"
        test_file.touch()
        test_file.unlink()
    except (ValueError, OSError, PermissionError):
        root = Path(tempfile.gettempdir()) / "healforge_workspace"
        root.mkdir(parents=True, exist_ok=True)
    return root


def _save_session(session_id: str, session: dict) -> None:
    SESSIONS[session_id] = session
    try:
        root = _workspace_root()
        s_dir = root / session_id
        s_dir.mkdir(parents=True, exist_ok=True)
        s_path = s_dir / "session.json"
        serializable = dict(session)
        if hasattr(serializable.get("ref"), "owner"):
            serializable["ref"] = {
                "owner": serializable["ref"].owner,
                "repo": serializable["ref"].repo,
                "number": serializable["ref"].number,
            }
        s_path.write_text(json.dumps(serializable, default=str), encoding="utf-8")
    except Exception:
        pass


async def _fetch_contents(gh, ref, paths: list[str], sha: str) -> dict[str, str]:
    unique = list(dict.fromkeys(paths))
    if not unique:
        return {}
    semaphore = asyncio.Semaphore(max(1, min(settings.github_concurrency, 16)))

    async def fetch(path: str) -> tuple[str, str]:
        async with semaphore:
            try:
                return path, await gh.content(ref, path, sha)
            except Exception:
                return path, ""

    results = await asyncio.gather(*(fetch(path) for path in unique))
    return {path: content for path, content in results if content}


def session_or_404(session_id: str) -> dict:
    session = SESSIONS.get(session_id)
    if not session:
        try:
            s_path = _workspace_root() / session_id / "session.json"
            if s_path.exists():
                session = json.loads(s_path.read_text(encoding="utf-8"))
                if isinstance(session.get("ref"), dict):
                    session["ref"] = RepoRef(**session["ref"])
                SESSIONS[session_id] = session
        except Exception as exc:
            logger.warning("Failed to load session %s from disk: %s", session_id, exc)
    if not session:
        raise HTTPException(404, "Session not found")
    return session


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/api/health")
def health():
    ai_configured = settings.is_ai_configured
    ai_status = "NOT_CONFIGURED"
    if ai_configured:
        cached_status = _AI_STATUS_CACHE["status"]
        if cached_status == "DAILY_QUOTA_EXHAUSTED":
            reset_ts = _AI_STATUS_CACHE.get("reset_timestamp")
            if reset_ts:
                try:
                    if float(reset_ts) <= time.time():
                        _AI_STATUS_CACHE["status"] = "AVAILABLE"
                        cached_status = "AVAILABLE"
                except (ValueError, TypeError):
                    pass
        ai_status = cached_status

    return {
        "ok": True,
        "github_configured": bool(settings.github_token),
        "ai_configured": ai_configured,
        "ai_provider": settings.ai_provider,
        "ai_status": ai_status,
        "ai_reset_timestamp": _AI_STATUS_CACHE.get("reset_timestamp"),
        "ai_remedy_hint": _AI_STATUS_CACHE.get("remedy_hint"),
        "ai_status_message": _AI_STATUS_CACHE.get("message"),
        "ai_models": settings.ai_models if ai_configured else [],
        "max_repair_attempts": settings.max_repair_attempts,
        "verifier_type": settings.verifier_type,
        "verifier_status": settings.verifier_status,
        "sandbox_provider": settings.sandbox_provider,
        "is_vercel": settings.is_vercel,
    }


def _safe_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return is_safe_repo_path(normalized) and not is_sensitive_path(normalized)


def _path_candidates_from_import(path: str, source: str, tree_paths: set[str]) -> set[str]:
    result: set[str] = set()
    suffix = Path(path).suffix.lower()
    parent = Path(path).parent

    if suffix == ".py":
        modules = re.findall(
            r"(?m)^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_.]*)",
            source,
        )
        for module_name in modules:
            module = module_name.replace(".", "/")
            possible = {
                f"{module}.py",
                f"{module}/__init__.py",
                (parent / f"{module}.py").as_posix(),
                (parent / module / "__init__.py").as_posix(),
            }
            for candidate in possible:
                if candidate in tree_paths:
                    result.add(candidate)

    elif suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
        imports = re.findall(
            r"(?:from\s+|import\s*\(\s*|require\s*\(\s*)['\"]([^'\"]+)['\"]",
            source,
        )
        extensions = ["", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", "/index.js", "/index.ts"]
        for imported in imports:
            if not imported.startswith("."):
                continue
            base = (parent / imported).as_posix()
            for extension in extensions:
                candidate = base + extension
                if candidate in tree_paths:
                    result.add(candidate)

    return result


def _score_repository_paths(
    tree_paths: list[str],
    changed_paths: list[str],
) -> list[str]:
    changed = {p.replace("\\", "/") for p in changed_paths}
    changed_stems = {Path(p).stem.lower() for p in changed}
    changed_parents = {str(Path(p).parent).replace("\\", "/") for p in changed}

    metadata = {
        "pyproject.toml", "requirements.txt", "requirements-dev.txt", "setup.py", "setup.cfg",
        "pytest.ini", "tox.ini", "package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock",
        "tsconfig.json", "vite.config.js", "vite.config.ts", "jest.config.js", "jest.config.ts",
        "vitest.config.js", "vitest.config.ts", "pom.xml", "build.gradle", "build.gradle.kts",
        "settings.gradle", "settings.gradle.kts", "go.mod", "go.sum", "cargo.toml", "cargo.lock",
        "cmakelists.txt", "makefile",
    }

    scored: list[tuple[int, str]] = []
    for path in tree_paths:
        if not path or is_sensitive_path(path):
            continue
        normalized = path.replace("\\", "/")
        name = Path(normalized).name.lower()
        score = 0

        if normalized in changed:
            score += 1000
        if normalized.startswith(".github/workflows/"):
            score += 700
        if name in metadata:
            score += 600
        if str(Path(normalized).parent) in changed_parents:
            score += 250
        if name.startswith("test_") or name.endswith("_test.py") or ".test." in name or ".spec." in name or name.endswith("_test.go"):
            score += 450
        if any(stem and stem in name for stem in changed_stems):
            score += 180

        suffix = Path(normalized).suffix.lower()
        if suffix in {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".java", ".go", ".rs", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"}:
            score += 80

        if score:
            scored.append((score, normalized))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [path for _, path in scored]


@app.post("/api/inspect")
async def inspect(req: InspectRequest):
    t_start = time.perf_counter()
    try:
        ref = parse_pr_url(req.pr_url)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    gh = GitHubClient(settings.github_token, settings.github_timeout_seconds)
    try:
        try:
            pr = await gh.pull_request(ref)
            target_sha = (
                req.base_sha.strip()
                if req.base_sha and req.base_sha.strip()
                else pr["head"]["sha"]
            )
            files, checks_payload = await asyncio.gather(
                gh.files(ref),
                gh.checks(ref, target_sha),
            )
        except Exception as exc:
            raise HTTPException(502, f"GitHub inspection failed: {str(exc)[:500]}") from exc

        check_runs = checks_payload.get("check_runs", [])

        # Tree retrieval is useful but should not make a valid PR
        # uninspectable when the tree endpoint is temporarily unavailable.
        try:
            tree_objects = await gh.tree(ref, target_sha)
        except Exception:
            tree_objects = []

        changed_paths = [
            f["filename"]
            for f in files
            if f.get("status") != "removed" and f.get("filename")
        ]

        tree_paths = [
            item.get("path", "")
            for item in tree_objects
            if item.get("type") == "blob" and item.get("path")
        ] or changed_paths[:]

        # Failed-check annotations are independent requests; fetch them with
        # a small concurrency bound rather than serially.
        failed_checks = [
            check for check in check_runs
            if check.get("conclusion")
            in {"failure", "cancelled", "timed_out", "action_required"}
            and check.get("id")
        ]
        semaphore = asyncio.Semaphore(max(1, min(settings.github_concurrency, 8)))

        async def annotate(check: dict) -> None:
            async with semaphore:
                try:
                    check["annotations"] = await gh.check_annotations(
                        ref, int(check["id"])
                    )
                except Exception:
                    check["annotations"] = []

        await asyncio.gather(*(annotate(check) for check in failed_checks))

        tree_set = set(tree_paths)
        ranked = _score_repository_paths(tree_paths, changed_paths)
        candidates: list[str] = []

        def add(path: str) -> None:
            path = normalize_repo_path(path)
            if not _safe_path(path) or path in candidates:
                return
            if len(candidates) < settings.max_repo_candidates:
                candidates.append(path)

        # Changed files are always highest-value evidence.
        for path in changed_paths:
            add(path)

        contents = await _fetch_contents(gh, ref, candidates, target_sha)

        for path in ranked:
            add(path)
            if len(candidates) >= settings.max_repo_candidates:
                break

        # Resolve local imports using the actual repository tree.
        for path in changed_paths:
            source = contents.get(path, "")
            for dependency in _path_candidates_from_import(
                path, source, tree_set
            ):
                add(dependency)

        # Pull likely tests even when they were not changed.
        for path in tree_paths:
            normalized = path.replace("\\", "/")
            name = Path(normalized).name.lower()
            if any(
                Path(changed).stem.lower() in name
                for changed in changed_paths
            ):
                if (
                    name.startswith("test_")
                    or ".test." in name
                    or ".spec." in name
                    or name.endswith("_test.go")
                ):
                    add(normalized)

        missing = [path for path in candidates if path not in contents]
        contents.update(
            await _fetch_contents(gh, ref, missing, target_sha)
        )

        context = build_context(
            pr,
            files,
            check_runs,
            contents,
            tree_paths,
        )

        inspect_sec = round(time.perf_counter() - t_start, 3)
        session_id = secrets.token_urlsafe(12)
        session_data = {
            "ref": ref,
            "pr": pr,
            "files": files,
            "checks": check_runs,
            "contents": contents,
            "context": context,
            "tree_paths": tree_paths,
            "target_sha": target_sha,
            "repairs": [],
            "verifications": [],
            "metrics": {
                "inspect_seconds": inspect_sec,
                "repairs": {},
            },
        }
        _save_session(session_id, session_data)

        return {
            "session_id": session_id,
            "pr": {
                "title": pr.get("title"),
                "number": pr.get("number"),
                "url": pr.get("html_url"),
                "head_sha": target_sha,
            },
            "files": [
                {
                    "path": f.get("filename"),
                    "status": f.get("status"),
                    "additions": f.get("additions", 0),
                    "deletions": f.get("deletions", 0),
                }
                for f in files
            ],
            "checks": [
                {
                    "name": c.get("name"),
                    "status": c.get("status"),
                    "conclusion": c.get("conclusion"),
                }
                for c in check_runs
            ],
            "evidence_files": len(contents),
            "context_chars": len(context),
            "metrics": SESSIONS[session_id]["metrics"],
        }
    finally:
        close = getattr(gh, "aclose", None)
        if close is not None:
            await close()

@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    t_start = time.perf_counter()
    session = session_or_404(req.session_id)
    check_ai_quota_status()
    try:
        diagnosis = AIEngine().diagnose(session["context"])
        session["diagnosis"] = diagnosis
        _save_session(req.session_id, session)
        record_ai_success()
        diag_sec = round(time.perf_counter() - t_start, 3)
        session.setdefault("metrics", {})["diagnosis_seconds"] = diag_sec
        diagnosis["metrics"] = session["metrics"]
        return diagnosis
    except HTTPException:
        raise
    except AIQuotaExhaustedError as exc:
        record_ai_quota_exhausted(reset_ts=exc.reset_timestamp, hint=exc.remedy_hint, message=str(exc))
        raise HTTPException(
            429,
            detail={
                "error": "AI_QUOTA_EXHAUSTED",
                "message": str(exc),
                "reset_timestamp": exc.reset_timestamp,
                "remedy_hint": exc.remedy_hint,
            },
        )
    except Exception as exc:
        logger.exception("AI diagnosis failure")
        record_ai_temporarily_unavailable(str(exc)[:200])
        raise HTTPException(
            502,
            f"AI diagnosis failed safely: {type(exc).__name__}: {str(exc)[:500]}",
        )


@app.post("/api/repair")
def repair(req: RepairRequest):
    t_start = time.perf_counter()
    session = session_or_404(req.session_id)
    if "diagnosis" not in session:
        raise HTTPException(400, "Run diagnosis first")

    check_ai_quota_status()

    feedback = ""
    if req.attempt > 1:
        previous = session.get("verifications", [])
        if previous:
            last_v = previous[-1]
            raw_out = last_v.get("output", "") or last_v.get("reason", "")
            feedback = summarize_verification_failure(
                output=raw_out,
                exit_code=last_v.get("exit_code"),
                command=last_v.get("command", ""),
            )

    try:
        try:
            result = AIEngine().generate_patch(
                session["context"],
                session["diagnosis"],
                session["contents"],
                feedback,
                previous_repairs=session.get("repairs", []),
            )
        except TypeError as t_err:
            if "previous_repairs" in str(t_err):
                result = AIEngine().generate_patch(
                    session["context"],
                    session["diagnosis"],
                    session["contents"],
                    feedback,
                )
            else:
                raise
        result["attempt"] = req.attempt
        session.setdefault("repairs", []).append(result)
        _save_session(req.session_id, session)
        record_ai_success()
        rep_sec = round(time.perf_counter() - t_start, 3)
        session.setdefault("metrics", {}).setdefault("repairs", {}).setdefault(str(req.attempt), {})["generate_seconds"] = rep_sec
        result["metrics"] = session["metrics"]
        return result
    except HTTPException:
        raise
    except AIQuotaExhaustedError as exc:
        record_ai_quota_exhausted(reset_ts=exc.reset_timestamp, hint=exc.remedy_hint, message=str(exc))
        raise HTTPException(
            429,
            detail={
                "error": "AI_QUOTA_EXHAUSTED",
                "message": str(exc),
                "reset_timestamp": exc.reset_timestamp,
                "remedy_hint": exc.remedy_hint,
            },
        )
    except Exception as exc:
        logger.exception("AI repair generation failure")
        record_ai_temporarily_unavailable(str(exc)[:200])
        raise HTTPException(
            502,
            f"Repair generation failed safely: {type(exc).__name__}: {str(exc)[:500]}",
        )


@app.post("/api/verify")
def verify(req: RepairRequest):
    t_start = time.perf_counter()
    session = session_or_404(req.session_id)
    if req.attempt > settings.max_repair_attempts:
        raise HTTPException(
            400,
            f"Repair attempt {req.attempt} exceeds the configured maximum of "
            f"{settings.max_repair_attempts}.",
        )

    repairs = session.get("repairs", [])
    candidates = [
        r for r in repairs
        if int(r.get("attempt", 1)) == req.attempt
    ]
    repair = candidates[-1] if candidates else None

    if not repair:
        raise HTTPException(
            400,
            f"Generate repair attempt {req.attempt} before verification.",
        )

    patch = repair.get("patch", "")
    if not patch:
        result = {
            "passed": False,
            "stage": "repair",
            "reason": "The AI refused to produce a safe patch.",
            "attempt": req.attempt,
            "metrics": session.get("metrics", {}),
        }
        session.setdefault("verifications", []).append(result)
        return result

    try:
        validate_patch_paths(patch)
    except RuntimeError:
        raise HTTPException(400, "Repair patch rejected by the safety policy")

    # Early exit if no sandbox is available — avoid cloning untrusted code
    verifier = get_verifier()
    if isinstance(verifier, UnavailableVerifier):
        result = {
            "passed": False,
            "stage": "sandbox",
            "status": "SANDBOX_UNAVAILABLE",
            "verifier_type": "none",
            "category": "ENVIRONMENT_FAILURE",
            "output": verifier.reason,
            "attempt": req.attempt,
            "metrics": session.get("metrics", {}),
        }
        session.setdefault("verifications", []).append(result)
        session["verification"] = result
        _save_session(req.session_id, session)
        return result

    ref: RepoRef = session["ref"]
    session_dir = _workspace_root() / req.session_id

    try:
        target_sha = (
            session.get("target_sha")
            or session["pr"].get("base", {}).get("sha")
            or session["pr"]["head"]["sha"]
        )
        pr_num = session["pr"].get("number") if isinstance(session.get("pr"), dict) else None
        root = prepare_attempt_workspace(
            f"https://github.com/{ref.owner}/{ref.repo}.git",
            target_sha,
            session_dir,
            req.attempt,
            checkout_fn=checkout,
            pr_number=pr_num,
        )

        patch_file = root.parent / f"repair-{req.attempt}.patch"
        patch_file.write_text(patch, encoding="utf-8")
        applied, patch_output = apply_patch(root, patch_file)

        if not applied:
            v_sec = round(time.perf_counter() - t_start, 3)
            session.setdefault("metrics", {}).setdefault("repairs", {}).setdefault(str(req.attempt), {})["verify_seconds"] = v_sec
            result = {
                "passed": False,
                "stage": "patch",
                "output": patch_output,
                "category": "PATCH_FAILURE",
                "attempt": req.attempt,
                "metrics": session["metrics"],
            }
            session.setdefault("verifications", []).append(result)
            return result

        language, default_command = detect_project(root)

        # Directive 14: Target the relevant regression test first before broader tests
        test_command = default_command
        test_files: list[str] = []
        for f in session.get("files", []):
            fn = str(f.get("filename", "")).replace("\\", "/")
            if fn.endswith((".py", ".js", ".ts", ".go", ".rs", ".java")) and ("test" in fn.lower() or "spec" in fn.lower()):
                test_files.append(fn)

        if not test_files and session.get("diagnosis"):
            for aff in session["diagnosis"].get("affected_files", []):
                aff_norm = str(aff).replace("\\", "/")
                if "test" in aff_norm.lower() or "spec" in aff_norm.lower():
                    test_files.append(aff_norm)

        if language == "python" and test_files:
            target_test = test_files[0]
            if (root / target_test).exists():
                test_command = f"python -m pytest -q -p no:cacheprovider -W default {target_test}"

        try:
            profile = detect_project_profile(root)
        except Exception:
            profile = ProjectProfile(
                language=language or "python",
                framework="pytest" if language == "python" else "generic",
                package_manager="pip" if language == "python" else "generic",
                test_command=test_command,
                docker_image="python:3.11-slim" if language == "python" else "ubuntu:22.04",
                confidence=0.5,
                evidence=["inferred"],
            )
        verifier = get_verifier()
        verif_res: VerificationResult = verifier.verify(
            root=root,
            profile=profile,
            command=test_command,
            patch=patch,
            files=session.get("contents"),
        )
        v_sec = round(time.perf_counter() - t_start, 3)
        session.setdefault("metrics", {}).setdefault("repairs", {}).setdefault(str(req.attempt), {})["verify_seconds"] = v_sec
        result = {
            "passed": verif_res.passed,
            "stage": "sandbox",
            "status": verif_res.status,
            "language": profile.language,
            "framework": profile.framework,
            "package_manager": profile.package_manager,
            "detected_test_command": profile.test_command,
            "command": verif_res.command or test_command,
            "exit_code": verif_res.exit_code,
            "output": verif_res.output,
            "category": verif_res.category,
            "verifier_type": verif_res.verifier_type,
            "detection_evidence": list(profile.evidence),
            "attempt": req.attempt,
            "root": str(root),

            "metrics": session["metrics"],
        }
        session.setdefault("verifications", []).append(result)
        session["verification"] = result

        if verif_res.passed:
            session["verified_patch"] = patch

        _save_session(req.session_id, session)
        return result
    except Exception:
        logger.exception("Unexpected HEALFORGE verification failure")
        raise HTTPException(500, "HEALFORGE verification error. Check the server logs for safe diagnostics.")


@app.get("/api/session/{session_id}")
def session_state(session_id: str):
    session = session_or_404(session_id)
    return {
        k: v
        for k, v in session.items()
        if k not in {"context", "contents", "tree_paths"}
    }


@app.get("/api/session/{session_id}/patch")
def patch_download(session_id: str):
    session = session_or_404(session_id)
    patch = session.get("verified_patch")
    if not patch:
        repairs = session.get("repairs") or []
        patch = repairs[-1].get("patch", "") if repairs else ""

    if not patch:
        raise HTTPException(404, "No patch available")

    path = _workspace_root() / f"{session_id}.patch"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(patch, encoding="utf-8")
    return FileResponse(path, media_type="text/plain", filename="healforge.patch")


@app.post("/api/publish")
async def publish(req: PublishRequest):
    session = session_or_404(req.session_id)
    if not settings.allow_write_actions:
        raise HTTPException(
            403,
            "Write actions are disabled. Set ALLOW_WRITE_ACTIONS=true only when you intentionally want HEALFORGE to create a branch and PR.",
        )

    verification = session.get("verification", {})
    if not verification.get("passed"):
        raise HTTPException(400, "Only a verified repair can be published")

    patch = session.get("verified_patch", "")
    if not patch:
        raise HTTPException(400, "No verified patch")

    paths = re.findall(r"^\+\+\+\s+b/(.+)$", patch, re.MULTILINE)
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise HTTPException(400, "Could not identify patched files")
    if any(not _safe_path(path) for path in paths):
        raise HTTPException(400, "Patch contains a sensitive or unsafe file path")

    gh = GitHubClient(settings.github_token, settings.github_timeout_seconds)
    ref: RepoRef = session["ref"]
    branch = f"healforge/fix-{session['pr']['number']}-{secrets.token_hex(3)}"
    await gh.create_branch(ref, session["pr"]["head"]["sha"], branch)

    for path in paths:
        root = Path(verification["root"]) / path
        if not root.is_file():
            raise HTTPException(400, f"Patched file is not a regular text file: {path}")

        data = await gh._get(
            f"/repos/{ref.owner}/{ref.repo}/contents/{path}",
            {"ref": session["pr"]["head"]["sha"]},
        )
        await gh.update_file(
            ref,
            path,
            f"HEALFORGE: repair {path}",
            root.read_text(encoding="utf-8"),
            data["sha"],
            branch,
        )

    created = await gh.create_pr(
        ref,
        branch,
        session["pr"]["base"]["ref"],
        req.title,
        req.body,
    )
    return {
        "url": created["html_url"],
        "number": created["number"],
        "branch": branch,
    }
