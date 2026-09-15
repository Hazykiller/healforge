import logging
import re
import secrets
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .ai import AIEngine
from .config import settings
from .context import build_context, is_sensitive_path
from .security import is_safe_repo_path
from .github import GitHubClient, RepoRef, parse_pr_url
from .runner import apply_patch, checkout, detect_project, docker_test, validate_patch_paths
from .schemas import AnalyzeRequest, InspectRequest, PublishRequest, RepairRequest


app = FastAPI(title="HEALFORGE", version="2.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
SESSIONS: dict[str, dict] = {}
logger = logging.getLogger(__name__)


def session_or_404(session_id: str) -> dict:
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "github_configured": bool(settings.github_token),
        "ai_configured": bool(settings.openrouter_api_key),
        "ai_models": settings.ai_models if settings.openrouter_api_key else [],
        "max_repair_attempts": settings.max_repair_attempts,
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
    try:
        ref = parse_pr_url(req.pr_url)
        gh = GitHubClient(settings.github_token, settings.github_timeout_seconds)
        pr = await gh.pull_request(ref)
        head_sha = pr["head"]["sha"]
        files = await gh.files(ref)
        checks_payload = await gh.checks(ref, head_sha)
        check_runs = checks_payload.get("check_runs", [])

        # Add concrete annotations from failed checks when GitHub exposes them.
        for check in check_runs:
            if check.get("conclusion") not in {"failure", "cancelled", "timed_out", "action_required"}:
                continue
            run_id = check.get("id")
            if not run_id:
                continue
            try:
                check["annotations"] = await gh.check_annotations(ref, int(run_id))
            except Exception:
                check["annotations"] = []

        changed_paths = [
            f["filename"] for f in files
            if f.get("status") != "removed" and f.get("filename")
        ]

        try:
            tree_objects = await gh.tree(ref, head_sha)
            tree_paths = [
                item.get("path", "")
                for item in tree_objects
                if item.get("type") == "blob"
            ]
        except Exception:
            tree_paths = changed_paths[:]

        tree_set = set(tree_paths)
        ranked = _score_repository_paths(tree_paths, changed_paths)
        candidates: list[str] = []

        def add(path: str) -> None:
            path = path.replace("\\", "/").lstrip("./")
            if not _safe_path(path) or path in candidates:
                return
            if len(candidates) < settings.max_repo_candidates:
                candidates.append(path)

        # Always include changed files first.
        for path in changed_paths:
            add(path)

        # Fetch changed files first so imports can be resolved.
        contents: dict[str, str] = {}
        for path in candidates:
            try:
                contents[path] = await gh.content(ref, path, head_sha)
            except Exception:
                continue

        # Add intelligently ranked repository evidence.
        for path in ranked:
            add(path)
            if len(candidates) >= settings.max_repo_candidates:
                break

        # Resolve local imports from changed files against the actual tree.
        for path in changed_paths:
            source = contents.get(path, "")
            for dependency in _path_candidates_from_import(path, source, tree_set):
                add(dependency)

        # Pull likely tests even when they were not changed.
        for path in tree_paths:
            normalized = path.replace("\\", "/")
            name = Path(normalized).name.lower()
            if any(Path(changed).stem.lower() in name for changed in changed_paths):
                if name.startswith("test_") or ".test." in name or ".spec." in name or name.endswith("_test.go"):
                    add(normalized)

        # Fetch everything newly selected.
        for path in candidates:
            if path in contents:
                continue
            try:
                contents[path] = await gh.content(ref, path, head_sha)
            except Exception:
                continue

        context = build_context(
            pr,
            files,
            check_runs,
            contents,
            tree_paths,
        )

        session_id = secrets.token_urlsafe(12)
        SESSIONS[session_id] = {
            "ref": ref,
            "pr": pr,
            "files": files,
            "checks": check_runs,
            "contents": contents,
            "context": context,
            "tree_paths": tree_paths,
            "repairs": [],
            "verifications": [],
        }

        return {
            "session_id": session_id,
            "pr": {
                "title": pr.get("title"),
                "number": pr.get("number"),
                "url": pr.get("html_url"),
                "head_sha": head_sha,
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
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    session = session_or_404(req.session_id)
    try:
        diagnosis = AIEngine().diagnose(session["context"])
        session["diagnosis"] = diagnosis
        return diagnosis
    except Exception as exc:
        raise HTTPException(422, str(exc))


@app.post("/api/repair")
def repair(req: RepairRequest):
    session = session_or_404(req.session_id)
    if "diagnosis" not in session:
        raise HTTPException(400, "Run diagnosis first")

    feedback = ""
    if req.attempt > 1:
        previous = session.get("verifications", [])
        if previous:
            feedback = previous[-1].get("output", "") or previous[-1].get("reason", "")

    try:
        result = AIEngine().generate_patch(
            session["context"],
            session["diagnosis"],
            feedback,
        )
        result["attempt"] = req.attempt
        session.setdefault("repairs", []).append(result)
        return result
    except Exception as exc:
        raise HTTPException(
            422,
            f"Repair generation failed: {type(exc).__name__}: {exc}",
        )


@app.post("/api/verify")
def verify(req: RepairRequest):
    session = session_or_404(req.session_id)
    repairs = session.get("repairs", [])
    candidates = [r for r in repairs if int(r.get("attempt", 1)) == req.attempt]
    if not candidates:
        raise HTTPException(400, f"No repair exists for attempt {req.attempt}. Generate that repair first.")
    repair = candidates[-1]

    patch = repair.get("patch", "")
    if not patch:
        result = {
            "passed": False,
            "stage": "repair",
            "reason": "The AI refused to produce a safe patch.",
            "attempt": req.attempt,
        }
        session.setdefault("verifications", []).append(result)
        return result

    try:
        validate_patch_paths(patch)
    except RuntimeError:
        raise HTTPException(400, "Repair patch rejected by the safety policy")

    ref: RepoRef = session["ref"]
    root = Path(settings.host_workspace).resolve() / req.session_id / f"attempt-{req.attempt}"

    try:
        checkout(
            f"https://github.com/{ref.owner}/{ref.repo}.git",
            session["pr"]["head"]["sha"],
            root,
            ref.number,
        )

        patch_file = root.parent / f"repair-{req.attempt}.patch"
        patch_file.write_text(patch, encoding="utf-8")
        applied, patch_output = apply_patch(root, patch_file)

        if not applied:
            result = {
                "passed": False,
                "stage": "patch",
                "output": patch_output,
                "attempt": req.attempt,
            }
            session.setdefault("verifications", []).append(result)
            return result

        language, command = detect_project(root)
        run = docker_test(root, language, command)
        result = {
            "passed": run.passed,
            "stage": "sandbox",
            "language": language,
            "command": run.command,
            "exit_code": run.exit_code,
            "output": run.output,
            "attempt": req.attempt,
            "root": str(root),
        }
        session.setdefault("verifications", []).append(result)
        session["verification"] = result

        if run.passed:
            session["verified_patch"] = patch

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

    path = Path(settings.host_workspace).resolve() / f"{session_id}.patch"
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
