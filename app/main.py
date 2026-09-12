import json
import secrets
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .config import settings
from .schemas import InspectRequest, AnalyzeRequest, RepairRequest, PublishRequest
from .github import GitHubClient, parse_pr_url, RepoRef
from .context import build_context
from .ai import AIEngine
from .runner import checkout, detect_project, apply_patch, docker_test

app = FastAPI(title="HEALFORGE", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
SESSIONS: dict[str, dict] = {}


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
    return {"ok": True, "github_configured": bool(settings.github_token), "ai_configured": bool(settings.openrouter_api_key)}

@app.post("/api/inspect")
async def inspect(req: InspectRequest):
    try:
        ref = parse_pr_url(req.pr_url)
        gh = GitHubClient(settings.github_token)
        pr = await gh.pull_request(ref)
        files = await gh.files(ref)
        checks_payload = await gh.checks(ref, pr["head"]["sha"])
        changed_paths = [f["filename"] for f in files if f.get("status") != "removed"]
        # Pull changed source plus high-signal project metadata. This keeps the context bounded without inventing repository state.
        candidates = []
        for path in changed_paths:
            if len(candidates) >= 30:
                break
            candidates.append(path)
        metadata_names = {"pyproject.toml", "requirements.txt", "package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "pytest.ini", "tox.ini"}
        for name in metadata_names:
            if name not in candidates:
                candidates.append(name)
        contents = {}
        for path in candidates:
            try:
                contents[path] = await gh.content(ref, path, pr["head"]["sha"])
            except Exception:
                continue
        context = build_context(pr, files, checks_payload.get("check_runs", []), contents)
        session_id = secrets.token_urlsafe(12)
        SESSIONS[session_id] = {"ref": ref, "pr": pr, "files": files, "checks": checks_payload.get("check_runs", []), "contents": contents, "context": context}
        return {"session_id": session_id, "pr": {"title": pr["title"], "number": pr["number"], "url": pr["html_url"], "head_sha": pr["head"]["sha"]}, "files": [{"path": f["filename"], "status": f["status"], "additions": f["additions"], "deletions": f["deletions"]} for f in files], "checks": [{"name": c["name"], "status": c["status"], "conclusion": c["conclusion"]} for c in checks_payload.get("check_runs", [])], "context_chars": len(context)}
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
        raise HTTPException(400, str(exc))

@app.post("/api/repair")
def repair(req: RepairRequest):
    session = session_or_404(req.session_id)

    if "diagnosis" not in session:
        raise HTTPException(400, "Run diagnosis first")

    try:
        patch = AIEngine().generate_patch(
            session["context"],
            session["diagnosis"],
        )

        session.setdefault("repairs", []).append(patch)

        return patch

    except Exception as exc:
        # Keep the error visible to the frontend.
        # This makes model/parser/validation failures diagnosable
        # instead of appearing as a mysterious 400.
        raise HTTPException(
            status_code=422,
            detail=f"Repair generation failed: {type(exc).__name__}: {exc}",
        )

@app.post("/api/verify")
def verify(req: RepairRequest):
    session = session_or_404(req.session_id)
    repairs = session.get("repairs", [])
    if not repairs:
        raise HTTPException(400, "Generate a repair first")
    patch = repairs[-1].get("patch", "")
    if not patch:
        return {"passed": False, "reason": "The repair component did not produce a safe patch."}
    ref: RepoRef = session["ref"]
    root = Path(settings.host_workspace).resolve() / req.session_id / f"attempt-{req.attempt}"
    try:
        checkout(f"https://github.com/{ref.owner}/{ref.repo}.git", session["pr"]["head"]["sha"], root)
        patch_file = root.parent / "repair.patch"
        patch_file.write_text(patch, encoding="utf-8")
        applied, patch_output = apply_patch(root, patch_file)
        if not applied:
            return {"passed": False, "stage": "patch", "output": patch_output}
        language, command = detect_project(root)
        result = docker_test(root, language, command)
        session["verification"] = {"passed": result.passed, "language": language, "command": result.command, "exit_code": result.exit_code, "output": result.output, "root": str(root)}
        if result.passed:
            session["verified_patch"] = patch
        return session["verification"]
    except Exception as exc:
        raise HTTPException(400, str(exc))

@app.get("/api/session/{session_id}")
def session_state(session_id: str):
    session = session_or_404(session_id)
    return {k: v for k, v in session.items() if k not in {"context", "contents"}}

@app.get("/api/session/{session_id}/patch")
def patch_download(session_id: str):
    session = session_or_404(session_id)
    patch = session.get("verified_patch") or (session.get("repairs") or [{}])[-1].get("patch", "")
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
        raise HTTPException(403, "Write actions are disabled. Set ALLOW_WRITE_ACTIONS=true only when you intentionally want HEALFORGE to create a branch and PR.")
    verification = session.get("verification", {})
    if not verification.get("passed"):
        raise HTTPException(400, "Only a verified repair can be published")
    patch = session.get("verified_patch", "")
    if not patch:
        raise HTTPException(400, "No verified patch")
    # Publishing is intentionally limited to text files that GitHub can update through the Contents API.
    import re
    paths = re.findall(r"^\+\+\+ b/(.+)$", patch, re.MULTILINE)
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise HTTPException(400, "Could not identify patched files")
    gh = GitHubClient(settings.github_token)
    ref: RepoRef = session["ref"]
    branch = f"healforge/fix-{session['pr']['number']}-{secrets.token_hex(3)}"
    await gh.create_branch(ref, session["pr"]["head"]["sha"], branch)
    for path in paths:
        root = Path(verification["root"]) / path
        if not root.is_file():
            raise HTTPException(400, f"Patched file is not a regular text file: {path}")
        # Find current blob SHA from the PR head using the API.
        data = await gh._get(f"/repos/{ref.owner}/{ref.repo}/contents/{path}", {"ref": session["pr"]["head"]["sha"]})
        await gh.update_file(ref, path, f"HEALFORGE: repair {path}", root.read_text(encoding="utf-8"), data["sha"], branch)
    created = await gh.create_pr(ref, branch, session["pr"]["base"]["ref"], req.title, req.body)
    return {"url": created["html_url"], "number": created["number"], "branch": branch}
