import json
from pathlib import Path
from .config import settings

IGNORED = {".git", "node_modules", ".venv", "venv", "dist", "build", "__pycache__"}


def build_context(pr: dict, files: list[dict], checks: list[dict], repository_files: dict[str, str]) -> str:
    changed = []
    for item in files:
        changed.append({
            "path": item.get("filename"),
            "status": item.get("status"),
            "additions": item.get("additions", 0),
            "deletions": item.get("deletions", 0),
            "patch": item.get("patch", ""),
        })
    check_data = [{"name": c.get("name"), "status": c.get("status"), "conclusion": c.get("conclusion"), "output": c.get("output", {}).get("summary", "")} for c in checks]
    parts = [
        "PULL REQUEST:\n" + json.dumps({"title": pr.get("title"), "body": pr.get("body"), "head_sha": pr.get("head", {}).get("sha"), "base": pr.get("base", {}).get("ref")}, indent=2),
        "CHANGED FILES:\n" + json.dumps(changed, indent=2),
        "CI CHECKS:\n" + json.dumps(check_data, indent=2),
    ]
    for path, content in repository_files.items():
        parts.append(f"FILE: {path}\n```\n{content}\n```")
        if sum(len(p) for p in parts) >= settings.max_context_chars:
            break
    return "\n\n".join(parts)[:settings.max_context_chars]
