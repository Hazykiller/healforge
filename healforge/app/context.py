import json
import re
from pathlib import Path

from .config import settings
from .security import is_sensitive_path as _is_sensitive_file

IGNORED_PARTS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    "target",
    "workspace",
}

def is_sensitive_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return _is_sensitive_file(normalized) or any(part in IGNORED_PARTS for part in normalized.split("/"))


def _sanitize(content: str) -> str:
    patterns = [
        (r"(?i)(github_pat_[A-Za-z0-9_\-]+)", "[REDACTED_GITHUB_TOKEN]"),
        (r"(?i)(ghp_[A-Za-z0-9]+)", "[REDACTED_GITHUB_TOKEN]"),
        (r"(?i)(sk-or-v1-[A-Za-z0-9_\-]+)", "[REDACTED_OPENROUTER_KEY]"),
        (r"(?i)(-----BEGIN [^-]+ PRIVATE KEY-----).*?(-----END [^-]+ PRIVATE KEY-----)", "[REDACTED_PRIVATE_KEY]"),
    ]

    result = content
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result, flags=re.DOTALL)
    return result


def _priority(path: str, changed: set[str]) -> int:
    normalized = path.replace("\\", "/")
    name = Path(normalized).name.lower()

    if normalized in changed:
        return 120
    if normalized.startswith(".github/workflows/"):
        return 105
    if name in {
        "pyproject.toml", "requirements.txt", "requirements-dev.txt",
        "package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock",
        "tsconfig.json", "pom.xml", "build.gradle", "build.gradle.kts",
        "go.mod", "go.sum", "cargo.toml", "cargo.lock", "cmakelists.txt",
        "makefile", "pytest.ini", "tox.ini",
    }:
        return 100
    if (
        name.startswith("test_")
        or name.endswith("_test.py")
        or ".test." in name
        or ".spec." in name
        or name.endswith("_test.go")
    ):
        return 90
    if Path(normalized).suffix.lower() in {
        ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
        ".java", ".go", ".rs", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp",
    }:
        return 60
    return 30


def build_context(
    pr: dict,
    files: list[dict],
    checks: list[dict],
    repository_files: dict[str, str],
    tree_paths: list[str] | None = None,
) -> str:
    changed_paths = {
        str(item.get("filename", "")).replace("\\", "/")
        for item in files
        if item.get("filename")
    }

    changed = [
        {
            "path": item.get("filename"),
            "status": item.get("status"),
            "additions": item.get("additions", 0),
            "deletions": item.get("deletions", 0),
            "patch": item.get("patch", ""),
        }
        for item in files
    ]

    check_data = []
    for item in checks:
        output = item.get("output") or {}
        check_data.append(
            {
                "name": item.get("name"),
                "status": item.get("status"),
                "conclusion": item.get("conclusion"),
                "summary": output.get("summary", ""),
                "text": output.get("text", ""),
                "annotations": item.get("annotations", []),
            }
        )

    parts = [
        "=== HEALFORGE DIAGNOSTIC EVIDENCE ===",
        "=== PULL REQUEST ===\n" + json.dumps(
            {
                "title": pr.get("title"),
                "body": pr.get("body"),
                "head_sha": pr.get("head", {}).get("sha"),
                "base": pr.get("base", {}).get("ref"),
            },
            indent=2,
        ),
        "=== CHANGED FILES / DIFF ===\n" + json.dumps(changed, indent=2),
        "=== CI / CHECKS ===\n" + json.dumps(check_data, indent=2),
    ]

    if tree_paths:
        safe_tree = [
            p for p in tree_paths
            if not is_sensitive_path(p)
        ][:4000]
        parts.append(
            "=== REPOSITORY TREE (ABRIDGED) ===\n"
            + "\n".join(safe_tree)
        )

    parts.append("=== REPOSITORY FILE CONTENT ===")

    ranked = sorted(
        repository_files.items(),
        key=lambda item: _priority(item[0], changed_paths),
        reverse=True,
    )

    current = sum(len(p) for p in parts)
    remaining = max(1000, settings.max_context_chars - current)

    for path, content in ranked:
        normalized = path.replace("\\", "/")
        if is_sensitive_path(normalized):
            continue
        if not isinstance(content, str):
            content = str(content)

        limit = min(settings.max_file_chars, max(500, remaining - 300))
        safe_content = _sanitize(content)
        if len(safe_content) > limit:
            safe_content = safe_content[:limit] + "\n[TRUNCATED]"

        section = (
            f"\n--- FILE: {normalized} ---\n"
            f"{safe_content}\n"
            f"--- END FILE: {normalized} ---"
        )
        if len(section) > remaining:
            break
        parts.append(section)
        remaining -= len(section)

    parts.append(
        """
=== DIAGNOSTIC RULES ===
Use only supplied evidence. Trace the failure through the repository; the
changed line is not automatically the root cause. Prefer failing CI output,
stack traces, tests, changed-file diffs, imported local modules, dependency
manifests, and workflow configuration. If evidence is insufficient, explicitly
say so. Never invent files, dependencies, APIs, test results, CI results, or
repository state.
""".strip()
    )

    return "\n\n".join(parts)[:settings.max_context_chars]
