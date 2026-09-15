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
    is_test = (
        name.startswith("test_")
        or name.endswith("_test.py")
        or ".test." in name
        or ".spec." in name
        or name.endswith("_test.go")
        or "/tests/" in f"/{normalized}/"
        or "/test/" in f"/{normalized}/"
    )

    changed_stems = {
        Path(p).stem.lower()
        for p in changed
        if Path(p).stem and Path(p).stem.lower() not in {"__init__", "index"}
    }

    # 1. Changed implementation files (highest priority)
    if normalized in changed and not is_test:
        return 150

    # 2. Files directly related / referenced by changed code
    if not is_test and any(stem in normalized.lower() for stem in changed_stems):
        return 120

    # 3. Relevant tests (matched to changed files or explicitly modified in PR)
    if is_test and (normalized in changed or any(stem in name for stem in changed_stems)):
        return 85

    # 4. Configuration and build files needed for understanding dependencies
    if name in {
        "pyproject.toml", "requirements.txt", "requirements-dev.txt",
        "package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock",
        "tsconfig.json", "pom.xml", "build.gradle", "build.gradle.kts",
        "go.mod", "go.sum", "cargo.toml", "cargo.lock", "cmakelists.txt",
        "makefile", "pytest.ini", "tox.ini",
    }:
        return 90

    if normalized.startswith(".github/workflows/"):
        return 80

    # 5. Other implementation dependencies in the codebase
    if Path(normalized).suffix.lower() in {
        ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
        ".java", ".go", ".rs", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp",
    }:
        return 65 if not is_test else 35

    return 20


def _extract_targeted_content(content: str, max_chars: int, hints: set[str]) -> str:
    """
    Extract relevant sections of large files instead of blindly cutting mid-construct.
    Preserves windows around changed symbols/hints with priority scoring so function
    definitions and key logic are never truncated.
    """
    if len(content) <= max_chars:
        return content

    lines = content.splitlines()
    if not lines:
        return content[:max_chars]

    clean_hints = {h.strip("`'\",():;.") for h in hints if len(h.strip("`'\",():;.")) >= 3}
    candidates: list[tuple[int, int]] = []
    for idx, line in enumerate(lines):
        line_strip = line.strip()
        score = 0
        for h in clean_hints:
            h_lower = h.lower()
            if h_lower in line.lower():
                score += 5
                if h in line:
                    score += 10
                if line_strip.startswith(f"def {h}") or line_strip.startswith(f"class {h}"):
                    score += 100
                elif f"def {h}" in line or f"class {h}" in line:
                    score += 80
                elif f"{h}(" in line or f"{h} =" in line:
                    score += 30
        if score > 0:
            candidates.append((score, idx))

    selected_lines: set[int] = set()
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        total_len = 0
        for score, idx in candidates:
            line_str = lines[idx].strip()
            if line_str.startswith("def ") or line_str.startswith("class "):
                start = idx
                end = min(len(lines), idx + 25)
            else:
                start = max(0, idx - 4)
                end = min(len(lines), idx + 16)
            window = set(range(start, end))
            new_lines = window - selected_lines
            added_len = sum(len(lines[i]) + 1 for i in new_lines)
            if selected_lines and (total_len + added_len > max_chars):
                continue
            selected_lines.update(window)
            total_len += added_len
            if total_len >= max_chars:
                break
    else:
        selected_lines = set(range(min(40, len(lines))))

    sorted_indices = sorted(selected_lines)
    output_lines = []
    curr_len = 0
    prev_idx = -1

    for idx in sorted_indices:
        line_text = lines[idx]
        snip = f"... [snip: {idx - prev_idx - 1} lines] ..." if (prev_idx != -1 and idx > prev_idx + 1) else None
        cost = len(line_text) + 1 + (len(snip) + 1 if snip else 0)
        if curr_len + cost > max_chars and output_lines:
            break
        if snip:
            output_lines.append(snip)
        output_lines.append(line_text)
        curr_len += cost
        prev_idx = idx

    return "\n".join(output_lines)


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

    # Extract hints from changed files, diffs, and PR metadata
    hints: set[str] = set()
    title_words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", str(pr.get("title", "")).lower())
    hints.update(title_words)
    for item in files:
        fn = Path(str(item.get("filename", ""))).stem.lower()
        if fn and fn not in {"__init__", "index"}:
            hints.add(fn)
        patch = str(item.get("patch", ""))
        for line in patch.splitlines()[:50]:
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", line.lower())
                hints.update(words)

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
        ][:40]
        tree_text = "\n".join(safe_tree)
        if len(tree_text) > 1000:
            tree_text = tree_text[:1000] + "\n[TRUNCATED_TREE]"
        parts.append(
            "=== REPOSITORY TREE (ABRIDGED) ===\n"
            + tree_text
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

        prio = _priority(normalized, changed_paths)
        # Bounded relevance: do not include low-priority unrelated repository files
        # if sufficient relevant context (>= 4000 chars) has already been extracted.
        if prio < 60 and (settings.max_context_chars - remaining) >= 4000:
            continue

        if not isinstance(content, str):
            content = str(content)

        limit = min(settings.max_file_chars, max(500, remaining - 300))
        targeted = _extract_targeted_content(content, limit, hints)
        safe_content = _sanitize(targeted)
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
