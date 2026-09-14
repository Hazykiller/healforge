from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import settings
from .security import is_sensitive_path, is_safe_repo_path


# ============================================================================
# RESULT / PROJECT MODELS
# ============================================================================

@dataclass
class RunResult:
    passed: bool
    command: str
    exit_code: int
    output: str


@dataclass
class ProjectProfile:
    language: str
    framework: str
    package_manager: str
    test_command: str
    docker_image: str
    confidence: float
    evidence: list[str] = field(default_factory=list)


# ============================================================================
# CONSTANTS
# ============================================================================

IGNORED = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".idea",
    ".vscode",
    "workspace",
    "target",
    "build",
    "dist",
    "coverage",
    ".gradle",
}

MAX_SOURCE_FILES = 5000
MAX_OUTPUT_CHARS = 40000


# ============================================================================
# PROCESS EXECUTION
# ============================================================================

def _decode_output(data: bytes | str | None) -> str:
    """
    Decode subprocess output safely on Windows and Linux.

    Docker/Git may emit UTF-8 bytes while Windows PowerShell uses a
    different console encoding. Never let decoding crash verification.
    """
    if data is None:
        return ""

    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")

    return str(data)


def run_process(
    args: list[str],
    cwd: Path | None = None,
    timeout: int = 120,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """
    Run a process and return (exit_code, output).

    IMPORTANT:
    cwd and timeout intentionally remain positional-compatible because
    existing HEALFORGE code calls:

        run_process(command, cwd, timeout)

    This prevents the interface regression that previously broke verify.
    """
    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
            text=False,
        )

        output = _decode_output(completed.stdout)

        return completed.returncode, output[-MAX_OUTPUT_CHARS:]

    except subprocess.TimeoutExpired as exc:
        output = _decode_output(exc.stdout)

        return (
            124,
            output[-MAX_OUTPUT_CHARS:]
            + f"\n\nProcess timed out after {timeout} seconds.",
        )

    except OSError as exc:
        return 127, f"Failed to start process: {exc}"


# ============================================================================
# DOCKER
# ============================================================================

def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False

    code, _ = run_process(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        Path.cwd(),
        15,
    )

    return code == 0


# ============================================================================
# FILE DISCOVERY
# ============================================================================

def _read_text(
    root: Path,
    name: str,
    limit: int = 30000,
) -> str:
    path = root / name

    if not path.is_file():
        return ""

    try:
        return path.read_text(
            encoding="utf-8",
            errors="replace",
        )[:limit]
    except OSError:
        return ""


def _source_files(root: Path) -> list[Path]:
    result: list[Path] = []

    try:
        for path in root.rglob("*"):
            if not path.is_file():
                continue

            try:
                relative = path.relative_to(root)
            except ValueError:
                continue

            if any(
                part in IGNORED
                for part in relative.parts
            ):
                continue

            if is_sensitive_path(relative.as_posix()):
                continue

            result.append(path)

            if len(result) >= MAX_SOURCE_FILES:
                break

    except OSError:
        pass

    return result


def _has(root: Path, *names: str) -> bool:
    return any(
        (root / name).exists()
        for name in names
    )


# ============================================================================
# PYTHON DETECTION
# ============================================================================

def _detect_python(root: Path) -> ProjectProfile | None:
    files = _source_files(root)

    python_files = [
        path
        for path in files
        if path.suffix == ".py"
    ]

    markers = [
        name
        for name in (
            "pyproject.toml",
            "requirements.txt",
            "requirements-dev.txt",
            "setup.py",
            "setup.cfg",
            "Pipfile",
            "poetry.lock",
            "uv.lock",
        )
        if (root / name).exists()
    ]

    if not python_files and not markers:
        return None

    pyproject = _read_text(
        root,
        "pyproject.toml",
    ).lower()

    requirements = (
        _read_text(root, "requirements.txt")
        + "\n"
        + _read_text(root, "requirements-dev.txt")
    ).lower()

    combined = pyproject + "\n" + requirements

    if "django" in combined:
        framework = "Django"
    elif "fastapi" in combined:
        framework = "FastAPI"
    elif "flask" in combined:
        framework = "Flask"
    elif (
        "pytest" in combined
        or _has(root, "pytest.ini", "tox.ini")
    ):
        framework = "pytest"
    else:
        framework = "Python"

    has_pytest = (
        "pytest" in combined
        or _has(root, "pytest.ini", "tox.ini")
        or (root / "tests").is_dir()
        or any(
            path.name.startswith("test_")
            for path in python_files
        )
    )

    if has_pytest:
        test_command = "python -m pytest -q -p no:cacheprovider"
    else:
        test_command = "python -m unittest discover -v"

    if (root / "uv.lock").exists():
        package_manager = "uv"
    elif (root / "poetry.lock").exists():
        package_manager = "poetry"
    elif (root / "Pipfile").exists():
        package_manager = "pipenv"
    else:
        package_manager = "pip"

    evidence = [
        f"Python files: {len(python_files)}",
        *markers,
    ]

    return ProjectProfile(
        language="python",
        framework=framework,
        package_manager=package_manager,
        test_command=test_command,
        docker_image=settings.docker_image_python,
        confidence=0.99,
        evidence=evidence,
    )


# ============================================================================
# NODE / TYPESCRIPT DETECTION
# ============================================================================

def _detect_node(root: Path) -> ProjectProfile | None:
    package_file = root / "package.json"

    files = _source_files(root)

    js_files = [
        path
        for path in files
        if path.suffix.lower()
        in {
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".mjs",
            ".cjs",
        }
    ]

    if not package_file.is_file() and not js_files:
        return None

    raw = _read_text(
        root,
        "package.json",
    )

    try:
        package = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        package = {}

    dependencies = {
        **package.get("dependencies", {}),
        **package.get("devDependencies", {}),
    }

    combined = json.dumps(
        dependencies
    ).lower()

    if "next" in combined:
        framework = "Next.js"
    elif "react" in combined:
        framework = "React"
    elif "vue" in combined:
        framework = "Vue"
    elif "express" in combined:
        framework = "Express"
    elif "nestjs" in combined or "@nestjs" in combined:
        framework = "NestJS"
    elif "vite" in combined:
        framework = "Vite"
    else:
        framework = "Node.js"

    if (root / "pnpm-lock.yaml").exists():
        package_manager = "pnpm"
    elif (root / "yarn.lock").exists():
        package_manager = "yarn"
    else:
        package_manager = "npm"

    scripts = package.get("scripts", {})
    test_script = scripts.get("test")

    if test_script:
        test_command = f"{package_manager} test"
    elif "vitest" in combined:
        test_command = f"{package_manager} exec vitest run"
    elif "jest" in combined:
        test_command = f"{package_manager} exec jest --runInBand"
    else:
        test_command = f"{package_manager} test"

    evidence = [
        f"JS/TS files: {len(js_files)}",
        *[
            name
            for name in (
                "package.json",
                "package-lock.json",
                "pnpm-lock.yaml",
                "yarn.lock",
            )
            if (root / name).exists()
        ],
    ]

    return ProjectProfile(
        language="node",
        framework=framework,
        package_manager=package_manager,
        test_command=test_command,
        docker_image=settings.docker_image_node,
        confidence=0.99,
        evidence=evidence,
    )


# ============================================================================
# JAVA DETECTION
# ============================================================================

def _detect_java(root: Path) -> ProjectProfile | None:
    files = _source_files(root)

    java_files = [
        path
        for path in files
        if path.suffix == ".java"
    ]

    if not java_files and not _has(
        root,
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "gradlew",
    ):
        return None

    pom = _read_text(root, "pom.xml").lower()

    gradle = (
        _read_text(root, "build.gradle").lower()
        + "\n"
        + _read_text(root, "build.gradle.kts").lower()
    )

    combined = pom + "\n" + gradle

    framework = (
        "Spring Boot"
        if "spring-boot" in combined
        or "springframework" in combined
        else "Java"
    )

    if (root / "pom.xml").exists():
        package_manager = "maven"

        test_command = "mvn test -q"

        docker_image = "maven:3.9-eclipse-temurin-21"

    elif (root / "gradlew").is_file():
        package_manager = "gradle-wrapper"

        test_command = "./gradlew test --no-daemon"

        docker_image = "gradle:8.10-jdk21"

    else:
        package_manager = "gradle"

        test_command = "gradle test"

        docker_image = "gradle:8.10-jdk21"

    return ProjectProfile(
        language="java",
        framework=framework,
        package_manager=package_manager,
        test_command=test_command,
        docker_image=docker_image,
        confidence=0.98,
        evidence=[
            f"Java files: {len(java_files)}",
            *[
                name
                for name in (
                    "pom.xml",
                    "build.gradle",
                    "build.gradle.kts",
                    "gradlew",
                )
                if (root / name).exists()
            ],
        ],
    )


# ============================================================================
# GO DETECTION
# ============================================================================

def _detect_go(root: Path) -> ProjectProfile | None:
    files = _source_files(root)

    go_files = [
        path
        for path in files
        if path.suffix == ".go"
    ]

    if not go_files and not (root / "go.mod").exists():
        return None

    return ProjectProfile(
        language="go",
        framework="Go",
        package_manager="go-modules",
        test_command="go test ./...",
        docker_image="golang:1.25-bookworm",
        confidence=0.99,
        evidence=[
            f"Go files: {len(go_files)}",
            "go.mod" if (root / "go.mod").exists() else "",
        ],
    )


# ============================================================================
# RUST DETECTION
# ============================================================================

def _detect_rust(root: Path) -> ProjectProfile | None:
    files = _source_files(root)

    rust_files = [
        path
        for path in files
        if path.suffix == ".rs"
    ]

    if not rust_files and not (root / "Cargo.toml").exists():
        return None

    return ProjectProfile(
        language="rust",
        framework="Cargo",
        package_manager="cargo",
        test_command="cargo test",
        docker_image="rust:1-bookworm",
        confidence=0.99,
        evidence=[
            f"Rust files: {len(rust_files)}",
            "Cargo.toml",
        ],
    )


# ============================================================================
# C / C++ DETECTION
# ============================================================================

def _detect_cpp(root: Path) -> ProjectProfile | None:
    files = _source_files(root)

    cpp_files = [
        path
        for path in files
        if path.suffix.lower()
        in {
            ".c",
            ".cc",
            ".cpp",
            ".cxx",
            ".h",
            ".hpp",
        }
    ]

    if not cpp_files and not _has(
        root,
        "CMakeLists.txt",
        "Makefile",
    ):
        return None

    if (root / "CMakeLists.txt").exists():
        return ProjectProfile(
            language="cpp",
            framework="CMake",
            package_manager="cmake",
            test_command="ctest --test-dir build --output-on-failure",
            docker_image="ubuntu:24.04",
            confidence=0.96,
            evidence=[
                f"C/C++ files: {len(cpp_files)}",
                "CMakeLists.txt",
            ],
        )

    return ProjectProfile(
        language="cpp",
        framework="Make",
        package_manager="make",
        test_command="make test",
        docker_image="ubuntu:24.04",
        confidence=0.90,
        evidence=[
            f"C/C++ files: {len(cpp_files)}",
            "Makefile",
        ],
    )


# ============================================================================
# DYNAMIC PROJECT DETECTION
# ============================================================================

def detect_project_profile(root: Path) -> ProjectProfile:
    detectors: list[
        Callable[[Path], ProjectProfile | None]
    ] = [
        _detect_python,
        _detect_node,
        _detect_java,
        _detect_go,
        _detect_rust,
        _detect_cpp,
    ]

    matches: list[ProjectProfile] = []

    for detector in detectors:
        try:
            result = detector(root)
        except Exception:
            result = None

        if result is not None:
            matches.append(result)

    if not matches:
        raise RuntimeError(
            "Unsupported project. "
            "HEALFORGE supports Python, "
            "Node.js/TypeScript, Java, Go, Rust "
            "and C/C++."
        )

    priority = {
        "python": 6,
        "node": 5,
        "java": 4,
        "go": 3,
        "rust": 2,
        "cpp": 1,
    }

    matches.sort(
        key=lambda profile: (
            priority.get(profile.language, 0),
            profile.confidence,
        ),
        reverse=True,
    )

    return matches[0]


def detect_project(root: Path) -> tuple[str, str]:
    profile = detect_project_profile(root)

    return (
        profile.language,
        profile.test_command,
    )


# ============================================================================
# GIT CHECKOUT
# ============================================================================

def checkout(
    repo_url: str,
    sha: str,
    destination: Path,
    pr_number: int | None = None,
) -> None:
    """
    Checkout the exact PR head commit.

    Compatible with current main.py:
        checkout(repo_url, sha, root)

    Also supports PR refs when pr_number is supplied.
    """

    destination = destination.resolve()

    if destination.exists():
        shutil.rmtree(destination)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    code, output = run_process(
        [
            "git",
            "clone",
            "--no-tags",
            "--depth",
            "1",
            "--no-single-branch",
            repo_url,
            str(destination),
        ],
        destination.parent,
        180,
    )

    if code != 0:
        raise RuntimeError(
            f"Git clone failed:\n{output}"
        )

    if pr_number is not None:
        refspec = (
            f"pull/{pr_number}/head:"
            f"refs/remotes/origin/"
            f"healforge-pr-{pr_number}"
        )

        code, output = run_process(
            [
                "git",
                "fetch",
                "--depth",
                "1",
                "origin",
                refspec,
            ],
            destination,
            120,
        )

        if code != 0:
            raise RuntimeError(
                f"Git PR fetch failed:\n{output}"
            )

        target = (
            f"refs/remotes/origin/"
            f"healforge-pr-{pr_number}"
        )

    else:
        code, output = run_process(
            [
                "git",
                "fetch",
                "--depth",
                "1",
                "origin",
                sha,
            ],
            destination,
            120,
        )

        if code != 0:
            raise RuntimeError(
                f"Git commit fetch failed:\n{output}"
            )

        target = sha

    code, output = run_process(
        [
            "git",
            "checkout",
            "--detach",
            target,
        ],
        destination,
        60,
    )

    if code != 0:
        raise RuntimeError(
            f"Git checkout failed:\n{output}"
        )


# ============================================================================
# PATCH SAFETY
# ============================================================================

def validate_patch_paths(patch: str) -> None:
    patterns = (
        r"(?m)^---\s+[ab]/([^\s]+)",
        r"(?m)^\+\+\+\s+[ab]/([^\s]+)",
    )

    for pattern in patterns:
        for match in re.finditer(
            pattern,
            patch,
        ):
            path = (
                match.group(1)
                .replace("\\", "/")
                .strip()
            )

            if path == "/dev/null":
                continue

            if not is_safe_repo_path(path):
                raise RuntimeError(
                    "Patch contains an unsafe path."
                )

            if is_sensitive_path(path):
                raise RuntimeError(
                    "Patch attempts to modify a sensitive file."
                )


# ============================================================================
# PATCH APPLICATION
# ============================================================================

def apply_patch(
    root: Path,
    patch_file: Path,
) -> tuple[bool, str]:

    patch = patch_file.read_text(
        encoding="utf-8",
        errors="replace",
    )

    validate_patch_paths(patch)

    code, output = run_process(
        [
            "git",
            "apply",
            "--check",
            str(patch_file),
        ],
        root,
        30,
    )

    if code != 0:
        return False, output

    code, output = run_process(
        [
            "git",
            "apply",
            "--whitespace=nowarn",
            str(patch_file),
        ],
        root,
        30,
    )

    return code == 0, output


# ============================================================================
# DOCKER BUILD CONTEXT
# ============================================================================

def _copy_build_context(
    root: Path,
    destination: Path,
) -> None:

    root = root.resolve()

    def ignore(
        directory: str,
        names: list[str],
    ) -> list[str]:

        ignored: list[str] = []

        base = Path(directory)

        for name in names:
            source = base / name

            try:
                relative = source.relative_to(
                    root
                ).as_posix()
            except ValueError:
                ignored.append(name)
                continue

            if name in IGNORED:
                ignored.append(name)
                continue

            if is_sensitive_path(relative):
                ignored.append(name)
                continue

            if source.is_symlink():
                ignored.append(name)
                continue

            try:
                if (
                    source.is_file()
                    and source.stat().st_size
                    > settings.max_sandbox_file_bytes
                ):
                    ignored.append(name)
            except OSError:
                ignored.append(name)

        return ignored

    shutil.copytree(
        root,
        destination,
        ignore=ignore,
        symlinks=False,
    )


# ============================================================================
# DOCKERFILE GENERATION
# ============================================================================

def _dockerfile_for(
    profile: ProjectProfile,
) -> str:

    if profile.language == "python":

        if profile.package_manager == "uv":
            install = (
                "python -m pip install "
                "--no-cache-dir uv && "
                "uv sync --frozen"
            )

        elif profile.package_manager == "poetry":
            install = (
                "python -m pip install "
                "--no-cache-dir poetry && "
                "poetry install --no-interaction"
            )

        elif profile.package_manager == "pipenv":
            install = (
                "python -m pip install "
                "--no-cache-dir pipenv && "
                "pipenv install --dev"
            )

        elif (
            "requirements.txt"
            in profile.evidence
        ):
            install = (
                "python -m pip install "
                "--no-cache-dir "
                "-r requirements.txt"
            )

        else:
            install = "true"

        return f"""
FROM {profile.docker_image}

WORKDIR /opt/healforge-repo

COPY . /opt/healforge-repo

RUN {install}

RUN python -m pip install --no-cache-dir pytest

# Runtime verification copies the immutable repository into a disposable
# writable workspace before executing the test command.
# cp -a /opt/healforge-repo/. /work/
# cd /work

CMD ["sh", "-lc", "{profile.test_command}"]
""".strip() + "\n"

    if profile.language == "node":

        if profile.package_manager == "pnpm":
            install = (
                "corepack enable && "
                "pnpm install --frozen-lockfile"
            )

        elif profile.package_manager == "yarn":
            install = (
                "corepack enable && "
                "yarn install --immutable"
            )

        elif (
            "package-lock.json"
            in profile.evidence
        ):
            install = "npm ci"

        else:
            install = "npm install"

        return f"""
FROM {profile.docker_image}

WORKDIR /opt/healforge-repo

COPY . /opt/healforge-repo

RUN {install}

CMD ["sh", "-lc", "{profile.test_command}"]
""".strip() + "\n"

    if profile.language == "java":

        if profile.package_manager == "maven":
            install = (
                "mvn -q -DskipTests "
                "dependency:go-offline"
            )

        elif profile.package_manager == "gradle-wrapper":
            install = (
                "chmod +x gradlew && "
                "./gradlew dependencies --no-daemon"
            )

        else:
            install = (
                "gradle dependencies "
                "--no-daemon"
            )

        return f"""
FROM {profile.docker_image}

WORKDIR /opt/healforge-repo

COPY . /opt/healforge-repo

RUN {install}

CMD ["sh", "-lc", "{profile.test_command}"]
""".strip() + "\n"

    if profile.language == "go":

        return f"""
FROM {profile.docker_image}

WORKDIR /opt/healforge-repo

COPY go.mod ./
COPY go.sum* ./

RUN go mod download

COPY . /opt/healforge-repo

CMD ["sh", "-lc", "{profile.test_command}"]
""".strip() + "\n"

    if profile.language == "rust":

        return f"""
FROM {profile.docker_image}

WORKDIR /opt/healforge-repo

COPY Cargo.toml ./
COPY Cargo.lock* ./

RUN cargo fetch

COPY . /opt/healforge-repo

CMD ["sh", "-lc", "{profile.test_command}"]
""".strip() + "\n"

    if profile.language == "cpp":

        return f"""
FROM {profile.docker_image}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        make \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/healforge-repo

COPY . /opt/healforge-repo

RUN if [ -f CMakeLists.txt ]; then \
        cmake -S . -B build && \
        cmake --build build; \
    elif [ -f Makefile ]; then \
        make; \
    fi

CMD ["sh", "-lc", "{profile.test_command}"]
""".strip() + "\n"

    raise RuntimeError(
        f"Unsupported sandbox language: "
        f"{profile.language}"
    )


# ============================================================================
# CONTENT FINGERPRINT
# ============================================================================

def _content_fingerprint(
    root: Path,
) -> str:

    digest = hashlib.sha256()

    for path in sorted(
        _source_files(root)
    ):
        try:
            relative = path.relative_to(
                root
            ).as_posix()
        except ValueError:
            continue

        if is_sensitive_path(relative):
            continue

        digest.update(
            relative.encode(
                "utf-8",
                errors="replace",
            )
        )

        digest.update(b"\0")

        try:
            digest.update(
                path.read_bytes()
            )
        except OSError:
            continue

    return digest.hexdigest()[:16]


# ============================================================================
# DOCKER IMAGE BUILD
# ============================================================================

def _build_sandbox_image(
    root: Path,
    profile: ProjectProfile,
) -> tuple[bool, str]:

    if not _docker_available():
        return (
            False,
            "Docker is required for sandbox verification. "
            "Start Docker Desktop and retry.",
        )

    fingerprint = _content_fingerprint(
        root
    )

    image_tag = (
        f"healforge-sandbox-"
        f"{profile.language}-"
        f"{fingerprint}:latest"
    )

    with tempfile.TemporaryDirectory(
        prefix="healforge-build-"
    ) as temp:

        context = Path(temp)

        repo_context = (
            context / "repo"
        )

        try:
            _copy_build_context(
                root,
                repo_context,
            )

        except Exception as exc:
            return (
                False,
                "Could not prepare sandbox "
                f"build context: {exc}",
            )

        dockerfile = (
            context / "Dockerfile"
        )

        dockerfile.write_text(
            _dockerfile_for(profile),
            encoding="utf-8",
        )

        build_network = getattr(
            settings,
            "sandbox_build_network",
            "default",
        )

        timeout = getattr(
            settings,
            "sandbox_timeout_seconds",
            180,
        )

        code, output = run_process(
            [
                "docker",
                "build",
                "--network",
                build_network,
                "-t",
                image_tag,
                "-f",
                str(dockerfile),
                str(repo_context),
            ],
            context,
            timeout,
        )

        if code != 0:
            return (
                False,
                "Sandbox image preparation failed.\n\n"
                + output,
            )

    return True, image_tag


# ============================================================================
# SANDBOX RUNTIME
# ============================================================================

def _run_sandbox(
    image: str,
    command: str,
) -> RunResult:

    timeout = getattr(
        settings,
        "sandbox_timeout_seconds",
        180,
    )

    safe_command = (
        command
        .replace(
            "'",
            "'\"'\"'",
        )
    )

    runtime_command = command

    if command == "pytest -q":
        runtime_command = "pytest -q -p no:cacheprovider"

    elif command == "python -m pytest -q":
        runtime_command = "python -m pytest -q -p no:cacheprovider"

    elif command == "gradle test":
        runtime_command = "gradle test --no-daemon"

    safe_command = (
        runtime_command
        .replace("'", "'\"'\"'")
    )

    runtime_script = (
        "set -eu; "
        "mkdir -p /work; "
        "cp -a /opt/healforge-repo/. /work/; "
        "cd /work; "
        f"exec sh -lc '{safe_command}'"
    )

    args = [
        "docker",
        "run",
        "--rm",

        # No network during verification.
        "--network",
        "none",

        # Resource limits.
        "--cpus",
        "1.5",

        "--memory",
        "768m",

        "--pids-limit",
        "128",

        # Drop Linux capabilities.
        "--cap-drop",
        "ALL",

        # Prevent privilege escalation.
        "--security-opt",
        "no-new-privileges:true",

        # Immutable container root.
        "--read-only",

        # Writable disposable workspace.
        "--tmpfs",
        "/work:rw,nosuid,nodev,size=512m",

        # Writable temporary directory but no executable files.
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=256m",

        image,

        "sh",
        "-lc",
        runtime_script,
    ]

    code, output = run_process(
        args,
        Path.cwd(),
        timeout,
    )

    return RunResult(
        passed=code == 0,
        command=command,
        exit_code=code,
        output=output,
    )


# ============================================================================
# MAIN SANDBOX ENTRY POINT
# ============================================================================

def docker_test(
    root: Path,
    language: str,
    command: str,
) -> RunResult:

    if not root.exists():
        raise RuntimeError(
            f"Sandbox repository does not exist: "
            f"{root}"
        )

    profile = detect_project_profile(
        root
    )

    # Detection wins over stale caller information.
    language = profile.language

    ready, image_or_error = (
        _build_sandbox_image(
            root,
            profile,
        )
    )

    if not ready:
        return RunResult(
            passed=False,
            command=profile.test_command,
            exit_code=1,
            output=image_or_error,
        )

    return _run_sandbox(
        image_or_error,
        profile.test_command,
    )