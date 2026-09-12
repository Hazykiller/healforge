import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .config import settings


# ---------------------------------------------------------------------------
# RESULT TYPES
# ---------------------------------------------------------------------------

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
    install_command: str
    docker_image: str
    confidence: float
    evidence: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PROCESS HELPERS
# ---------------------------------------------------------------------------

def run_process(
    args: list[str],
    cwd: Path,
    timeout: int = 120,
) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return completed.returncode, completed.stdout[-30000:]
    except subprocess.TimeoutExpired as exc:
        return 124, f"Process timed out after {timeout}s: {exc}"


def _docker_available() -> bool:
    return shutil.which("docker") is not None


# ---------------------------------------------------------------------------
# PROJECT DETECTION
# ---------------------------------------------------------------------------

def _read_text(root: Path, name: str, limit: int = 20000) -> str:
    path = root / name

    if not path.is_file():
        return ""

    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    except Exception:
        return ""


def _has_any(root: Path, names: list[str]) -> bool:
    return any((root / name).exists() for name in names)


def _source_files(root: Path) -> list[Path]:
    ignored = {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "target",
        "build",
        "dist",
        "__pycache__",
        ".pytest_cache",
        "workspace",
    }

    files: list[Path] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if any(part in ignored for part in path.parts):
            continue

        files.append(path)

        if len(files) >= 500:
            break

    return files


def _detect_python(root: Path) -> ProjectProfile | None:
    markers = [
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        "setup.py",
        "setup.cfg",
        "Pipfile",
        "poetry.lock",
        "uv.lock",
    ]

    source = _source_files(root)

    python_files = [
        p for p in source
        if p.suffix == ".py"
    ]

    if not _has_any(root, markers) and not python_files:
        return None

    pyproject = _read_text(root, "pyproject.toml")
    requirements = (
        _read_text(root, "requirements.txt")
        + "\n"
        + _read_text(root, "requirements-dev.txt")
    )

    if "django" in pyproject.lower() or "django" in requirements.lower():
        framework = "Django"
    elif "fastapi" in pyproject.lower() or "fastapi" in requirements.lower():
        framework = "FastAPI"
    elif "flask" in pyproject.lower() or "flask" in requirements.lower():
        framework = "Flask"
    elif "pytest" in pyproject.lower() or "pytest" in requirements.lower() or (root / "pytest.ini").exists():
        framework = "pytest"
    else:
        framework = "Python"

    if (root / "uv.lock").exists():
        package_manager = "uv"
        install = "uv sync"
    elif (root / "poetry.lock").exists():
        package_manager = "poetry"
        install = "poetry install --no-interaction"
    elif (root / "Pipfile").exists():
        package_manager = "pipenv"
        install = "pipenv install --dev"
    elif (root / "requirements.txt").exists():
        package_manager = "pip"
        install = "python -m pip install -r requirements.txt"
    elif (root / "requirements-dev.txt").exists():
        package_manager = "pip"
        install = "python -m pip install -r requirements-dev.txt"
    else:
        package_manager = "pip"
        install = "python -m pip install -e ."

    if (
        (root / "pytest.ini").exists()
        or (root / "tests").is_dir()
        or "pytest" in pyproject.lower()
        or "pytest" in requirements.lower()
    ):
        test_command = "pytest -q"
    elif any(p.name.startswith("test_") for p in python_files):
        test_command = "pytest -q"
    else:
        test_command = "python -m unittest discover -v"

    return ProjectProfile(
        language="python",
        framework=framework,
        package_manager=package_manager,
        test_command=test_command,
        install_command=install,
        docker_image=settings.docker_image_python,
        confidence=0.98,
        evidence=[
            f"Python source files: {len(python_files)}",
            *[m for m in markers if (root / m).exists()],
        ],
    )


def _detect_node(root: Path) -> ProjectProfile | None:
    package = root / "package.json"

    js_files = [
        p for p in _source_files(root)
        if p.suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
    ]

    if not package.is_file() and not js_files:
        return None

    package_text = _read_text(root, "package.json")

    try:
        package_json = json.loads(package_text) if package_text else {}
    except json.JSONDecodeError:
        package_json = {}

    dependencies = json.dumps(
        package_json.get("dependencies", {}),
    ).lower()

    dev_dependencies = json.dumps(
        package_json.get("devDependencies", {}),
    ).lower()

    combined = dependencies + dev_dependencies

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
        install = "corepack enable && pnpm install --frozen-lockfile"
        runner = "pnpm"
    elif (root / "yarn.lock").exists():
        package_manager = "yarn"
        install = "corepack enable && yarn install --immutable"
        runner = "yarn"
    elif (root / "package-lock.json").exists():
        package_manager = "npm"
        install = "npm ci"
        runner = "npm"
    else:
        package_manager = "npm"
        install = "npm install"
        runner = "npm"

    scripts = package_json.get("scripts", {})
    test_script = scripts.get("test")

    if test_script:
        test_command = f"{runner} test"
    elif "vitest" in combined:
        test_command = f"{runner} exec vitest run"
    elif "jest" in combined:
        test_command = f"{runner} exec jest --runInBand"
    else:
        test_command = f"{runner} test"

    return ProjectProfile(
        language="node",
        framework=framework,
        package_manager=package_manager,
        test_command=test_command,
        install_command=install,
        docker_image=settings.docker_image_node,
        confidence=0.98,
        evidence=[
            f"JavaScript/TypeScript source files: {len(js_files)}",
            *[
                name
                for name in [
                    "package.json",
                    "package-lock.json",
                    "pnpm-lock.yaml",
                    "yarn.lock",
                ]
                if (root / name).exists()
            ],
        ],
    )


def _detect_java(root: Path) -> ProjectProfile | None:
    java_files = [
        p for p in _source_files(root)
        if p.suffix == ".java"
    ]

    if not java_files and not _has_any(
        root,
        [
            "pom.xml",
            "build.gradle",
            "build.gradle.kts",
            "gradlew",
        ],
    ):
        return None

    pom = _read_text(root, "pom.xml")
    gradle = (
        _read_text(root, "build.gradle")
        + "\n"
        + _read_text(root, "build.gradle.kts")
    )

    combined = (pom + "\n" + gradle).lower()

    if "spring-boot" in combined or "springframework" in combined:
        framework = "Spring Boot"
    elif "junit" in combined:
        framework = "JUnit"
    else:
        framework = "Java"

    if (root / "pom.xml").exists():
        package_manager = "Maven"
        install = "mvn -q -DskipTests dependency:go-offline"
        test_command = "mvn test -q"
    else:
        package_manager = "Gradle"
        install = "./gradlew dependencies --no-daemon"
        test_command = "./gradlew test --no-daemon"

    return ProjectProfile(
        language="java",
        framework=framework,
        package_manager=package_manager,
        test_command=test_command,
        install_command=install,
        docker_image="eclipse-temurin:21-jdk",
        confidence=0.98,
        evidence=[
            f"Java source files: {len(java_files)}",
            *[
                name
                for name in [
                    "pom.xml",
                    "build.gradle",
                    "build.gradle.kts",
                    "gradlew",
                ]
                if (root / name).exists()
            ],
        ],
    )


def _detect_go(root: Path) -> ProjectProfile | None:
    go_files = [
        p for p in _source_files(root)
        if p.suffix == ".go"
    ]

    if not go_files and not (root / "go.mod").exists():
        return None

    return ProjectProfile(
        language="go",
        framework="Go",
        package_manager="Go Modules",
        test_command="go test ./...",
        install_command="go mod download",
        docker_image="golang:1.25-bookworm",
        confidence=0.99,
        evidence=[
            f"Go source files: {len(go_files)}",
            "go.mod" if (root / "go.mod").exists() else "",
        ],
    )


def _detect_rust(root: Path) -> ProjectProfile | None:
    rust_files = [
        p for p in _source_files(root)
        if p.suffix == ".rs"
    ]

    if not rust_files and not (root / "Cargo.toml").exists():
        return None

    return ProjectProfile(
        language="rust",
        framework="Cargo",
        package_manager="Cargo",
        test_command="cargo test",
        install_command="cargo fetch",
        docker_image="rust:1-bookworm",
        confidence=0.99,
        evidence=[
            f"Rust source files: {len(rust_files)}",
            "Cargo.toml" if (root / "Cargo.toml").exists() else "",
        ],
    )


def _detect_c_cpp(root: Path) -> ProjectProfile | None:
    source_files = [
        p for p in _source_files(root)
        if p.suffix in {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"}
    ]

    if not source_files and not _has_any(
        root,
        ["CMakeLists.txt", "Makefile"],
    ):
        return None

    if (root / "CMakeLists.txt").exists():
        return ProjectProfile(
            language="cpp",
            framework="CMake",
            package_manager="CMake",
            test_command="ctest --output-on-failure",
            install_command="cmake -S . -B build && cmake --build build",
            docker_image="gcc:15-bookworm",
            confidence=0.95,
            evidence=[
                "CMakeLists.txt",
                f"C/C++ source files: {len(source_files)}",
            ],
        )

    return ProjectProfile(
        language="cpp",
        framework="Make",
        package_manager="Make",
        test_command="make test",
        install_command="make",
        docker_image="gcc:15-bookworm",
        confidence=0.90,
        evidence=[
            "Makefile",
            f"C/C++ source files: {len(source_files)}",
        ],
    )


def detect_project(root: Path) -> tuple[str, str]:
    profile = detect_project_profile(root)

    return profile.language, profile.test_command


def detect_project_profile(root: Path) -> ProjectProfile:
    detectors = [
        _detect_python,
        _detect_node,
        _detect_java,
        _detect_go,
        _detect_rust,
        _detect_c_cpp,
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
            "HEALFORGE could not identify a supported project. "
            "Supported ecosystems: Python, Node.js/TypeScript, Java, Go, Rust, C/C++."
        )

    # Prefer explicit build-system markers over loose source extensions.
    priority = {
        "python": 6,
        "node": 5,
        "java": 4,
        "go": 3,
        "rust": 2,
        "cpp": 1,
    }

    matches.sort(
        key=lambda item: (
            priority.get(item.language, 0),
            item.confidence,
        ),
        reverse=True,
    )

    return matches[0]


# ---------------------------------------------------------------------------
# GIT OPERATIONS
# ---------------------------------------------------------------------------

def checkout(
    repo_url: str,
    sha: str,
    destination: Path,
) -> None:
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
            repo_url,
            str(destination),
        ],
        destination.parent,
        180,
    )

    if code != 0:
        raise RuntimeError(
            f"git clone failed:\n{output}"
        )

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
            f"git fetch failed:\n{output}"
        )

    code, output = run_process(
        [
            "git",
            "checkout",
            "--detach",
            sha,
        ],
        destination,
        60,
    )

    if code != 0:
        raise RuntimeError(
            f"git checkout failed:\n{output}"
        )


def apply_patch(
    root: Path,
    patch_file: Path,
) -> tuple[bool, str]:
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
            str(patch_file),
        ],
        root,
        30,
    )

    return code == 0, output


# ---------------------------------------------------------------------------
# DOCKER BUILD CONTEXT
# ---------------------------------------------------------------------------

def _copy_build_context(
    root: Path,
    destination: Path,
) -> None:
    ignored_names = {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        "workspace",
        "target",
        "build",
        "dist",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".idea",
        ".vscode",
    }

    def ignore(directory: str, names: list[str]):
        ignored = []

        for name in names:
            if name in ignored_names:
                ignored.append(name)

        return ignored

    shutil.copytree(
        root,
        destination,
        ignore=ignore,
    )


def _dockerfile_for(profile: ProjectProfile) -> str:
    language = profile.language

    if language == "python":
        return f"""
FROM {profile.docker_image}

WORKDIR /work

COPY . /work

RUN python -m pip install --disable-pip-version-check --no-cache-dir --upgrade pip

RUN if [ -f requirements.txt ]; then \
        python -m pip install --disable-pip-version-check --no-cache-dir -r requirements.txt; \
    fi

RUN if [ -f requirements-dev.txt ]; then \
        python -m pip install --disable-pip-version-check --no-cache-dir -r requirements-dev.txt; \
    fi

RUN if [ -f pyproject.toml ] && [ ! -f requirements.txt ]; then \
        python -m pip install --disable-pip-version-check --no-cache-dir -e .; \
    fi

RUN python -m pip install --disable-pip-version-check --no-cache-dir pytest

CMD ["sh", "-lc", "{profile.test_command}"]
"""

    if language == "node":
        if profile.package_manager == "pnpm":
            install = "corepack enable && pnpm install --frozen-lockfile"
        elif profile.package_manager == "yarn":
            install = "corepack enable && yarn install --immutable"
        else:
            install = "npm ci"

        return f"""
FROM {profile.docker_image}

WORKDIR /work

COPY . /work

RUN {install}

CMD ["sh", "-lc", "{profile.test_command}"]
"""

    if language == "java":
        if profile.package_manager == "Maven":
            return f"""
FROM {profile.docker_image}

WORKDIR /work

COPY . /work

RUN mvn -q -DskipTests dependency:go-offline

CMD ["sh", "-lc", "{profile.test_command}"]
"""

        return f"""
FROM {profile.docker_image}

WORKDIR /work

COPY . /work

RUN chmod +x gradlew 2>/dev/null || true
RUN ./gradlew dependencies --no-daemon

CMD ["sh", "-lc", "{profile.test_command}"]
"""

    if language == "go":
        return f"""
FROM {profile.docker_image}

WORKDIR /work

COPY go.mod go.sum* ./

RUN go mod download

COPY . /work

CMD ["sh", "-lc", "{profile.test_command}"]
"""

    if language == "rust":
        return f"""
FROM {profile.docker_image}

WORKDIR /work

COPY Cargo.toml Cargo.lock* ./

RUN cargo fetch

COPY . /work

CMD ["sh", "-lc", "{profile.test_command}"]
"""

    if language == "cpp":
        return f"""
FROM {profile.docker_image}

WORKDIR /work

COPY . /work

RUN if [ -f CMakeLists.txt ]; then \
        cmake -S . -B build && cmake --build build; \
    elif [ -f Makefile ]; then \
        make; \
    fi

CMD ["sh", "-lc", "{profile.test_command}"]
"""

    raise RuntimeError(
        f"Unsupported sandbox language: {language}"
    )


def _build_sandbox_image(
    root: Path,
    profile: ProjectProfile,
) -> tuple[bool, str]:
    if not _docker_available():
        return False, (
            "Docker is required for sandbox verification. "
            "Install and start Docker Desktop, then retry."
        )

    image_tag = (
        "healforge-sandbox-"
        + re.sub(
            r"[^a-zA-Z0-9_.-]",
            "-",
            profile.language,
        )
        + "-"
        + re.sub(
            r"[^a-zA-Z0-9_.-]",
            "-",
            profile.package_manager.lower(),
        )
        + ":latest"
    )

    with tempfile.TemporaryDirectory(
        prefix="healforge-build-"
    ) as temp:
        context = Path(temp)

        try:
            _copy_build_context(
                root,
                context / "repo",
            )
        except Exception as exc:
            return False, (
                f"Could not prepare sandbox build context: {exc}"
            )

        dockerfile = context / "Dockerfile"

        dockerfile.write_text(
            _dockerfile_for(profile),
            encoding="utf-8",
        )

        code, output = run_process(
            [
                "docker",
                "build",
                "--pull",
                "-t",
                image_tag,
                "-f",
                str(dockerfile),
                str(context / "repo"),
            ],
            context,
            600,
        )

        if code != 0:
            return False, (
                "Sandbox image preparation failed.\n\n"
                + output
            )

    return True, image_tag


# ---------------------------------------------------------------------------
# SANDBOX EXECUTION
# ---------------------------------------------------------------------------

def _run_sandbox(
    image: str,
    command: str,
) -> RunResult:
    args = [
        "docker",
        "run",
        "--rm",

        # No outbound network during actual verification.
        "--network",
        "none",

        # Resource limits.
        "--cpus",
        "1.5",
        "--memory",
        "768m",
        "--pids-limit",
        "128",

        # Immutable container filesystem.
        "--read-only",

        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=256m",

        image,
        "sh",
        "-lc",
        command,
    ]

    code, output = run_process(
        args,
        Path.cwd(),
        300,
    )

    return RunResult(
        passed=code == 0,
        command=command,
        exit_code=code,
        output=output,
    )


def docker_test(
    root: Path,
    language: str,
    command: str,
) -> RunResult:
    """
    Build an isolated dependency-ready sandbox and execute the test
    command without network access.

    The project is detected dynamically and dependencies are prepared
    during image build. The actual verification phase has networking
    disabled.
    """

    if not root.exists():
        raise RuntimeError(
            f"Sandbox repository does not exist: {root}"
        )

    profile = detect_project_profile(root)

    # The profile discovered during verification must agree with the
    # language selected by the caller.
    if profile.language != language:
        language = profile.language

    ready, image_or_error = _build_sandbox_image(
        root,
        profile,
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