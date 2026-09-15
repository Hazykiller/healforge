import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .config import settings
from .security import is_sensitive_path, is_safe_repo_path


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


IGNORED = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
    "workspace", "target", "build", "dist", ".mypy_cache", ".ruff_cache",
    ".tox", ".idea", ".vscode",
}


def run_process(args: list[str], cwd: Path, timeout: int = 120) -> tuple[int, str]:
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
        return completed.returncode, completed.stdout[-40000:]
    except subprocess.TimeoutExpired:
        return 124, f"Process timed out after {timeout}s"


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _read_text(root: Path, name: str, limit: int = 30000) -> str:
    path = root / name
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    except OSError:
        return ""


def _source_files(root: Path) -> list[Path]:
    result: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED for part in path.relative_to(root).parts):
            continue
        result.append(path)
        if len(result) >= 1000:
            break
    return result


def _has(root: Path, *names: str) -> bool:
    return any((root / name).exists() for name in names)


def _detect_python(root: Path) -> ProjectProfile | None:
    files = _source_files(root)
    py = [p for p in files if p.suffix == ".py"]
    markers = [
        name for name in [
            "pyproject.toml", "requirements.txt", "requirements-dev.txt",
            "setup.py", "setup.cfg", "Pipfile", "poetry.lock", "uv.lock",
        ] if (root / name).exists()
    ]
    if not py and not markers:
        return None

    pyproject = _read_text(root, "pyproject.toml").lower()
    req = (_read_text(root, "requirements.txt") + _read_text(root, "requirements-dev.txt")).lower()
    combined = pyproject + "\n" + req

    if "django" in combined:
        framework = "Django"
    elif "fastapi" in combined:
        framework = "FastAPI"
    elif "flask" in combined:
        framework = "Flask"
    elif "pytest" in combined or _has(root, "pytest.ini", "tox.ini"):
        framework = "pytest"
    else:
        framework = "Python"

    if _has(root, "pytest.ini") or (root / "tests").is_dir() or any(p.name.startswith("test_") for p in py) or "pytest" in combined:
        test = "pytest -q"
    else:
        test = "python -m unittest discover -v"

    return ProjectProfile(
        language="python",
        framework=framework,
        package_manager=("uv" if (root / "uv.lock").exists() else "poetry" if (root / "poetry.lock").exists() else "pipenv" if (root / "Pipfile").exists() else "pip"),
        test_command=test,
        docker_image=settings.docker_image_python,
        confidence=0.99,
        evidence=[f"Python files: {len(py)}", *markers],
    )


def _detect_node(root: Path) -> ProjectProfile | None:
    package = root / "package.json"
    files = _source_files(root)
    js = [p for p in files if p.suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}]
    if not package.is_file() and not js:
        return None

    raw = _read_text(root, "package.json")
    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        data = {}

    combined = json.dumps(
        {"dependencies": data.get("dependencies", {}), "devDependencies": data.get("devDependencies", {})}
    ).lower()

    if "next" in combined:
        framework = "Next.js"
    elif "react" in combined:
        framework = "React"
    elif "vue" in combined:
        framework = "Vue"
    elif "express" in combined:
        framework = "Express"
    elif "@nestjs" in combined or "nestjs" in combined:
        framework = "NestJS"
    elif "vite" in combined:
        framework = "Vite"
    else:
        framework = "Node.js"

    if (root / "pnpm-lock.yaml").exists():
        pm = "pnpm"
        command = "pnpm test"
    elif (root / "yarn.lock").exists():
        pm = "yarn"
        command = "yarn test"
    else:
        pm = "npm"
        command = "npm test"

    scripts = data.get("scripts", {})
    if not scripts.get("test"):
        if "vitest" in combined:
            command = f"{pm} exec vitest run"
        elif "jest" in combined:
            command = f"{pm} exec jest --runInBand"

    return ProjectProfile(
        language="node",
        framework=framework,
        package_manager=pm,
        test_command=command,
        docker_image=settings.docker_image_node,
        confidence=0.99,
        evidence=[
            f"JS/TS files: {len(js)}",
            *[name for name in ["package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock"] if (root / name).exists()],
        ],
    )


def _detect_java(root: Path) -> ProjectProfile | None:
    files = _source_files(root)
    java = [p for p in files if p.suffix == ".java"]
    if not java and not _has(root, "pom.xml", "build.gradle", "build.gradle.kts", "gradlew"):
        return None

    combined = (_read_text(root, "pom.xml") + _read_text(root, "build.gradle") + _read_text(root, "build.gradle.kts")).lower()
    framework = "Spring Boot" if "spring-boot" in combined or "springframework" in combined else "Java"

    if (root / "pom.xml").exists():
        pm = "maven"
        test = "mvn test -q"
        image = "eclipse-temurin:21-jdk"
    else:
        pm = "gradle"
        if (root / "gradlew").is_file():
            test = "./gradlew test --no-daemon"
            image = "eclipse-temurin:21-jdk"
        else:
            test = "gradle test"
            image = "gradle:8.10-jdk21"

    return ProjectProfile("java", framework, pm, test, image, 0.98, [f"Java files: {len(java)}"])


def _detect_go(root: Path) -> ProjectProfile | None:
    files = _source_files(root)
    go = [p for p in files if p.suffix == ".go"]
    if not go and not (root / "go.mod").exists():
        return None
    return ProjectProfile("go", "Go", "go-modules", "go test ./...", "golang:1.25-bookworm", 0.99, [f"Go files: {len(go)}"])


def _detect_rust(root: Path) -> ProjectProfile | None:
    files = _source_files(root)
    rs = [p for p in files if p.suffix == ".rs"]
    if not rs and not (root / "Cargo.toml").exists():
        return None
    return ProjectProfile("rust", "Cargo", "cargo", "cargo test", "rust:1-bookworm", 0.99, [f"Rust files: {len(rs)}"])


def _detect_cpp(root: Path) -> ProjectProfile | None:
    files = _source_files(root)
    cpp = [p for p in files if p.suffix in {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"}]
    if not cpp and not _has(root, "CMakeLists.txt", "Makefile"):
        return None
    if (root / "CMakeLists.txt").exists():
        return ProjectProfile("cpp", "CMake", "cmake", "ctest --test-dir build --output-on-failure", "gcc:15-bookworm", 0.96, [f"C/C++ files: {len(cpp)}"])
    return ProjectProfile("cpp", "Make", "make", "make test", "gcc:15-bookworm", 0.90, [f"C/C++ files: {len(cpp)}"])


def detect_project_profile(root: Path) -> ProjectProfile:
    detectors = [_detect_python, _detect_node, _detect_java, _detect_go, _detect_rust, _detect_cpp]
    matches = [result for detector in detectors if (result := detector(root)) is not None]
    if not matches:
        raise RuntimeError("Unsupported project. HEALFORGE supports Python, Node.js/TypeScript, Java, Go, Rust and C/C++.")
    priority = {"python": 6, "node": 5, "java": 4, "go": 3, "rust": 2, "cpp": 1}
    matches.sort(key=lambda p: (priority[p.language], p.confidence), reverse=True)
    return matches[0]


def detect_project(root: Path) -> tuple[str, str]:
    profile = detect_project_profile(root)
    return profile.language, profile.test_command


def checkout(repo_url: str, sha: str, destination: Path, pr_number: int | None = None) -> None:
    """Create a detached checkout of the exact PR head commit.

    GitHub exposes open PR heads through refs/pull/<number>/head. Fetching that
    ref is more reliable than assuming the commit SHA is reachable from the
    base repository's normal refs, especially for fork-based PRs.
    """
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    code, output = run_process(
        [
            "git", "clone", "--no-tags", "--depth", "1",
            "--no-single-branch", repo_url, str(destination),
        ],
        destination.parent,
        180,
    )
    if code != 0:
        raise RuntimeError(f"git clone failed:\n{output}")

    if pr_number is not None:
        refspec = f"pull/{pr_number}/head:refs/remotes/origin/healforge-pr-{pr_number}"
        code, output = run_process(
            ["git", "fetch", "--depth", "1", "origin", refspec],
            destination,
            120,
        )
        if code != 0:
            raise RuntimeError(f"git fetch PR head failed:\n{output}")
        checkout_target = f"refs/remotes/origin/healforge-pr-{pr_number}"
    else:
        code, output = run_process(
            ["git", "fetch", "--depth", "1", "origin", sha],
            destination,
            120,
        )
        if code != 0:
            raise RuntimeError(f"git fetch commit failed:\n{output}")
        checkout_target = sha

    code, output = run_process(["git", "checkout", "--detach", checkout_target], destination, 60)
    if code != 0:
        raise RuntimeError(f"git checkout failed:\n{output}")


def validate_patch_paths(patch: str) -> None:
    for match in re.finditer(r"(?m)^(?:---|\+\+\+)\s+[ab]/([^\s]+)", patch):
        path = match.group(1).replace("\\", "/")
        if not is_safe_repo_path(path):
            raise RuntimeError("Patch contains an unsafe path")
        if is_sensitive_path(path):
            raise RuntimeError("Patch attempts to modify a sensitive file")


def apply_patch(root: Path, patch_file: Path) -> tuple[bool, str]:
    patch = patch_file.read_text(encoding="utf-8", errors="replace")
    validate_patch_paths(patch)

    code, output = run_process(["git", "apply", "--check", str(patch_file)], root, 30)
    if code != 0:
        return False, output

    code, output = run_process(["git", "apply", "--whitespace=nowarn", str(patch_file)], root, 30)
    return code == 0, output


def _copy_build_context(root: Path, destination: Path) -> None:
    """Copy only ordinary, non-sensitive repository files into Docker context."""
    root = root.resolve()

    def ignore(directory: str, names: list[str]) -> list[str]:
        ignored: list[str] = []
        base = Path(directory)
        for name in names:
            source = base / name
            relative = source.relative_to(root).as_posix()
            if name in IGNORED or is_sensitive_path(relative):
                ignored.append(name)
                continue
            # A repository symlink can point outside the checkout. Do not put
            # symlinks into a build context that will execute untrusted code.
            if source.is_symlink():
                ignored.append(name)
                continue
            try:
                if source.is_file() and source.stat().st_size > settings.max_sandbox_file_bytes:
                    ignored.append(name)
            except OSError:
                ignored.append(name)
        return ignored

    shutil.copytree(root, destination, ignore=ignore, symlinks=False)


def _dockerfile_for(profile: ProjectProfile) -> str:
    if profile.language == "python":
        if profile.package_manager == "uv":
            install = "python -m pip install --no-cache-dir uv && uv sync --frozen"
        elif profile.package_manager == "poetry":
            install = "python -m pip install --no-cache-dir poetry && poetry install --no-interaction"
        elif profile.package_manager == "pipenv":
            install = "python -m pip install --no-cache-dir pipenv && pipenv install --dev"
        elif "requirements.txt" in profile.evidence:
            install = "python -m pip install --no-cache-dir -r requirements.txt"
        else:
            # A plain source repository is not necessarily an installable
            # Python package. Do not run `pip install -e .` unless packaging
            # metadata actually exists.
            install = "true"

        pytest_step = "RUN python -m pip install --no-cache-dir pytest\n"

        return f"""FROM {profile.docker_image}\nWORKDIR /opt/healforge-repo\nCOPY . /opt/healforge-repo\nRUN {install}\n{pytest_step}CMD [\"sh\", \"-lc\", \"{profile.test_command}\"]\n"""

    if profile.language == "node":
        if profile.package_manager == "pnpm":
            install = "corepack enable && pnpm install --frozen-lockfile"
        elif profile.package_manager == "yarn":
            install = "corepack enable && yarn install --immutable"
        elif (Path("package-lock.json").name in profile.evidence):
            install = "npm ci"
        else:
            install = "npm install"
        return f"""FROM {profile.docker_image}\nWORKDIR /work\nCOPY . /work\nRUN {install}\nCMD [\"sh\", \"-lc\", \"{profile.test_command}\"]\n"""

    if profile.language == "java":
        if profile.package_manager == "maven":
            install = "mvn -q -DskipTests dependency:go-offline"
        elif profile.test_command.startswith("./gradlew "):
            install = "chmod +x gradlew && ./gradlew dependencies --no-daemon"
        else:
            install = "gradle dependencies --no-daemon"
        return f"""FROM {profile.docker_image}\nWORKDIR /work\nCOPY . /work\nRUN {install}\nCMD [\"sh\", \"-lc\", \"{profile.test_command}\"]\n"""

    if profile.language == "go":
        return f"""FROM {profile.docker_image}\nWORKDIR /work\nCOPY go.mod go.sum* ./\nRUN go mod download\nCOPY . /work\nCMD [\"sh\", \"-lc\", \"{profile.test_command}\"]\n"""

    if profile.language == "rust":
        return f"""FROM {profile.docker_image}\nWORKDIR /work\nCOPY Cargo.toml Cargo.lock* ./\nRUN cargo fetch\nCOPY . /work\nCMD [\"sh\", \"-lc\", \"{profile.test_command}\"]\n"""

    if profile.language == "cpp":
        return f"""FROM {profile.docker_image}\nWORKDIR /work\nCOPY . /work\nRUN if [ -f CMakeLists.txt ]; then cmake -S . -B build && cmake --build build; elif [ -f Makefile ]; then make; fi\nCMD [\"sh\", \"-lc\", \"{profile.test_command}\"]\n"""

    raise RuntimeError(f"Unsupported sandbox language: {profile.language}")


def _content_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(_source_files(root)):
        rel = path.relative_to(root).as_posix()
        if is_sensitive_path(rel):
            continue
        digest.update(rel.encode())
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError:
            continue
    return digest.hexdigest()[:16]


def _build_sandbox_image(root: Path, profile: ProjectProfile) -> tuple[bool, str]:
    if not _docker_available():
        return False, "Docker is required for sandbox verification. Start Docker Desktop and retry."

    image_tag = f"healforge-sandbox-{profile.language}-{_content_fingerprint(root)}:latest"

    with tempfile.TemporaryDirectory(prefix="healforge-build-") as temp:
        context = Path(temp)
        repo_context = context / "repo"
        try:
            _copy_build_context(root, repo_context)
        except Exception as exc:
            return False, f"Could not prepare sandbox build context: {exc}"

        dockerfile = context / "Dockerfile"
        dockerfile.write_text(_dockerfile_for(profile), encoding="utf-8")

        code, output = run_process(
            [
                "docker", "build", "--network", settings.sandbox_build_network, "-t", image_tag,
                "-f", str(dockerfile), str(repo_context),
            ],
            context,
            settings.sandbox_timeout_seconds,
        )
        if code != 0:
            return False, "Sandbox image preparation failed.\n\n" + output

    return True, image_tag


def _run_sandbox(image: str, command: str) -> RunResult:
    safe_command = command.replace("'", "'\"'\"'")
    runtime_script = (
        "set -eu; "
        "mkdir -p /work; "
        "cp -a /opt/healforge-repo/. /work/; "
        "cd /work; "
        f"exec sh -lc '{safe_command}'"
    )
    args = [
        "docker", "run", "--rm",
        "--network", "none",
        "--cpus", "1.5",
        "--memory", "768m",
        "--pids-limit", "128",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges:true",
        "--read-only",
        "--tmpfs", "/work:rw,nosuid,nodev,size=512m",
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=256m",
        image,
        "sh", "-lc", runtime_script,
    ]
    code, output = run_process(args, Path.cwd(), settings.sandbox_timeout_seconds)
    return RunResult(code == 0, command, code, output)


def docker_test(root: Path, language: str, command: str) -> RunResult:
    if not root.exists():
        raise RuntimeError(f"Sandbox repository does not exist: {root}")

    profile = detect_project_profile(root)
    if profile.language != language:
        language = profile.language

    ready, image_or_error = _build_sandbox_image(root, profile)
    if not ready:
        return RunResult(False, profile.test_command, 1, image_or_error)

    return _run_sandbox(image_or_error, profile.test_command)
