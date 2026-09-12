import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import settings


@dataclass
class RunResult:
    passed: bool
    command: str
    exit_code: int
    output: str


# ---------------------------------------------------------------------------
# PROCESS EXECUTION
# ---------------------------------------------------------------------------

def run_process(
    args: list[str],
    cwd: Path,
    timeout: int = 120,
) -> tuple[int, str]:
    """
    Run a host-side process and return its exit code and combined output.

    Output is capped so a noisy test suite cannot fill the application
    response/session with an unbounded amount of text.
    """

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
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""

        if isinstance(output, bytes):
            output = output.decode(errors="replace")

        return 124, output[-30000:]

    return completed.returncode, completed.stdout[-30000:]


# ---------------------------------------------------------------------------
# PROJECT DETECTION
# ---------------------------------------------------------------------------

def detect_project(root: Path) -> tuple[str, str]:
    """
    Detect the project type and test command.

    Python projects can be detected from normal project metadata,
    a tests directory, or conventional root-level test files.

    This intentionally supports tiny repositories such as:

        calculator.py
        test_calculator.py
    """

    # -----------------------------
    # Python
    # -----------------------------

    python_markers = (
        root / "pyproject.toml",
        root / "pytest.ini",
        root / "tox.ini",
        root / "setup.py",
        root / "setup.cfg",
        root / "requirements.txt",
    )

    if any(path.exists() for path in python_markers):
        return "python", "pytest -q"

    if (root / "tests").is_dir():
        if any(root.glob("tests/test_*.py")):
            return "python", "pytest -q"

        if any(root.glob("tests/*_test.py")):
            return "python", "pytest -q"

    # Root-level conventional pytest files.
    if any(root.glob("test_*.py")):
        return "python", "pytest -q"

    if any(root.glob("*_test.py")):
        return "python", "pytest -q"

    # -----------------------------
    # Node
    # -----------------------------

    if (root / "package.json").exists():
        return "node", "npm test -- --runInBand"

    # -----------------------------
    # Java
    # -----------------------------

    if (root / "pom.xml").exists():
        return "java", "mvn test -q"

    # -----------------------------
    # Go
    # -----------------------------

    if (root / "go.mod").exists():
        return "go", "go test ./..."

    raise RuntimeError(
        "Could not identify a supported test project. "
        "Expected Python, Node, Java, or Go project markers."
    )


# ---------------------------------------------------------------------------
# GIT CHECKOUT
# ---------------------------------------------------------------------------

def checkout(
    repo_url: str,
    sha: str,
    destination: Path,
) -> None:
    """
    Clone the repository and check out the exact PR HEAD SHA.

    HEALFORGE verifies the exact revision inspected by the application,
    rather than whatever happens to be on the default branch.
    """

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
            f"git clone failed: {output}"
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
            f"git fetch failed: {output}"
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
            f"git checkout failed: {output}"
        )


# ---------------------------------------------------------------------------
# PATCH APPLICATION
# ---------------------------------------------------------------------------

def apply_patch(
    root: Path,
    patch_file: Path,
) -> tuple[bool, str]:
    """
    Validate the patch first, then apply it.

    A patch that cannot be cleanly applied is rejected before
    sandbox execution.
    """

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
# DOCKER HELPERS
# ---------------------------------------------------------------------------

def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _docker_image_exists(image: str) -> bool:
    """
    Check whether an image is already available locally.
    """

    code, _ = run_process(
        [
            "docker",
            "image",
            "inspect",
            image,
        ],
        Path.cwd(),
        30,
    )

    return code == 0


def _python_image_name(root: Path) -> str:
    """
    Generate a deterministic image name.

    Repositories with different requirements files receive different
    dependency images, while repositories without requirements share
    the standard pytest image.
    """

    requirements = root / "requirements.txt"

    if not requirements.exists():
        return "healforge-python:3.12-pytest"

    digest = hashlib.sha256(
        requirements.read_bytes()
    ).hexdigest()[:16]

    return f"healforge-python:3.12-pytest-{digest}"


def _node_image_name(root: Path) -> str:
    """
    Generate a deterministic Node dependency image name based on the
    package lock file when available.
    """

    lock_files = (
        root / "package-lock.json",
        root / "npm-shrinkwrap.json",
        root / "yarn.lock",
        root / "pnpm-lock.yaml",
    )

    lock_file = next(
        (path for path in lock_files if path.exists()),
        None,
    )

    if lock_file is None:
        return "healforge-node:22"

    digest = hashlib.sha256(
        lock_file.read_bytes()
    ).hexdigest()[:16]

    return f"healforge-node:22-{digest}"


# ---------------------------------------------------------------------------
# PYTHON SANDBOX IMAGE
# ---------------------------------------------------------------------------

def _ensure_python_image(
    root: Path,
) -> tuple[bool, str]:
    """
    Prepare a Python sandbox image.

    Network access is used ONLY while building the image.

    The eventual test container is always executed with:

        --network none

    This avoids the previous failure where pytest was installed from
    PyPI inside a network-isolated container.
    """

    image = _python_image_name(root)

    if _docker_image_exists(image):
        return True, image

    requirements = root / "requirements.txt"

    with tempfile.TemporaryDirectory(
        prefix="healforge-python-image-"
    ) as temp_dir:

        build_dir = Path(temp_dir)

        dockerfile = build_dir / "Dockerfile"

        if requirements.exists():

            shutil.copy2(
                requirements,
                build_dir / "requirements.txt",
            )

            dockerfile.write_text(
                """
FROM python:3.12-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/healforge

COPY requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt \\
    && python -m pip install --no-cache-dir pytest

WORKDIR /work
""".strip()
                + "\n",
                encoding="utf-8",
            )

        else:

            dockerfile.write_text(
                """
FROM python:3.12-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN python -m pip install --no-cache-dir pytest

WORKDIR /work
""".strip()
                + "\n",
                encoding="utf-8",
            )

        code, output = run_process(
            [
                "docker",
                "build",
                "--tag",
                image,
                str(build_dir),
            ],
            build_dir,
            300,
        )

        if code != 0:
            return False, (
                "Failed to prepare Python sandbox image.\n\n"
                + output
            )

    return True, image


# ---------------------------------------------------------------------------
# NODE SANDBOX IMAGE
# ---------------------------------------------------------------------------

def _ensure_node_image(
    root: Path,
) -> tuple[bool, str]:
    """
    Prepare a Node sandbox image.

    Dependencies are installed during image creation, when network
    access is permitted.

    Runtime verification remains completely network-isolated.
    """

    image = _node_image_name(root)

    if _docker_image_exists(image):
        return True, image

    lock_files = (
        root / "package-lock.json",
        root / "npm-shrinkwrap.json",
        root / "yarn.lock",
        root / "pnpm-lock.yaml",
    )

    package_json = root / "package.json"

    with tempfile.TemporaryDirectory(
        prefix="healforge-node-image-"
    ) as temp_dir:

        build_dir = Path(temp_dir)

        shutil.copy2(
            package_json,
            build_dir / "package.json",
        )

        lock_file = next(
            (
                path
                for path in lock_files
                if path.exists()
            ),
            None,
        )

        if lock_file is not None:
            shutil.copy2(
                lock_file,
                build_dir / lock_file.name,
            )

        if (
            lock_file is not None
            and lock_file.name
            in {
                "package-lock.json",
                "npm-shrinkwrap.json",
            }
        ):
            install_command = (
                "npm ci --ignore-scripts"
            )

        elif lock_file is not None and lock_file.name == "yarn.lock":
            install_command = (
                "corepack enable && yarn install --frozen-lockfile"
            )

        elif lock_file is not None and lock_file.name == "pnpm-lock.yaml":
            install_command = (
                "corepack enable && "
                "corepack prepare pnpm@latest --activate && "
                "pnpm install --frozen-lockfile"
            )

        else:
            install_command = "npm install --ignore-scripts"

        dockerfile = build_dir / "Dockerfile"

        dockerfile.write_text(
            f"""
FROM node:22-bookworm-slim

WORKDIR /opt/healforge

COPY package*.json ./
COPY yarn.lock* ./
COPY pnpm-lock.yaml* ./

RUN {install_command}

WORKDIR /work
""".strip()
            + "\n",
            encoding="utf-8",
        )

        code, output = run_process(
            [
                "docker",
                "build",
                "--tag",
                image,
                str(build_dir),
            ],
            build_dir,
            300,
        )

        if code != 0:
            return False, (
                "Failed to prepare Node sandbox image.\n\n"
                + output
            )

    return True, image


# ---------------------------------------------------------------------------
# PYTHON SANDBOX
# ---------------------------------------------------------------------------

def _run_python_sandbox(
    root: Path,
    image: str,
) -> RunResult:

    command = "pytest -q"

    args = [
        "docker",
        "run",
        "--rm",

        # No network access during verification.
        "--network",
        "none",

        # Resource limits.
        "--cpus",
        "1.5",
        "--memory",
        "768m",
        "--pids-limit",
        "128",

        # Keep the container filesystem immutable.
        "--read-only",

        # Pytest/Python may need temporary files.
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=256m",

        # Only the repository under test is writable.
        "-v",
        f"{root.resolve()}:/work:rw",

        "-w",
        "/work",

        image,

        "pytest",
        "-q",
    ]

    try:
        code, output = run_process(
            args,
            root,
            180,
        )
    except Exception as exc:
        return RunResult(
            passed=False,
            command=command,
            exit_code=1,
            output=str(exc),
        )

    return RunResult(
        passed=code == 0,
        command=command,
        exit_code=code,
        output=output,
    )


# ---------------------------------------------------------------------------
# NODE SANDBOX
# ---------------------------------------------------------------------------

def _run_node_sandbox(
    root: Path,
    image: str,
) -> RunResult:

    command = "npm test -- --runInBand"

    args = [
        "docker",
        "run",
        "--rm",

        # No network access during verification.
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

        # Patched repository.
        "-v",
        f"{root.resolve()}:/work:rw",

        "-w",
        "/work",

        image,

        "npm",
        "test",
        "--",
        "--runInBand",
    ]

    try:
        code, output = run_process(
            args,
            root,
            180,
        )
    except Exception as exc:
        return RunResult(
            passed=False,
            command=command,
            exit_code=1,
            output=str(exc),
        )

    return RunResult(
        passed=code == 0,
        command=command,
        exit_code=code,
        output=output,
    )


# ---------------------------------------------------------------------------
# MAIN SANDBOX ENTRY POINT
# ---------------------------------------------------------------------------

def docker_test(
    root: Path,
    language: str,
    command: str,
) -> RunResult:
    """
    Verify the repaired repository inside a restricted Docker sandbox.

    Dependency installation happens during image preparation.

    Actual verification happens with network access disabled.
    """

    if not _docker_available():
        raise RuntimeError(
            "Docker is required for sandbox verification. "
            "Install and start Docker Desktop, then retry."
        )

    if not root.exists():
        raise RuntimeError(
            f"Sandbox repository does not exist: {root}"
        )

    if language == "python":

        ready, image_or_error = _ensure_python_image(
            root
        )

        if not ready:
            return RunResult(
                passed=False,
                command=command,
                exit_code=1,
                output=image_or_error,
            )

        return _run_python_sandbox(
            root,
            image_or_error,
        )

    if language == "node":

        ready, image_or_error = _ensure_node_image(
            root
        )

        if not ready:
            return RunResult(
                passed=False,
                command=command,
                exit_code=1,
                output=image_or_error,
            )

        return _run_node_sandbox(
            root,
            image_or_error,
        )

    raise RuntimeError(
        "Sandbox execution currently supports "
        "Python and Node projects. "
        f"Detected language: {language}"
    )