import re
from pathlib import PurePosixPath

_SECRET_FILE_NAMES = frozenset({
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    ".npmrc",
    ".pypirc",
    ".netrc",
    "credentials.json",
    "credentials",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
})

_SECRET_EXTENSIONS = frozenset({
    ".pem", ".key", ".p12", ".pfx", ".jks", ".keystore", ".crt",
})

_PRIVATE_PREFIXES = ("id_rsa", "id_ed25519", "id_ecdsa", "private_key")


def normalize_repo_path(path: str) -> str:
    value = str(path).replace("\\", "/").strip()
    while value.startswith("./"):
        value = value[2:]
    return value


def is_safe_repo_path(path: str) -> bool:
    normalized = normalize_repo_path(path)
    if not normalized or normalized.startswith("/"):
        return False
    if "\x00" in normalized:
        return False
    if re.match(r"^[A-Za-z]:", normalized):
        return False
    parts = PurePosixPath(normalized).parts
    return ".." not in parts and "" not in parts


def is_sensitive_path(path: str) -> bool:
    normalized = normalize_repo_path(path)
    name = PurePosixPath(normalized).name.lower()

    if name in _SECRET_FILE_NAMES:
        return True
    if PurePosixPath(name).suffix.lower() in _SECRET_EXTENSIONS:
        return True
    if name.startswith(".env") and name != ".env.example":
        return True
    if name.startswith(_PRIVATE_PREFIXES):
        return True
    return False


def validate_patch_paths(paths: list[str]) -> list[str]:
    safe: list[str] = []
    for path in paths:
        normalized = normalize_repo_path(path)
        if not is_safe_repo_path(normalized):
            raise ValueError(f"Unsafe patch path rejected: {path}")
        if is_sensitive_path(normalized):
            raise ValueError(f"Sensitive file patch rejected: {normalized}")
        if normalized not in safe:
            safe.append(normalized)
    return safe
