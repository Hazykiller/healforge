import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {
        "1", "true", "yes", "on"
    }


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_base_url: str = os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    )
    openrouter_model: str = os.getenv(
        "OPENROUTER_MODEL",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
    )
    openrouter_fallback_models: str = os.getenv(
        "OPENROUTER_FALLBACK_MODELS",
        "qwen/qwen3-coder:free,openrouter/free",
    )

    max_context_chars: int = _env_int("MAX_CONTEXT_CHARS", 60000)
    max_patch_chars: int = _env_int("MAX_PATCH_CHARS", 30000)
    max_repo_candidates: int = _env_int("MAX_REPO_CANDIDATES", 60)
    max_file_chars: int = _env_int("MAX_FILE_CHARS", 18000)
    github_timeout_seconds: int = _env_int("GITHUB_TIMEOUT_SECONDS", 30)
    ai_timeout_seconds: int = _env_int("AI_TIMEOUT_SECONDS", 120)
    sandbox_timeout_seconds: int = _env_int("SANDBOX_TIMEOUT_SECONDS", 600)
    github_concurrency: int = _env_int("GITHUB_CONCURRENCY", 8)
    docker_pull_images: bool = _env_bool("DOCKER_PULL_IMAGES", False)
    max_sandbox_file_bytes: int = _env_int("MAX_SANDBOX_FILE_BYTES", 10_000_000)
    max_sandbox_context_bytes: int = _env_int("MAX_SANDBOX_CONTEXT_BYTES", 100_000_000)
    max_repair_attempts: int = _env_int("MAX_REPAIR_ATTEMPTS", 2)

    docker_image_python: str = os.getenv(
        "DOCKER_IMAGE_PYTHON",
        "python:3.12-slim",
    )
    docker_image_node: str = os.getenv(
        "DOCKER_IMAGE_NODE",
        "node:22-bookworm-slim",
    )

    allow_write_actions: bool = _env_bool("ALLOW_WRITE_ACTIONS")
    host_workspace: str = os.getenv("HOST_WORKSPACE", "./workspace")

    @property
    def ai_models(self) -> list[str]:
        values = [self.openrouter_model]
        values.extend(
            item.strip()
            for item in self.openrouter_fallback_models.split(",")
            if item.strip()
        )
        return list(dict.fromkeys(values))


settings = Settings()
