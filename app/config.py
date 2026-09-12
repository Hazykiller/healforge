import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "60000"))
    max_patch_chars: int = int(os.getenv("MAX_PATCH_CHARS", "20000"))
    docker_image_python: str = os.getenv("DOCKER_IMAGE_PYTHON", "python:3.12-slim")
    docker_image_node: str = os.getenv("DOCKER_IMAGE_NODE", "node:22-bookworm-slim")
    allow_write_actions: bool = _env_bool("ALLOW_WRITE_ACTIONS")
    host_workspace: str = os.getenv("HOST_WORKSPACE", "./workspace")


settings = Settings()
