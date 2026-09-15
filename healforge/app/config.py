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
    ai_provider: str = os.getenv("AI_PROVIDER", "gemini").strip().lower()

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_base_url: str = os.getenv(
        "GEMINI_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta",
    )
    gemini_model: str = os.getenv(
        "GEMINI_MODEL",
        "gemini-3.5-flash",
    )
    gemini_fallback_models: str = os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-3.1-flash-lite,gemini-3.7-flash",
    )

    github_token: str = os.getenv("GITHUB_TOKEN", "")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_base_url: str = os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    )
    openrouter_model: str = os.getenv(
        "OPENROUTER_MODEL",
        "openrouter/free",
    )
    openrouter_fallback_models: str = os.getenv(
        "OPENROUTER_FALLBACK_MODELS",
        "cohere/north-mini-code:free,liquid/lfm-2.5-2.6b:free,openrouter/free",
    )

    max_context_chars: int = _env_int("MAX_CONTEXT_CHARS", 40000)
    max_patch_chars: int = _env_int("MAX_PATCH_CHARS", 50000)
    max_repo_candidates: int = _env_int("MAX_REPO_CANDIDATES", 60)
    max_file_chars: int = _env_int("MAX_FILE_CHARS", 12000)
    github_timeout_seconds: int = _env_int("GITHUB_TIMEOUT_SECONDS", 30)
    ai_timeout_seconds: int = min(_env_int("AI_TIMEOUT_SECONDS", 60), 75)
    ai_request_timeout_seconds: int = min(_env_int("AI_REQUEST_TIMEOUT_SECONDS", 45), 75)
    sandbox_timeout_seconds: int = _env_int("SANDBOX_TIMEOUT_SECONDS", 600)
    github_concurrency: int = _env_int("GITHUB_CONCURRENCY", 8)
    docker_pull_images: bool = _env_bool("DOCKER_PULL_IMAGES", False)
    max_sandbox_file_bytes: int = _env_int("MAX_SANDBOX_FILE_BYTES", 10_000_000)
    max_sandbox_context_bytes: int = _env_int("MAX_SANDBOX_CONTEXT_BYTES", 100_000_000)
    max_repair_attempts: int = _env_int("MAX_REPAIR_ATTEMPTS", 3)

    docker_image_python: str = os.getenv(
        "DOCKER_IMAGE_PYTHON",
        "python:3.10-slim",
    )
    docker_image_node: str = os.getenv(
        "DOCKER_IMAGE_NODE",
        "node:22-bookworm-slim",
    )

    allow_write_actions: bool = _env_bool("ALLOW_WRITE_ACTIONS")
    host_workspace: str = os.getenv("HOST_WORKSPACE", "./workspace")

    sandbox_provider: str = os.getenv("SANDBOX_PROVIDER", "auto").strip().lower()
    sandbox_url: str = os.getenv("SANDBOX_URL", "").strip()
    sandbox_token: str = os.getenv("SANDBOX_TOKEN", "").strip()
    is_vercel: bool = bool(os.getenv("VERCEL"))

    @property
    def verifier_type(self) -> str:
        from .runner import _docker_available
        mode = self.sandbox_provider
        if mode == "remote":
            return "remote" if self.sandbox_url else "none"
        if mode == "docker":
            return "docker" if _docker_available() else "none"
        if self.is_vercel:
            return "remote" if self.sandbox_url else "none"
        if _docker_available():
            return "docker"
        return "remote" if self.sandbox_url else "none"

    @property
    def verifier_status(self) -> str:
        vt = self.verifier_type
        if vt == "docker":
            return "LOCAL_DOCKER_AVAILABLE"
        if vt == "remote":
            return "REMOTE_SANDBOX_CONFIGURED"
        return "SANDBOX_UNAVAILABLE"

    @property
    def is_ai_configured(self) -> bool:
        if self.ai_provider == "gemini":
            return bool(self.gemini_api_key)
        return bool(self.openrouter_api_key)

    @property
    def ai_models(self) -> list[str]:
        if self.ai_provider == "gemini":
            models_env = os.getenv("GEMINI_MODELS", "").strip()
            if models_env:
                raw_list = [m.strip() for m in models_env.split(",") if m.strip()]
            else:
                primary = self.gemini_model.strip()
                fallbacks = [m.strip() for m in self.gemini_fallback_models.split(",") if m.strip()]
                raw_list = [primary] + fallbacks
            return list(dict.fromkeys(raw_list)) or ["gemini-3.5-flash"]

        # OpenRouter model chain
        models_env = os.getenv("OPENROUTER_MODELS", "").strip()
        if models_env:
            raw_list = [m.strip() for m in models_env.split(",") if m.strip()]
        else:
            primary = self.openrouter_model.strip()
            fallbacks = [m.strip() for m in self.openrouter_fallback_models.split(",") if m.strip()]
            raw_list = [primary] + fallbacks

        if "openrouter/free" not in raw_list:
            raw_list.append("openrouter/free")

        # Filter out known dead or invalid model IDs
        broken = {"qwen/qwen3-coder:free", "nvidia/nemotron-3.5-content-safety:free"}
        cleaned = [m for m in raw_list if m and m not in broken]

        # Fast coding & lightweight models prioritized ahead of giant 550b models
        def _speed_priority(model_name: str) -> int:
            m = model_name.lower()
            if "mini-code" in m or "cohere" in m:
                return 0
            if "lfm" in m or "liquid" in m:
                return 1
            if "openrouter/free" in m:
                return 2
            if "coder" in m:
                return 3
            if "mini" in m or "micro" in m or "nano" in m:
                return 4
            if "ultra" in m or "550b" in m:
                return 10
            return 5

        cleaned.sort(key=_speed_priority)

        # Deduplicate preserving order
        result = list(dict.fromkeys(cleaned))
        return result or ["openrouter/free"]


settings = Settings()
