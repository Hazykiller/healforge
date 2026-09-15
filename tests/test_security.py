import pytest

from app.security import is_sensitive_path, is_safe_repo_path, validate_patch_paths


def test_rejects_path_traversal():
    assert not is_safe_repo_path("../outside.py")
    assert not is_safe_repo_path("/absolute.py")
    assert not is_safe_repo_path("C:/absolute.py")


def test_blocks_sensitive_files():
    assert is_sensitive_path(".env")
    assert is_sensitive_path("config/private.pem")
    assert is_sensitive_path("certs/client.p12")
    assert is_sensitive_path("secrets/private_key_backup")
    assert is_sensitive_path("keys/id_ecdsa_work")
    assert is_sensitive_path(".npmrc")
    assert is_sensitive_path("config/credentials.json")
    assert is_sensitive_path("secrets/private_key_backup")
    assert not is_sensitive_path(".env.example")


def test_validate_patch_paths_rejects_secrets_and_traversal():
    with pytest.raises(ValueError):
        validate_patch_paths(["../outside.py"])
    with pytest.raises(ValueError):
        validate_patch_paths([".env"])


def test_validate_patch_paths_normalizes_safe_paths():
    assert validate_patch_paths(["./src/main.py", "src/main.py"]) == ["src/main.py"]


def test_docker_build_context_excludes_environment_secrets(tmp_path):
    from app.runner import _copy_build_context

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "app.py").write_text("print(1)")
    (source / ".env").write_text("SECRET=do-not-copy")
    (source / "private.pem").write_text("PRIVATE KEY")
    (source / "signing.key").write_text("PRIVATE KEY")
    (source / ".env.example").write_text("SECRET=")

    _copy_build_context(source, destination)

    assert (destination / "app.py").exists()
    assert not (destination / ".env").exists()
    assert not (destination / "private.pem").exists()
    assert not (destination / "signing.key").exists()
    assert (destination / ".env.example").exists()


def test_rejects_windows_drive_paths():
    assert not is_safe_repo_path("D:\\secrets\\key.pem")
    assert not is_safe_repo_path("C:/Users/admin/.ssh/id_rsa")


def test_rejects_null_and_empty_paths():
    assert not is_safe_repo_path("")
    assert not is_safe_repo_path("   ")
    # Path with null byte embedded
    assert not is_safe_repo_path("src/app\x00.py")


def test_sensitive_path_blocks_all_env_variants():
    """All .env.* variants except .env.example must be blocked."""
    assert is_sensitive_path(".env")
    assert is_sensitive_path(".env.local")
    assert is_sensitive_path(".env.staging")
    assert is_sensitive_path("config/.env.production")
    assert not is_sensitive_path(".env.example")
