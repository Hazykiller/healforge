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


def test_blocks_additional_credential_files():
    assert is_sensitive_path("config/.npmrc")
    assert is_sensitive_path("config/.pypirc")
    assert is_sensitive_path("config/.netrc")
