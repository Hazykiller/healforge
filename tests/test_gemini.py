import json
import pytest
from unittest.mock import MagicMock, patch
import httpx

from app.ai import AIEngine, AIQuotaExhaustedError, _redact_secrets
from app.config import Settings


def test_gemini_provider_initialization_success(monkeypatch):
    """Test 1: Gemini provider initializes cleanly when GEMINI_API_KEY is configured."""
    fake_settings = Settings(
        ai_provider="gemini",
        gemini_api_key="fake-test-gemini-key-12345",
        gemini_model="gemini-3.5-flash",
    )
    monkeypatch.setattr("app.ai.settings", fake_settings)
    engine = AIEngine(provider="gemini")
    assert engine.provider == "gemini"


def test_gemini_provider_initialization_missing_key(monkeypatch):
    """Test 2: Missing Gemini API key raises RuntimeError on initialization."""
    fake_settings = Settings(
        ai_provider="gemini",
        gemini_api_key="",
    )
    monkeypatch.setattr("app.ai.settings", fake_settings)
    with pytest.raises(RuntimeError) as exc_info:
        AIEngine(provider="gemini")
    assert "GEMINI_API_KEY is not configured" in str(exc_info.value)


def test_provider_selection_gemini_vs_openrouter(monkeypatch):
    """Test 3: Provider selection switches between Gemini and OpenRouter."""
    fake_settings_gemini = Settings(
        ai_provider="gemini",
        gemini_api_key="fake-key",
        openrouter_api_key="fake-openrouter",
    )
    monkeypatch.setattr("app.ai.settings", fake_settings_gemini)
    engine_gemini = AIEngine()
    assert engine_gemini.provider == "gemini"

    fake_settings_openrouter = Settings(
        ai_provider="openrouter",
        gemini_api_key="fake-key",
        openrouter_api_key="fake-openrouter",
    )
    monkeypatch.setattr("app.ai.settings", fake_settings_openrouter)
    with patch("app.ai.OpenAI"):
        engine_openrouter = AIEngine()
        assert engine_openrouter.provider == "openrouter"


def test_gemini_malformed_response_handling(monkeypatch):
    """Test 4: Malformed response from Gemini raises appropriate error."""
    fake_settings = Settings(
        ai_provider="gemini",
        gemini_api_key="fake-key",
        gemini_model="gemini-3.5-flash",
    )
    monkeypatch.setattr("app.ai.settings", fake_settings)
    engine = AIEngine(provider="gemini")

    # Mock httpx response returning non-JSON garbage
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "not a valid json response at all"}]}}]
    }

    with patch("httpx.Client.post", return_value=mock_resp):
        with pytest.raises(RuntimeError):
            engine.diagnose("test context")


def test_gemini_fenced_json_response_parsing(monkeypatch):
    """Test 5: Fenced markdown JSON response is cleanly extracted and normalized."""
    fake_settings = Settings(
        ai_provider="gemini",
        gemini_api_key="fake-key",
        gemini_model="gemini-3.5-flash",
    )
    monkeypatch.setattr("app.ai.settings", fake_settings)
    engine = AIEngine(provider="gemini")

    fenced_content = """```json
{
  "summary": "Fix index error",
  "root_cause": "Off by one in loop limit",
  "confidence": 0.95,
  "affected_files": ["math_lib.py"],
  "evidence": ["test_index failed"],
  "repair_strategy": "Change range upper bound",
  "risk_notes": []
}
```"""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": fenced_content}]}}]
    }

    with patch("httpx.Client.post", return_value=mock_resp):
        diagnosis = engine.diagnose("test evidence")
        assert diagnosis["summary"] == "Fix index error"
        assert diagnosis["root_cause"] == "Off by one in loop limit"
        assert diagnosis["confidence"] == 0.95
        assert diagnosis["affected_files"] == ["math_lib.py"]


def test_gemini_valid_structured_repair_response(monkeypatch):
    """Test 6: Valid structured semantic repair response from Gemini is accepted."""
    fake_settings = Settings(
        ai_provider="gemini",
        gemini_api_key="fake-key",
        gemini_model="gemini-3.5-flash",
    )
    monkeypatch.setattr("app.ai.settings", fake_settings)
    engine = AIEngine(provider="gemini")

    repair_json = {
        "summary": "Fix off-by-one bug",
        "edits": [
            {
                "file": "calc.py",
                "old_text": "return a - b",
                "new_text": "return a + b",
                "occurrence": 1,
            }
        ],
        "confidence": 0.98,
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": json.dumps(repair_json)}]}}]
    }

    contents = {"calc.py": "def add(a, b):\n    return a - b\n"}
    diagnosis = {
        "summary": "Subtraction used instead of addition",
        "root_cause": "Typo in operator",
        "affected_files": ["calc.py"],
    }

    with patch("httpx.Client.post", return_value=mock_resp):
        res = engine.generate_patch(
            context="test context",
            diagnosis=diagnosis,
            contents=contents,
        )
        assert res["summary"] == "Fix off-by-one bug"
        assert len(res["edits"]) == 1
        assert "--- a/calc.py" in res["patch"]
        assert "+    return a + b" in res["patch"]


def test_gemini_semantic_edit_validation_rejects_test_modification(monkeypatch):
    """Test 7: Semantic edit validation strictly rejects repair attempts that touch test files."""
    fake_settings = Settings(
        ai_provider="gemini",
        gemini_api_key="fake-key",
        gemini_model="gemini-3.5-flash",
    )
    monkeypatch.setattr("app.ai.settings", fake_settings)
    engine = AIEngine(provider="gemini")

    illegal_edit = {
        "summary": "Modify test to make it pass",
        "edits": [
            {
                "file": "tests/test_calc.py",
                "old_text": "assert add(1, 2) == 3",
                "new_text": "assert True",
                "occurrence": 1,
            }
        ],
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": json.dumps(illegal_edit)}]}}]
    }

    contents = {"tests/test_calc.py": "def test_add():\n    assert add(1, 2) == 3\n"}
    diagnosis = {"summary": "bug", "root_cause": "bug", "affected_files": ["tests/test_calc.py"]}

    with patch("httpx.Client.post", return_value=mock_resp):
        with pytest.raises(RuntimeError) as exc_info:
            engine.generate_patch("context", diagnosis, contents)
        assert "test file" in str(exc_info.value).lower() or "unusable responses" in str(exc_info.value).lower()


def test_gemini_and_openrouter_share_same_deterministic_patch_pipeline():
    """Test 8 & 10: Downstream patch generation is deterministic and provider-independent."""
    edits = [
        {
            "file": "service.py",
            "old_text": "    return x * 2",
            "new_text": "    return x ** 2",
            "occurrence": 1,
        }
    ]
    contents = {
        "service.py": "def power_two(x):\n    return x * 2\n"
    }

    # Verify deterministic patch generation regardless of which provider generated the edits
    engine = object.__new__(AIEngine)
    patch_str, touched = engine._apply_edits(edits, contents)
    assert "--- a/service.py" in patch_str
    assert "+++ b/service.py" in patch_str
    assert "-    return x * 2" in patch_str
    assert "+    return x ** 2" in patch_str
    assert touched == ["service.py"]


def test_secret_redaction_guarantees_no_credentials_leaked():
    """Test 11: Ensure secrets, keys, and authorization tokens are strictly redacted."""
    sensitive_key = "AIzaSySecretGeminiKey1234567890"
    fake_settings = Settings(gemini_api_key=sensitive_key, openrouter_api_key="sk-or-secret987654321")
    
    with patch("app.ai.settings", fake_settings):
        sample_error = f"HTTP 403: Forbidden for key {sensitive_key} with header Bearer sk-or-secret987654321"
        cleaned = _redact_secrets(sample_error)
        assert sensitive_key not in cleaned
        assert "sk-or-secret987654321" not in cleaned
        assert "[REDACTED" in cleaned
