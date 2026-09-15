from types import SimpleNamespace

import pytest

from app.ai import AIEngine


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    def __init__(self, responses):
        self.completions = FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


def response(content):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content)
            )
        ]
    )


def test_diagnosis_normalizes_scalar_arrays(monkeypatch):
    engine = object.__new__(AIEngine)
    engine.client = FakeClient([
        response(
            '{"summary":"bad operator","root_cause":"minus used","confidence":"0.9",'
            '"affected_files":"calculator.py","evidence":"test failure",'
            '"repair_strategy":"replace operator","risk_notes":"low risk"}'
        )
    ])

    monkeypatch.setattr(
        "app.ai.settings",
        type("TestSettings", (), {"ai_models": ["primary"], "max_patch_chars": 30000})(),
    )

    result = engine.diagnose("evidence")

    assert result["evidence"] == ["test failure"]
    assert result["affected_files"] == ["calculator.py"]
    assert result["risk_notes"] == ["low risk"]
    assert result["confidence"] == 0.9


def test_model_failure_retries_the_openrouter_fallback_route(monkeypatch):
    engine = object.__new__(AIEngine)
    engine.client = FakeClient([
        RuntimeError("provider unavailable"),
        response(
            '{"summary":"ok","root_cause":"bug","confidence":0.8,'
            '"affected_files":[],"evidence":[],"repair_strategy":"fix",' 
            '"risk_notes":[]}'
        ),
    ])

    monkeypatch.setattr(
        "app.ai.settings",
        type("TestSettings", (), {"ai_models": ["primary", "fallback"], "max_patch_chars": 30000})(),
    )

    result = engine.diagnose("evidence")

    assert result["summary"] == "ok"
    assert len(engine.client.completions.calls) == 2
    assert engine.client.completions.calls[0]["model"] == "primary"
    assert engine.client.completions.calls[1]["model"] == "primary"
    assert engine.client.completions.calls[0]["extra_body"]["models"] == ["primary", "fallback"]


def test_empty_choices_retry_the_openrouter_fallback_route(monkeypatch):
    engine = object.__new__(AIEngine)
    empty = SimpleNamespace(choices=[])
    engine.client = FakeClient([
        empty,
        response(
            '{"summary":"ok","root_cause":"bug","confidence":0.8,'
            '"affected_files":[],"evidence":[],"repair_strategy":"fix",'
            '"risk_notes":[]}'
        ),
    ])

    monkeypatch.setattr(
        "app.ai.settings",
        type("TestSettings", (), {"ai_models": ["primary", "fallback"], "max_patch_chars": 30000})(),
    )

    result = engine.diagnose("evidence")
    assert result["summary"] == "ok"


def test_patch_path_validation_blocks_sensitive_files():
    with pytest.raises(RuntimeError):
        AIEngine._validate_patch({
            "patch": "--- a/.env\n+++ b/.env\n@@ -1 +1 @@\n-x\n+y\n",
            "touched_files": [".env"],
            "confidence": 1,
        })

    with pytest.raises(RuntimeError):
        AIEngine._validate_patch({
            "patch": "--- a/../secret.txt\n+++ b/../secret.txt\n@@ -1 +1 @@\n-x\n+y\n",
            "touched_files": ["../secret.txt"],
            "confidence": 1,
        })


def test_repair_parser_accepts_json_diff(monkeypatch):
    engine = object.__new__(AIEngine)
    engine.client = FakeClient([
        response(
            '{"patch":"--- a/calculator.py\\n+++ b/calculator.py\\n@@ -1 +1 @@\\n-a\\n+b\\n",'
            '"explanation":"fix","touched_files":["calculator.py"],"confidence":0.95}'
        )
    ])
    monkeypatch.setattr(
        "app.ai.settings",
        type("TestSettings", (), {"ai_models": ["primary"], "max_patch_chars": 30000})(),
    )
    result = engine.generate_patch("FILE: calculator.py", {"summary": "x"})
    assert result["touched_files"] == ["calculator.py"]
    assert result["patch"].startswith("--- a/calculator.py")
