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


def test_model_fallback_chain_is_sent_once_to_openrouter(monkeypatch):
    engine = object.__new__(AIEngine)
    engine.client = FakeClient([
        response(
            '{"summary":"ok","root_cause":"bug","confidence":0.8,'
            '"affected_files":[],"evidence":[],"repair_strategy":"fix",'
            '"risk_notes":[]}'
        ),
    ])

    monkeypatch.setattr(
        "app.ai.settings",
        type(
            "TestSettings",
            (),
            {"ai_models": ["primary", "fallback"], "max_patch_chars": 30000},
        )(),
    )

    result = engine.diagnose("evidence")

    assert result["summary"] == "ok"
    assert len(engine.client.completions.calls) == 1
    request = engine.client.completions.calls[0]
    assert request["model"] == "primary"
    assert request["extra_body"]["models"] == ["primary", "fallback"]


def test_empty_choices_fails_over(monkeypatch):
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


def test_repair_parser_accepts_structured_edits(monkeypatch):
    engine = object.__new__(AIEngine)
    engine.client = FakeClient([
        response(
            '{"edits":[{"file":"calculator.py","old_text":"a","new_text":"b","occurrence":1}],'
            '"explanation":"fix","confidence":0.95}'
        )
    ])
    monkeypatch.setattr(
        "app.ai.settings",
        type("TestSettings", (), {"ai_models": ["primary"], "max_patch_chars": 30000})(),
    )
    
    contents = {"calculator.py": "a\n"}
    
    result = engine.generate_patch("FILE: calculator.py", {"summary": "x"}, contents)
    assert result["touched_files"] == ["calculator.py"]
    assert "--- a/calculator.py" in result["patch"]


def test_repair_parser_handles_malformed_json(monkeypatch):
    engine = object.__new__(AIEngine)
    
    # 1. JSON in markdown fences
    engine.client = FakeClient([
        response(
            '''```json\n{"edits":[{"file":"calculator.py","old_text":"a","new_text":"b"}],"explanation":"fix","confidence":0.95}\n```'''
        )
    ])
    monkeypatch.setattr("app.ai.settings", type("TestSettings", (), {"ai_models": ["primary"], "max_patch_chars": 30000})())
    result = engine.generate_patch("FILE: calculator.py", {"summary": "x"}, {"calculator.py": "a\n"})
    assert result["touched_files"] == ["calculator.py"]
    
    # 2. JSON surrounded by explanation
    engine.client = FakeClient([
        response(
            '''Here is the fix:\n{"edits":[{"file":"calculator.py","old_text":"a","new_text":"c"}],"explanation":"fix","confidence":0.95}\nHope this helps!'''
        )
    ])
    result = engine.generate_patch("FILE: calculator.py", {"summary": "x"}, {"calculator.py": "a\n"})
    assert result["touched_files"] == ["calculator.py"]
    assert "+c" in result["patch"]

def test_repair_parser_fails_missing_old_text(monkeypatch):
    engine = object.__new__(AIEngine)
    engine.client = FakeClient([
        response(
            '''{"edits":[{"file":"calculator.py","new_text":"b"}],"explanation":"fix","confidence":0.95}'''
        )
    ])
    monkeypatch.setattr("app.ai.settings", type("TestSettings", (), {"ai_models": ["primary"], "max_patch_chars": 30000})())
    
    with pytest.raises(RuntimeError, match="missing 'old_text'"):
        engine._parse_repair_response(
            '''{"edits":[{"file":"calculator.py","new_text":"b"}],"explanation":"fix","confidence":0.95}''',
            {"calculator.py": "a\n"}
        )

def test_repair_parser_fails_unknown_file(monkeypatch):
    engine = object.__new__(AIEngine)
    with pytest.raises(RuntimeError, match="unknown file"):
        engine._parse_repair_response(
            '''{"edits":[{"file":"nonexistent.py","old_text":"a","new_text":"b"}]}''',
            {"calculator.py": "a\n"}
        )

def test_repair_parser_fails_old_text_not_found(monkeypatch):
    engine = object.__new__(AIEngine)
    with pytest.raises(RuntimeError, match="not found in calculator.py"):
        engine._parse_repair_response(
            '''{"edits":[{"file":"calculator.py","old_text":"xyz","new_text":"b"}]}''',
            {"calculator.py": "a\n"}
        )

def test_repair_parser_fails_multiple_occurrences_without_index(monkeypatch):
    engine = object.__new__(AIEngine)
    with pytest.raises(RuntimeError, match="Specify 'occurrence'"):
        engine._parse_repair_response(
            '''{"edits":[{"file":"calculator.py","old_text":"a","new_text":"b"}]}''',
            {"calculator.py": "a\na\n"}
        )

def test_repair_parser_handles_multi_file_repair(monkeypatch):
    engine = object.__new__(AIEngine)
    engine.client = FakeClient([
        response(
            '''{"edits":[
                {"file":"fileA.py","old_text":"contractA","new_text":"contractB"},
                {"file":"fileB.py","old_text":"useA","new_text":"useB"}
            ],"explanation":"fix","confidence":0.95}'''
        )
    ])
    monkeypatch.setattr("app.ai.settings", type("TestSettings", (), {"ai_models": ["primary"], "max_patch_chars": 30000})())
    contents = {
        "fileA.py": "def contractA(): pass\n",
        "fileB.py": "def run(): useA()\n"
    }


def test_repair_parser_handles_malformed_json(monkeypatch):
    engine = object.__new__(AIEngine)
    
    # 1. JSON in markdown fences
    engine.client = FakeClient([
        response(
            '''```json\n{"edits":[{"file":"calculator.py","old_text":"a","new_text":"b"}],"explanation":"fix","confidence":0.95}\n```'''
        )
    ])
    monkeypatch.setattr("app.ai.settings", type("TestSettings", (), {"ai_models": ["primary"], "max_patch_chars": 30000})())
    result = engine.generate_patch("FILE: calculator.py", {"summary": "x"}, {"calculator.py": "a\n"})
    assert result["touched_files"] == ["calculator.py"]
    
    # 2. JSON surrounded by explanation
    engine.client = FakeClient([
        response(
            '''Here is the fix:\n{"edits":[{"file":"calculator.py","old_text":"a","new_text":"c"}],"explanation":"fix","confidence":0.95}\nHope this helps!'''
        )
    ])
    result = engine.generate_patch("FILE: calculator.py", {"summary": "x"}, {"calculator.py": "a\n"})
    assert result["touched_files"] == ["calculator.py"]
    assert "+c" in result["patch"]

def test_repair_parser_fails_missing_old_text(monkeypatch):
    engine = object.__new__(AIEngine)
    engine.client = FakeClient([
        response(
            '''{"edits":[{"file":"calculator.py","new_text":"b"}],"explanation":"fix","confidence":0.95}'''
        )
    ])
    monkeypatch.setattr("app.ai.settings", type("TestSettings", (), {"ai_models": ["primary"], "max_patch_chars": 30000})())
    
    with pytest.raises(RuntimeError, match="missing 'old_text'"):
        engine._parse_repair_response(
            '''{"edits":[{"file":"calculator.py","new_text":"b"}],"explanation":"fix","confidence":0.95}''',
            {"calculator.py": "a\n"}
        )

def test_repair_parser_fails_unknown_file(monkeypatch):
    engine = object.__new__(AIEngine)
    with pytest.raises(RuntimeError, match="unknown file"):
        engine._parse_repair_response(
            '''{"edits":[{"file":"nonexistent.py","old_text":"a","new_text":"b"}]}''',
            {"calculator.py": "a\n"}
        )

def test_repair_parser_fails_old_text_not_found(monkeypatch):
    engine = object.__new__(AIEngine)
    with pytest.raises(RuntimeError, match="not found in calculator.py"):
        engine._parse_repair_response(
            '''{"edits":[{"file":"calculator.py","old_text":"xyz","new_text":"b"}]}''',
            {"calculator.py": "a\n"}
        )

def test_repair_parser_fails_multiple_occurrences_without_index(monkeypatch):
    engine = object.__new__(AIEngine)
    with pytest.raises(RuntimeError, match="Specify 'occurrence'"):
        engine._parse_repair_response(
            '''{"edits":[{"file":"calculator.py","old_text":"a","new_text":"b"}]}''',
            {"calculator.py": "a\na\n"}
        )

def test_repair_parser_handles_multi_file_repair(monkeypatch):
    engine = object.__new__(AIEngine)
    engine.client = FakeClient([
        response(
            '''{"edits":[
                {"file":"fileA.py","old_text":"contractA","new_text":"contractB"},
                {"file":"fileB.py","old_text":"useA","new_text":"useB"}
            ],"explanation":"fix","confidence":0.95}'''
        )
    ])
    monkeypatch.setattr("app.ai.settings", type("TestSettings", (), {"ai_models": ["primary"], "max_patch_chars": 30000})())
    contents = {
        "fileA.py": "def contractA(): pass\n",
        "fileB.py": "def run(): useA()\n"
    }
    result = engine.generate_patch("", {}, contents)
    
    assert set(result["touched_files"]) == {"fileA.py", "fileB.py"}
    assert "--- a/fileA.py" in result["patch"]
    assert "--- a/fileB.py" in result["patch"]
    assert "+def contractB(): pass" in result["patch"]
    assert "+def run(): useB()" in result["patch"]
