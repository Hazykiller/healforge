from __future__ import annotations

import pytest
from app.ai import AIEngine


def test_resolve_file_path_basename():
    engine = object.__new__(AIEngine)
    working = {"src/service.py": "content", "lib/utils.py": "utils"}
    assert engine._resolve_file_path("service.py", working) == "src/service.py"
    assert engine._resolve_file_path("./src/service.py", working) == "src/service.py"
    assert engine._resolve_file_path("utils.py", working) == "lib/utils.py"
    assert engine._resolve_file_path("nonexistent.py", working) is None


def test_resilient_replace_exact_match():
    engine = object.__new__(AIEngine)
    orig = "def add(a, b):\n    return a - b\n"
    old = "    return a - b"
    new = "    return a + b"
    res = engine._resilient_replace(orig, old, new, file_path="math.py")
    assert "return a + b" in res


def test_resilient_replace_indentation_adaptation():
    engine = object.__new__(AIEngine)
    # File has 8-space indentation
    orig = "class Calc:\n    def run(self):\n        return a - b\n"
    # LLM produced 0-space or 4-space indentation
    old = "return a - b"
    new = "return a + b"
    res = engine._resilient_replace(orig, old, new, file_path="calc.py")
    assert "        return a + b" in res


def test_resilient_replace_blank_lines_tolerance():
    engine = object.__new__(AIEngine)
    orig = "def test():\n    x = 1\n\n    y = 2\n    return x + y\n"
    # LLM omitted internal blank line in search block
    old = "    x = 1\n    y = 2"
    new = "    x = 10\n    y = 20"
    res = engine._resilient_replace(orig, old, new, file_path="test.py")
    assert "x = 10" in res
    assert "y = 20" in res


def test_resilient_replace_fuzzy_matcher_minor_variation():
    engine = object.__new__(AIEngine)
    orig = (
        "def compute(op, a, b):\n"
        "    # Check the operation string\n"
        "    if op == 'add':\n"
        "        return a + b\n"
    )
    # LLM had double quotes instead of single quotes and slightly different comment
    old = (
        "    # check operation\n"
        "    if op == \"add\":\n"
        "        return a + b"
    )
    new = (
        "    # Check operation\n"
        "    if op in ('add', 'plus'):\n"
        "        return a + b"
    )
    res = engine._resilient_replace(orig, old, new, file_path="compute.py")
    assert "if op in ('add', 'plus'):" in res


def test_unified_diff_to_edits():
    engine = object.__new__(AIEngine)
    diff = """--- a/src/service.py
+++ b/src/service.py
@@ -5,4 +5,4 @@
 def handle():
-    if op == "bad":
+    if op == "good":
         return True
"""
    edits = engine._unified_diff_to_edits(diff, {"src/service.py": "..."})
    assert len(edits) == 1
    assert edits[0]["file"] == "src/service.py"
    assert 'if op == "bad":' in edits[0]["old_text"]
    assert 'if op == "good":' in edits[0]["new_text"]


def test_extract_aider_blocks():
    engine = object.__new__(AIEngine)
    raw = """
I will fix the bug in src/service.py:

src/service.py
<<<<<<< SEARCH
    if op == "sub":
        return a - b
=======
    if op == "sub":
        return a - b if a > b else b - a
>>>>>>> REPLACE

Hope this helps!
"""
    edits = engine._extract_aider_blocks(raw, {"src/service.py": "..."})
    assert len(edits) == 1
    assert edits[0]["file"] == "src/service.py"
    assert 'return a - b' in edits[0]["old_text"]
    assert 'return a - b if a > b else b - a' in edits[0]["new_text"]


def test_parse_repair_response_with_unified_diff():
    engine = object.__new__(AIEngine)
    raw = """Here is the patch:
```diff
--- a/src/calculator.py
+++ b/src/calculator.py
@@ -1,3 +1,3 @@
-def foo(): return 1
+def foo(): return 2
```
"""
    contents = {"src/calculator.py": "def foo(): return 1\n"}
    res = engine._parse_repair_response(raw, contents)
    assert "src/calculator.py" in res["touched_files"]
    assert "+def foo(): return 2" in res["patch"]


def test_parse_repair_response_with_aider_blocks():
    engine = object.__new__(AIEngine)
    raw = """
src/calc.py
<<<<<<< SEARCH
def calc(x):
    return x * 1
=======
def calc(x):
    return x * 2
>>>>>>> REPLACE
"""
    contents = {"src/calc.py": "def calc(x):\n    return x * 1\n"}
    res = engine._parse_repair_response(raw, contents)
    assert "src/calc.py" in res["touched_files"]
    assert "+    return x * 2" in res["patch"]
