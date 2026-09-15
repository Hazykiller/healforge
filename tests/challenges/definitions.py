from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Challenge:
    id: str
    language: str
    description: str
    files: dict[str, str]
    bug_description: str
    expected_language: str
    expected_test_command: str
    sample_edit: dict[str, Any]
    expected_behavior: str


CHALLENGES: list[Challenge] = [
    Challenge(
        id="python_single_file",
        language="python",
        description="Single-file Python math function with sign bug",
        files={
            "calc.py": "def add(a: int, b: int) -> int:\n    return a - b\n",
            "test_calc.py": (
                "from calc import add\n\n"
                "def test_add():\n"
                "    assert add(2, 3) == 5\n"
            ),
        },
        bug_description="add() subtracts instead of adding",
        expected_language="python",
        expected_test_command="python -m pytest -q -p no:cacheprovider",
        sample_edit={
            "file": "calc.py",
            "old_text": "    return a - b",
            "new_text": "    return a + b",
        },
        expected_behavior="add(2, 3) returns 5",
    ),
    Challenge(
        id="python_multi_file",
        language="python",
        description="Multi-file Python package with arithmetic error",
        files={
            "src/__init__.py": "",
            "src/ops.py": "def multiply(a: int, b: int) -> int:\n    return a + b\n",
            "tests/__init__.py": "",
            "tests/test_ops.py": (
                "from src.ops import multiply\n\n"
                "def test_multiply():\n"
                "    assert multiply(3, 4) == 12\n"
            ),
            "requirements.txt": "pytest>=7.0.0\n",
        },
        bug_description="multiply() adds instead of multiplying",
        expected_language="python",
        expected_test_command="python -m pytest -q -p no:cacheprovider",
        sample_edit={
            "file": "src/ops.py",
            "old_text": "    return a + b",
            "new_text": "    return a * b",
        },
        expected_behavior="multiply(3, 4) returns 12",
    ),
    Challenge(
        id="javascript_node",
        language="node",
        description="CommonJS Node project with greeting bug",
        files={
            "index.js": (
                "function greet(name) {\n"
                "  return 'Goodbye, ' + name;\n"
                "}\n"
                "module.exports = { greet };\n"
            ),
            "index.test.js": (
                "const { greet } = require('./index');\n"
                "test('greet returns Hello', () => {\n"
                "  expect(greet('World')).toBe('Hello, World');\n"
                "});\n"
            ),
            "package.json": (
                '{\n'
                '  "name": "js-challenge",\n'
                '  "scripts": { "test": "jest" },\n'
                '  "devDependencies": { "jest": "^29.0.0" }\n'
                '}\n'
            ),
        },
        bug_description="greet() returns Goodbye instead of Hello",
        expected_language="node",
        expected_test_command="npm test",
        sample_edit={
            "file": "index.js",
            "old_text": "  return 'Goodbye, ' + name;",
            "new_text": "  return 'Hello, ' + name;",
        },
        expected_behavior="greet('World') returns 'Hello, World'",
    ),
    Challenge(
        id="typescript_node",
        language="node",
        description="TypeScript project with incorrect subtraction operator",
        files={
            "src/calc.ts": (
                "export function sub(a: number, b: number): number {\n"
                "  return a + b;\n"
                "}\n"
            ),
            "package.json": (
                '{\n'
                '  "name": "ts-challenge",\n'
                '  "scripts": { "test": "vitest run" },\n'
                '  "devDependencies": {\n'
                '    "typescript": "^5.0.0",\n'
                '    "vitest": "^1.0.0"\n'
                '  }\n'
                '}\n'
            ),
        },
        bug_description="sub() adds instead of subtracting",
        expected_language="node",
        expected_test_command="npm test",
        sample_edit={
            "file": "src/calc.ts",
            "old_text": "  return a + b;",
            "new_text": "  return a - b;",
        },
        expected_behavior="sub(5, 3) returns 2",
    ),
    Challenge(
        id="java_maven",
        language="java",
        description="Java Maven project with square function bug",
        files={
            "pom.xml": (
                '<project xmlns="http://maven.apache.org/POM/4.0.0">\n'
                '  <modelVersion>4.0.0</modelVersion>\n'
                '  <groupId>com.healforge</groupId>\n'
                '  <artifactId>challenge</artifactId>\n'
                '  <version>1.0.0</version>\n'
                '</project>\n'
            ),
            "src/main/java/Calculator.java": (
                "package com.healforge;\n\n"
                "public class Calculator {\n"
                "    public static int square(int x) {\n"
                "        return x + x;\n"
                "    }\n"
                "}\n"
            ),
        },
        bug_description="square() doubles instead of squaring",
        expected_language="java",
        expected_test_command="mvn test -q",
        sample_edit={
            "file": "src/main/java/Calculator.java",
            "old_text": "        return x + x;",
            "new_text": "        return x * x;",
        },
        expected_behavior="square(4) returns 16",
    ),
    Challenge(
        id="java_gradle",
        language="java",
        description="Java Gradle project with incorrect comparison",
        files={
            "build.gradle": (
                "plugins {\n"
                "    id 'java'\n"
                "}\n"
            ),
            "src/main/java/App.java": (
                "public class App {\n"
                "    public static boolean isPositive(int n) {\n"
                "        return n < 0;\n"
                "    }\n"
                "}\n"
            ),
        },
        bug_description="isPositive() checks for < 0 instead of > 0",
        expected_language="java",
        expected_test_command="gradle test",
        sample_edit={
            "file": "src/main/java/App.java",
            "old_text": "        return n < 0;",
            "new_text": "        return n > 0;",
        },
        expected_behavior="isPositive(5) returns true",
    ),
    Challenge(
        id="go_modules",
        language="go",
        description="Go module with absolute value bug",
        files={
            "go.mod": "module example.com/calc\n\ngo 1.22\n",
            "calc.go": (
                "package calc\n\n"
                "func Abs(x int) int {\n"
                "\tif x < 0 {\n"
                "\t\treturn x\n"
                "\t}\n"
                "\treturn x\n"
                "}\n"
            ),
        },
        bug_description="Abs() returns negative number when x < 0",
        expected_language="go",
        expected_test_command="go test ./...",
        sample_edit={
            "file": "calc.go",
            "old_text": "\tif x < 0 {\n\t\treturn x\n\t}",
            "new_text": "\tif x < 0 {\n\t\treturn -x\n\t}",
        },
        expected_behavior="Abs(-5) returns 5",
    ),
    Challenge(
        id="rust_cargo",
        language="rust",
        description="Rust Cargo crate with double function error",
        files={
            "Cargo.toml": (
                "[package]\n"
                "name = \"calc\"\n"
                "version = \"0.1.0\"\n"
                "edition = \"2021\"\n"
            ),
            "src/lib.rs": (
                "pub fn double(x: i32) -> i32 {\n"
                "    x + 1\n"
                "}\n"
            ),
        },
        bug_description="double() adds 1 instead of multiplying by 2",
        expected_language="rust",
        expected_test_command="cargo test",
        sample_edit={
            "file": "src/lib.rs",
            "old_text": "    x + 1",
            "new_text": "    x * 2",
        },
        expected_behavior="double(4) returns 8",
    ),
    Challenge(
        id="cpp_cmake",
        language="cpp",
        description="C++ CMake project with subtract vs add inversion",
        files={
            "CMakeLists.txt": (
                "cmake_minimum_required(VERSION 3.10)\n"
                "project(Calc)\n"
                "add_library(calc calc.cpp)\n"
            ),
            "calc.cpp": (
                "int add(int a, int b) {\n"
                "    return a - b;\n"
                "}\n"
            ),
        },
        bug_description="add() subtracts instead of adding",
        expected_language="cpp",
        expected_test_command="ctest --test-dir build --output-on-failure",
        sample_edit={
            "file": "calc.cpp",
            "old_text": "    return a - b;",
            "new_text": "    return a + b;",
        },
        expected_behavior="add(2, 3) returns 5",
    ),
]


def get_challenge(challenge_id: str) -> Challenge | None:
    for ch in CHALLENGES:
        if ch.id == challenge_id:
            return ch
    return None
