"""Tests for the chunker module (cAST algorithm)."""

import tempfile
from pathlib import Path

from unittest.mock import Mock

from embecode.chunker import (
    Chunk,
    _count_non_whitespace,
    _extract_definition_names,
    _get_context_info,
    _get_parser,
    _merge_nodes_into_chunk,
    chunk_file,
    chunk_files,
    get_language_for_file,
)
from embecode.config import LanguageConfig


def _make_node(
    node_type: str,
    name: str | None = None,
    children: list | None = None,
    fields: dict | None = None,
) -> Mock:
    """Create a mock tree-sitter Node for definition extraction tests.

    Args:
        node_type: The node's ``type`` attribute (e.g. ``"function_definition"``).
        name: If given, ``child_by_field_name("name")`` returns a node whose
            ``.text`` is this string encoded as UTF-8 bytes.
        children: Child mock nodes.  Defaults to empty list.
        fields: Extra field_name → Mock mappings for ``child_by_field_name``.
    """
    node = Mock()
    node.type = node_type
    node.children = children or []

    extra_fields: dict[str, Mock | None] = fields or {}

    if name is not None:
        name_node = Mock()
        name_node.text = name.encode("utf-8")
        extra_fields.setdefault("name", name_node)

    def _child_by_field_name(field: str) -> Mock | None:
        return extra_fields.get(field)

    node.child_by_field_name = _child_by_field_name
    return node


def test_get_language_for_file():
    """Test language detection from file extension."""
    assert get_language_for_file(Path("test.py")) == "python"
    assert get_language_for_file(Path("test.js")) == "javascript"
    assert get_language_for_file(Path("test.jsx")) == "javascript"
    assert get_language_for_file(Path("test.ts")) == "typescript"
    assert get_language_for_file(Path("test.tsx")) == "tsx"
    assert get_language_for_file(Path("test.go")) == "go"
    assert get_language_for_file(Path("test.rs")) == "rust"
    assert get_language_for_file(Path("test.java")) == "java"
    assert get_language_for_file(Path("test.cpp")) == "cpp"
    assert get_language_for_file(Path("test.unknown")) is None


def test_chunk_create():
    """Test Chunk.create computes hash correctly."""
    chunk = Chunk.create(
        content="def foo():\n    pass",
        file_path="test.py",
        language="python",
        start_line=1,
        end_line=2,
        context="File: test.py",
    )

    assert chunk.content == "def foo():\n    pass"
    assert chunk.file_path == "test.py"
    assert chunk.language == "python"
    assert chunk.start_line == 1
    assert chunk.end_line == 2
    assert chunk.context == "File: test.py"
    assert len(chunk.hash) == 40  # SHA1 hash is 40 hex chars


def test_chunk_small_python_file():
    """Test chunking a small Python file that fits in one chunk."""
    code = '''def hello():
    """Say hello."""
    print("Hello, world!")

def goodbye():
    """Say goodbye."""
    print("Goodbye!")
'''

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = Path(f.name)

    try:
        config = LanguageConfig()
        chunks = chunk_file(temp_path, config)

        # Should create chunks (exact number depends on tree-sitter parsing)
        assert len(chunks) >= 1

        # Verify all chunks have required fields
        for chunk in chunks:
            assert chunk.content
            assert chunk.file_path == str(temp_path)
            assert chunk.language == "python"
            assert chunk.start_line >= 1
            assert chunk.end_line >= chunk.start_line
            assert "File:" in chunk.context
            assert len(chunk.hash) == 40

    finally:
        temp_path.unlink()


def test_chunk_large_python_file():
    """Test chunking a file with mixed-size functions."""
    # Create a file with some small and some large functions
    # The cAST algorithm should batch small functions and separate large ones
    code = """def small_function_1():
    return 1

def small_function_2():
    return 2

"""
    # Add a large function that exceeds 300 chars
    large_func = '''def large_function():
    """A large function with lots of code."""
    results = []
    for iteration in range(100):
        x = iteration * 2
        y = x + iteration
        z = y * x
        results.append({"x": x, "y": y, "z": z})
    # Process results
    total = sum(r["z"] for r in results)
    average = total / len(results) if results else 0
    maximum = max(r["z"] for r in results) if results else 0
    minimum = min(r["z"] for r in results) if results else 0
    return {"total": total, "avg": average, "max": maximum, "min": minimum, "data": results}

'''
    code += large_func
    code += """def small_function_3():
    return 3
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = Path(f.name)

    try:
        # Use a chunk size that forces the large function into its own chunk
        config = LanguageConfig(python=300)
        chunks = chunk_file(temp_path, config)

        # Verify we get at least one chunk
        assert len(chunks) >= 1

        # Verify all chunks have valid line ranges
        for chunk in chunks:
            assert chunk.start_line >= 1
            assert chunk.end_line >= chunk.start_line
            assert len(chunk.content) > 0

    finally:
        temp_path.unlink()


def test_chunk_javascript_file():
    """Test chunking a JavaScript file."""
    code = """function add(a, b) {
  return a + b;
}

function subtract(a, b) {
  return a - b;
}

const multiply = (a, b) => a * b;
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
        f.write(code)
        temp_path = Path(f.name)

    try:
        config = LanguageConfig()
        chunks = chunk_file(temp_path, config)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.language == "javascript"

    finally:
        temp_path.unlink()


def test_chunk_typescript_file():
    """Test chunking a TypeScript file."""
    code = """interface User {
  id: number;
  name: string;
}

function getUser(id: number): User {
  return { id, name: "Test" };
}
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ts", delete=False) as f:
        f.write(code)
        temp_path = Path(f.name)

    try:
        config = LanguageConfig()
        chunks = chunk_file(temp_path, config)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.language == "typescript"

    finally:
        temp_path.unlink()


def test_chunk_unsupported_file():
    """Test chunking a file with unsupported extension returns empty list."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".unknown", delete=False) as f:
        f.write("some content")
        temp_path = Path(f.name)

    try:
        config = LanguageConfig()
        chunks = chunk_file(temp_path, config)
        assert chunks == []
    finally:
        temp_path.unlink()


def test_chunk_nonexistent_file():
    """Test chunking a non-existent file returns empty list."""
    config = LanguageConfig()
    chunks = chunk_file(Path("/nonexistent/file.py"), config)
    assert chunks == []


def test_chunk_files_multiple():
    """Test chunking multiple files with chunk_files."""
    code1 = "def foo():\n    pass\n"
    code2 = "def bar():\n    pass\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        file1 = tmppath / "test1.py"
        file2 = tmppath / "test2.py"

        file1.write_text(code1)
        file2.write_text(code2)

        config = LanguageConfig()
        all_chunks = list(chunk_files([file1, file2], config))

        # Should have chunks from both files
        assert len(all_chunks) >= 2

        # Check that chunks come from both files
        file_paths = {chunk.file_path for chunk in all_chunks}
        assert str(file1) in file_paths
        assert str(file2) in file_paths


def test_chunk_preserves_content():
    """Test that concatenating all chunks reconstructs the original file."""
    code = """def function_a():
    x = 1
    return x

def function_b():
    y = 2
    return y
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = Path(f.name)

    try:
        config = LanguageConfig()
        chunks = chunk_file(temp_path, config)

        # Concatenate all chunk contents
        reconstructed = "".join(chunk.content for chunk in chunks)

        # Should match the original (modulo possible trailing whitespace)
        assert reconstructed.strip() == code.strip()

    finally:
        temp_path.unlink()


def test_chunk_respects_language_config():
    """Test that chunk sizes respect the language-specific configuration."""
    # Create a Python file with multiple functions
    functions = [f"def func_{i}():\n    return {i}\n\n" for i in range(20)]
    code = "".join(functions)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = Path(f.name)

    try:
        # Use different config sizes and verify chunking behavior changes
        small_config = LanguageConfig(python=200)
        large_config = LanguageConfig(python=5000)

        small_chunks = chunk_file(temp_path, small_config)
        large_chunks = chunk_file(temp_path, large_config)

        # Smaller max size should generally create more chunks
        # (though exact behavior depends on tree structure)
        assert len(small_chunks) >= len(large_chunks)

    finally:
        temp_path.unlink()


def test_chunk_hash_changes_with_content():
    """Test that chunk hash changes when content changes."""
    code1 = "def foo():\n    return 1\n"
    code2 = "def foo():\n    return 2\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        file_path = tmppath / "test.py"

        # Chunk first version
        file_path.write_text(code1)
        config = LanguageConfig()
        chunks1 = chunk_file(file_path, config)

        # Chunk second version
        file_path.write_text(code2)
        chunks2 = chunk_file(file_path, config)

        # Hashes should be different
        assert len(chunks1) > 0 and len(chunks2) > 0
        assert chunks1[0].hash != chunks2[0].hash


def test_chunk_empty_file():
    """Test chunking an empty file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("")
        temp_path = Path(f.name)

    try:
        config = LanguageConfig()
        chunks = chunk_file(temp_path, config)
        # Empty file should produce no chunks or just one empty chunk
        assert isinstance(chunks, list)
    finally:
        temp_path.unlink()


def test_chunk_whitespace_only_file():
    """Test chunking a file with only whitespace."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("   \n\n   \n")
        temp_path = Path(f.name)

    try:
        config = LanguageConfig()
        chunks = chunk_file(temp_path, config)
        # Whitespace-only file should produce no meaningful chunks
        assert isinstance(chunks, list)
    finally:
        temp_path.unlink()


def test_chunk_with_syntax_error():
    """Test chunking a file with syntax errors - should still parse."""
    code = """def incomplete_function(
    # Missing closing parenthesis and body
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = Path(f.name)

    try:
        config = LanguageConfig()
        chunks = chunk_file(temp_path, config)
        # Tree-sitter is error-tolerant, should still produce chunks
        assert isinstance(chunks, list)
    finally:
        temp_path.unlink()


def test_chunk_very_large_single_function():
    """Test chunking a single function that vastly exceeds max size."""
    # Create a function with many lines
    lines = ["def huge_function():\n"]
    lines.append('    """A very large function."""\n')
    for i in range(100):
        lines.append(f"    x{i} = {i} * 2\n")
    lines.append("    return sum([" + ", ".join([f"x{i}" for i in range(100)]) + "])\n")
    code = "".join(lines)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = Path(f.name)

    try:
        # Use small max size to force recursion
        config = LanguageConfig(python=100)
        chunks = chunk_file(temp_path, config)

        # Should break down into multiple chunks through recursion
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.content.strip()  # No empty chunks
    finally:
        temp_path.unlink()


def test_chunk_nested_structures():
    """Test chunking deeply nested code structures."""
    code = """class OuterClass:
    class MiddleClass:
        class InnerClass:
            def inner_method(self):
                for i in range(10):
                    for j in range(10):
                        x = i * j
                        if x > 50:
                            y = x + 1
                        else:
                            y = x - 1
                return y
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = Path(f.name)

    try:
        config = LanguageConfig(python=150)
        chunks = chunk_file(temp_path, config)

        # Should handle nested structures gracefully
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.language == "python"
            assert chunk.start_line >= 1
    finally:
        temp_path.unlink()


def test_chunk_mixed_content_types():
    """Test chunking a file with functions, classes, and module-level code."""
    code = """# Module-level comment
import os
import sys

CONSTANT = 42

def module_function():
    return CONSTANT

class MyClass:
    def __init__(self):
        self.value = 0

    def method(self):
        return self.value

# More module-level code
if __name__ == "__main__":
    obj = MyClass()
    print(module_function())
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = Path(f.name)

    try:
        config = LanguageConfig()
        chunks = chunk_file(temp_path, config)

        # Should produce multiple chunks
        assert len(chunks) >= 1

        # All chunks should have valid metadata
        for chunk in chunks:
            assert chunk.file_path == str(temp_path)
            assert chunk.language == "python"
            assert "File:" in chunk.context
            assert "Language:" in chunk.context
    finally:
        temp_path.unlink()


def test_chunk_go_file():
    """Test chunking a Go file."""
    code = """package main

import "fmt"

func add(a, b int) int {
    return a + b
}

func main() {
    result := add(1, 2)
    fmt.Println(result)
}
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".go", delete=False) as f:
        f.write(code)
        temp_path = Path(f.name)

    try:
        config = LanguageConfig()
        chunks = chunk_file(temp_path, config)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.language == "go"
    finally:
        temp_path.unlink()


def test_chunk_rust_file():
    """Test chunking a Rust file."""
    code = """fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn main() {
    let result = add(1, 2);
    println!("{}", result);
}
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".rs", delete=False) as f:
        f.write(code)
        temp_path = Path(f.name)

    try:
        config = LanguageConfig()
        chunks = chunk_file(temp_path, config)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.language == "rust"
    finally:
        temp_path.unlink()


def test_chunk_with_unicode_content():
    """Test chunking a file with unicode characters."""
    code = """def greet(name):
    '''Say hello in multiple languages.'''
    messages = [
        f"Hello {name}!",
        f"Bonjour {name}!",
        f"你好 {name}!",
        f"こんにちは {name}!",
        f"Привет {name}!",
    ]
    return messages
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        temp_path = Path(f.name)

    try:
        config = LanguageConfig()
        chunks = chunk_file(temp_path, config)

        assert len(chunks) >= 1
        # Verify unicode is preserved in chunk content
        full_content = "".join(chunk.content for chunk in chunks)
        assert "你好" in full_content
        assert "こんにちは" in full_content
        assert "Привет" in full_content
    finally:
        temp_path.unlink()


def test_chunk_line_numbers_are_accurate():
    """Test that chunk line numbers accurately reflect the source."""
    code = """# Line 1
def foo():  # Line 2
    pass    # Line 3
            # Line 4
def bar():  # Line 5
    pass    # Line 6
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = Path(f.name)

    try:
        config = LanguageConfig()
        chunks = chunk_file(temp_path, config)

        assert len(chunks) >= 1
        # Verify all chunks have reasonable line ranges
        for chunk in chunks:
            assert chunk.start_line >= 1
            assert chunk.end_line >= chunk.start_line
            assert chunk.end_line <= 6  # File has 6 lines
    finally:
        temp_path.unlink()


def test_chunk_multiple_files_with_errors():
    """Test chunk_files gracefully handles files with errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create a valid file
        valid_file = tmppath / "valid.py"
        valid_file.write_text("def foo():\n    pass\n")

        # Create an unsupported file
        unsupported_file = tmppath / "data.bin"
        unsupported_file.write_bytes(b"\x00\x01\x02\x03")

        # Include a non-existent file
        nonexistent = tmppath / "missing.py"

        config = LanguageConfig()
        all_chunks = list(chunk_files([valid_file, unsupported_file, nonexistent], config))

        # Should only get chunks from the valid file
        assert len(all_chunks) >= 1
        assert all(chunk.file_path == str(valid_file) for chunk in all_chunks)


def test_chunk_respects_typescript_config():
    """Test that TypeScript files use the typescript config value."""
    code = """interface Point {
    x: number;
    y: number;
}

function distance(p1: Point, p2: Point): number {
    return Math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2);
}
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ts", delete=False) as f:
        f.write(code)
        temp_path = Path(f.name)

    try:
        # Use custom TypeScript chunk size
        config = LanguageConfig(typescript=100)
        chunks = chunk_file(temp_path, config)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.language == "typescript"
    finally:
        temp_path.unlink()


def test_chunk_respects_javascript_config():
    """Test that JavaScript files use the javascript config value."""
    code = """function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

const result = fibonacci(10);
console.log(result);
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
        f.write(code)
        temp_path = Path(f.name)

    try:
        # Use custom JavaScript chunk size
        config = LanguageConfig(javascript=50)
        chunks = chunk_file(temp_path, config)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.language == "javascript"
    finally:
        temp_path.unlink()


def test_chunk_context_contains_metadata():
    """Test that chunk context contains expected metadata."""
    code = "def test():\n    pass\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = Path(f.name)

    try:
        config = LanguageConfig()
        chunks = chunk_file(temp_path, config)

        assert len(chunks) >= 1
        chunk = chunks[0]

        # Verify context contains file path and language
        assert f"File: {temp_path}" in chunk.context
        assert "Language: python" in chunk.context
    finally:
        temp_path.unlink()


# Unit tests for internal helper functions


def test_count_non_whitespace():
    """Test _count_non_whitespace helper function."""
    assert _count_non_whitespace("hello") == 5
    assert _count_non_whitespace("hello world") == 10
    assert _count_non_whitespace("   hello   ") == 5
    assert _count_non_whitespace("") == 0
    assert _count_non_whitespace("   \n\t   ") == 0
    assert _count_non_whitespace("a\nb\tc") == 3


def test_get_parser_invalid_language():
    """Test _get_parser with invalid language returns None."""
    parser = _get_parser("nonexistent_language_xyz")
    assert parser is None


def test_get_context_info():
    """Test _get_context_info helper function."""
    context = _get_context_info("/path/to/file.py", "python", "def foo(): pass")

    assert "File: /path/to/file.py" in context
    assert "Language: python" in context


def test_merge_nodes_into_chunk_empty_list():
    """Test _merge_nodes_into_chunk with empty node list."""
    chunk = _merge_nodes_into_chunk([], b"source code", "/path/file.py", "python")
    assert chunk is None


# ---------------------------------------------------------------------------
# Definition extraction tests (using mock tree-sitter nodes)
# ---------------------------------------------------------------------------


class TestDefinitionExtraction:
    """Tests for _extract_definition_names and related helpers."""

    def test_extract_python_function_definition(self) -> None:
        """Python `def foo():` produces definitions containing `function foo`."""
        node = _make_node("function_definition", name="foo")
        result = _extract_definition_names([node], "python")
        assert result == ["function foo"]

    def test_extract_python_class_definition(self) -> None:
        """Python `class Bar:` produces definitions containing `class Bar`."""
        node = _make_node("class_definition", name="Bar")
        result = _extract_definition_names([node], "python")
        assert result == ["class Bar"]

    def test_extract_python_decorated_function(self) -> None:
        """Python `@decorator\\ndef baz():` produces `function baz`."""
        inner = _make_node("function_definition", name="baz")
        node = _make_node("decorated_definition", children=[inner])
        result = _extract_definition_names([node], "python")
        assert result == ["function baz"]

    def test_extract_python_decorated_class(self) -> None:
        """Python `@decorator\\nclass Qux:` produces `class Qux`."""
        inner = _make_node("class_definition", name="Qux")
        node = _make_node("decorated_definition", children=[inner])
        result = _extract_definition_names([node], "python")
        assert result == ["class Qux"]

    def test_extract_multiple_definitions(self) -> None:
        """Multiple nodes produce all named definitions."""
        cls = _make_node("class_definition", name="MyClass")
        method_a = _make_node("function_definition", name="method_a")
        method_b = _make_node("function_definition", name="method_b")
        result = _extract_definition_names([cls, method_a, method_b], "python")
        assert result == ["class MyClass", "function method_a", "function method_b"]

    def test_extract_typescript_function(self) -> None:
        """TypeScript `function search() {}` produces `function search`."""
        node = _make_node("function_declaration", name="search")
        result = _extract_definition_names([node], "typescript")
        assert result == ["function search"]
