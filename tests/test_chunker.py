"""Tests for the chunker module (cAST algorithm)."""

import tempfile
from pathlib import Path

import pytest

from embecode.chunker import (
    Chunk,
    chunk_file,
    chunk_files,
    get_language_for_file,
)
from embecode.config import LanguageConfig


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
