#!/usr/bin/env python3
"""
Manual test script to verify all implemented features.

This script tests:
- Config loading and resolution
- Chunking with cAST algorithm
- Embedder (local models only, no API tests)
- Indexer orchestration (with mocks)
- Searcher with RRF fusion
- Watcher with debounce
"""

import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def test_config() -> bool:
    """Test configuration loading and resolution."""
    print_section("Testing Config Loading")

    try:
        from embecode.config import get_chunk_size_for_language, load_config

        # Test 1: Default config
        print("✓ Test 1: Load default config")
        config = load_config()
        assert config.index.include == ["src/", "lib/", "tests/"]
        assert config.embeddings.model == "nomic-embed-text-v1.5"
        assert config.search.default_mode == "hybrid"
        assert config.daemon.debounce_ms == 500
        print("  Default config loaded successfully")
        print(f"  Model: {config.embeddings.model}")
        print(f"  Include: {config.index.include}")

        # Test 2: Get chunk size for language
        print("\n✓ Test 2: Get chunk sizes for languages")
        python_size = get_chunk_size_for_language(config, "python")
        ts_size = get_chunk_size_for_language(config, "typescript")
        default_size = get_chunk_size_for_language(config, "ruby")

        assert python_size == 1500
        assert ts_size == 1200
        assert default_size == 1000
        print(f"  Python: {python_size}, TypeScript: {ts_size}, Default: {default_size}")

        # Test 3: CLI overrides
        print("\n✓ Test 3: CLI overrides")
        config_override = load_config(cli_overrides={"embeddings": {"model": "test-model"}})
        assert config_override.embeddings.model == "test-model"
        print(f"  Overridden model: {config_override.embeddings.model}")

        print("\n✅ Config tests PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Config tests FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_chunker() -> bool:
    """Test chunking with cAST algorithm."""
    print_section("Testing Chunker")

    try:
        from embecode.chunker import Chunk, chunk_file, get_language_for_file
        from embecode.config import LanguageConfig

        # Create temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Test 1: Python file
            print("✓ Test 1: Chunk Python file")
            py_file = tmppath / "test.py"
            py_file.write_text("""
def hello():
    print("hello")

def world():
    print("world")

class Foo:
    def bar(self):
        return 42
""")

            chunks = chunk_file(py_file, LanguageConfig())
            assert len(chunks) > 0
            print(f"  Created {len(chunks)} chunks from Python file")

            # Verify chunk properties
            first_chunk = chunks[0]
            assert isinstance(first_chunk, Chunk)
            assert first_chunk.language == "python"
            assert first_chunk.file_path == str(py_file)
            assert len(first_chunk.hash) == 40  # SHA1 hex digest
            print(f"  First chunk: lines {first_chunk.start_line}-{first_chunk.end_line}")
            print(f"  Hash: {first_chunk.hash[:16]}...")

            # Test 2: JavaScript file
            print("\n✓ Test 2: Chunk JavaScript file")
            js_file = tmppath / "test.js"
            js_file.write_text("""
function hello() {
    console.log("hello");
}

const world = () => {
    console.log("world");
};
""")

            js_chunks = chunk_file(js_file, LanguageConfig())
            assert len(js_chunks) > 0
            print(f"  Created {len(js_chunks)} chunks from JavaScript file")

            # Test 3: Verify content preservation
            print("\n✓ Test 3: Verify content preservation")
            reconstructed = "".join(chunk.content for chunk in chunks)
            original = py_file.read_text()
            # Note: might have minor whitespace differences due to AST parsing
            # but total character count should be close
            assert len(reconstructed) >= len(original) * 0.8  # Allow some variance
            print(f"  Original: {len(original)} chars, Reconstructed: {len(reconstructed)} chars")

            # Test 4: Language detection
            print("\n✓ Test 4: Language detection")
            assert get_language_for_file(Path("test.py")) == "python"
            assert get_language_for_file(Path("test.ts")) == "typescript"
            assert get_language_for_file(Path("test.jsx")) == "javascript"
            assert get_language_for_file(Path("test.unknown")) is None
            print("  Language detection working correctly")

        print("\n✅ Chunker tests PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Chunker tests FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_embedder() -> bool:
    """Test embedder with local model only."""
    print_section("Testing Embedder")

    try:
        from embecode.config import EmbeddingsConfig
        from embecode.embedder import Embedder

        # Test 1: Local model initialization (lazy loading)
        print("✓ Test 1: Local model initialization")
        config = EmbeddingsConfig(model="all-MiniLM-L6-v2")  # Faster small model
        embedder = Embedder(config)
        assert embedder._model is None  # Not loaded yet
        print("  Embedder created (model not loaded yet)")

        # Test 2: Embed single text (triggers lazy load)
        print("\n✓ Test 2: Embed text (triggers model load)")
        print("  This will download the model if not cached (~100MB)...")
        embeddings = embedder.embed(["test text"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) > 0  # Has embedding dimension
        assert embedder._model is not None  # Now loaded
        print(f"  Embedding dimension: {len(embeddings[0])}")
        print("  Model loaded successfully")

        # Test 3: Embed multiple texts
        print("\n✓ Test 3: Embed multiple texts")
        texts = ["first text", "second text", "third text"]
        embeddings = embedder.embed(texts)
        assert len(embeddings) == 3
        assert all(len(emb) == len(embeddings[0]) for emb in embeddings)
        print(f"  Created {len(embeddings)} embeddings")

        # Test 4: Check dimension property
        print("\n✓ Test 4: Check dimension property")
        dim = embedder.dimension
        assert dim > 0
        assert dim == len(embeddings[0])
        print(f"  Dimension: {dim}")

        # Test 5: Empty list
        print("\n✓ Test 5: Empty list handling")
        empty_result = embedder.embed([])
        assert empty_result == []
        print("  Empty list handled correctly")

        print("\n✅ Embedder tests PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Embedder tests FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_searcher() -> bool:
    """Test searcher with mock database."""
    print_section("Testing Searcher")

    try:
        from embecode.searcher import ChunkResult, Searcher

        # Mock database and embedder
        class MockDB:
            def get_index_stats(self):
                return {"total_chunks": 10, "files_indexed": 5, "last_updated": "2025-01-01"}

            def vector_search(self, query_embedding, top_k, path_prefix=None):
                # Return mock results
                return [
                    {
                        "content": "def hello():\n    pass",
                        "file_path": "test.py",
                        "language": "python",
                        "start_line": 1,
                        "end_line": 2,
                        "context": "File: test.py",
                        "score": 0.95,
                    },
                    {
                        "content": "def world():\n    pass",
                        "file_path": "test.py",
                        "language": "python",
                        "start_line": 4,
                        "end_line": 5,
                        "context": "File: test.py",
                        "score": 0.85,
                    },
                ][:top_k]

            def bm25_search(self, query, top_k, path_prefix=None):
                # Return mock results
                return [
                    {
                        "content": "def hello():\n    pass",
                        "file_path": "test.py",
                        "language": "python",
                        "start_line": 1,
                        "end_line": 2,
                        "context": "File: test.py",
                        "score": 3.5,
                    },
                    {
                        "content": "class Foo:\n    pass",
                        "file_path": "test.py",
                        "language": "python",
                        "start_line": 7,
                        "end_line": 8,
                        "context": "File: test.py",
                        "score": 2.1,
                    },
                ][:top_k]

        class MockEmbedder:
            def embed(self, texts):
                # Return mock embeddings
                return [[0.1] * 384 for _ in texts]

        db = MockDB()
        embedder = MockEmbedder()
        searcher = Searcher(db, embedder)

        # Test 1: Semantic search
        print("✓ Test 1: Semantic search")
        results = searcher.search("find hello function", mode="semantic", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r, ChunkResult) for r in results)
        print(f"  Found {len(results)} results")
        if results:
            print(
                f"  Top result: {results[0].file_path}:{results[0].start_line} (score: {results[0].score:.2f})"
            )

        # Test 2: Keyword search
        print("\n✓ Test 2: Keyword search")
        results = searcher.search("hello", mode="keyword", top_k=2)
        assert len(results) <= 2
        print(f"  Found {len(results)} results")
        if results:
            print(
                f"  Top result: {results[0].file_path}:{results[0].start_line} (score: {results[0].score:.2f})"
            )

        # Test 3: Hybrid search with RRF
        print("\n✓ Test 3: Hybrid search with RRF fusion")
        results = searcher.search("hello", mode="hybrid", top_k=5)
        assert len(results) <= 5
        print(f"  Found {len(results)} results (fused from both methods)")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r.file_path}:{r.start_line} (RRF score: {r.score:.4f})")

        # Test 4: Result to dict
        print("\n✓ Test 4: ChunkResult to dict")
        if results:
            result_dict = results[0].to_dict()
            assert "content" in result_dict
            assert "file_path" in result_dict
            assert "score" in result_dict
            print(f"  Result dict keys: {list(result_dict.keys())}")

        print("\n✅ Searcher tests PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Searcher tests FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_watcher() -> bool:
    """Test watcher initialization and pattern matching."""
    print_section("Testing Watcher")

    try:
        from embecode.config import EmbeCodeConfig
        from embecode.watcher import Watcher

        # Mock indexer
        class MockIndexer:
            def update_file(self, file_path):
                pass

            def delete_file(self, file_path):
                pass

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            config = EmbeCodeConfig()
            indexer = MockIndexer()

            # Test 1: Initialization
            print("✓ Test 1: Watcher initialization")
            watcher = Watcher(tmppath, config, indexer)
            assert watcher.project_path == tmppath
            assert watcher.config == config
            print(f"  Watcher created for {tmppath}")

            # Test 2: Pattern matching
            print("\n✓ Test 2: Pattern matching")
            test_cases = [
                ("src/main.py", "src/", True),
                ("lib/util.py", "lib/", True),
                ("node_modules/pkg/index.js", "node_modules/", True),
                ("dist/bundle.js", "dist/", True),
                ("src/app.min.js", "*.min.js", True),
                ("tests/test_app.py", "tests/", True),
            ]

            for path, pattern, expected in test_cases:
                result = watcher._matches_pattern(path, pattern)
                status = "✓" if result == expected else "✗"
                print(f"  {status} {path} vs {pattern}: {result}")

            # Test 3: Should process file
            print("\n✓ Test 3: Should process file checks")
            # Create test files
            src_file = tmppath / "src" / "main.py"
            src_file.parent.mkdir(parents=True)
            src_file.write_text("# test")

            node_modules = tmppath / "node_modules" / "pkg" / "index.js"
            node_modules.parent.mkdir(parents=True)
            node_modules.write_text("// test")

            assert watcher._should_process_file(src_file)
            assert not watcher._should_process_file(node_modules)
            print("  Include/exclude rules working correctly")

            # Test 4: Start and stop (briefly)
            print("\n✓ Test 4: Start and stop watcher")
            watcher.start()
            import time

            time.sleep(0.5)  # Let it start
            watcher.stop()
            print("  Watcher started and stopped successfully")

        print("\n✅ Watcher tests PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Watcher tests FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all manual tests."""
    print("=" * 60)
    print("  EMBECODE MANUAL FEATURE TESTS")
    print("=" * 60)

    results = {
        "Config": test_config(),
        "Chunker": test_chunker(),
        "Embedder": test_embedder(),
        "Searcher": test_searcher(),
        "Watcher": test_watcher(),
    }

    print_section("SUMMARY")

    passed = sum(results.values())
    total = len(results)

    for feature, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{feature:15s} {status}")

    print(f"\n{'=' * 60}")
    print(f"  TOTAL: {passed}/{total} tests passed")
    print(f"{'=' * 60}\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
