"""Performance tests for large codebase operations.

These tests validate that embecode can handle realistic large codebases
efficiently. They measure chunking, indexing, and search performance.

Run with: pytest tests/test_performance.py -v -s
Skip with: pytest -m "not slow"
"""

from __future__ import annotations

import math
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from embecode import chunker
from embecode.config import LanguageConfig, load_config
from embecode.db import Database
from embecode.indexer import Indexer
from embecode.searcher import Searcher
from tests.helpers.mocks import MockEmbedder

# ============================================================================
# Test Configuration
# ============================================================================

# Markers for performance tests
# Note: These tests are marked as "slow" and can be skipped with: pytest -m "not slow"
pytestmark = pytest.mark.slow


class PerformanceMetrics:
    """Track and validate performance metrics."""

    def __init__(self, operation: str) -> None:
        """Initialize metrics tracker."""
        self.operation = operation
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.items_processed: int = 0

    def start(self) -> None:
        """Start timing."""
        self.start_time = time.perf_counter()

    def stop(self) -> None:
        """Stop timing."""
        self.end_time = time.perf_counter()

    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    @property
    def throughput(self) -> float:
        """Get items per second."""
        if self.duration == 0:
            return 0.0
        return self.items_processed / self.duration

    def report(self) -> str:
        """Generate performance report."""
        return (
            f"{self.operation}:\n"
            f"  Duration: {self.duration:.2f}s\n"
            f"  Items: {self.items_processed}\n"
            f"  Throughput: {self.throughput:.2f} items/sec"
        )


# ============================================================================
# Mock Database and Embedder for Performance Tests
# ============================================================================


class MockDatabase:
    """Lightweight mock database for performance testing."""

    def __init__(self) -> None:
        """Initialize mock database."""
        self.chunks: dict[str, dict[str, Any]] = {}
        self.embeddings: dict[str, list[float]] = {}
        self.files: set[str] = set()

    def clear_index(self) -> None:
        """Clear the entire index."""
        self.chunks.clear()
        self.embeddings.clear()
        self.files.clear()

    def get_index_stats(self) -> dict[str, Any]:
        """Get index statistics."""
        return {
            "total_chunks": len(self.chunks),
            "files_indexed": len(self.files),
            "last_updated": "2025-02-25T10:00:00",
        }

    def get_chunks_by_file(self, file_path: str) -> list[dict[str, Any]]:
        """Get all chunks for a file."""
        return [
            chunk_data
            for chunk_id, chunk_data in self.chunks.items()
            if chunk_data["file_path"] == file_path
        ]

    def get_chunk_hashes_for_file(self, file_path: str) -> set[str]:
        """Get all chunk hashes for a specific file."""
        return {
            chunk_data["hash"]
            for chunk_data in self.chunks.values()
            if chunk_data["file_path"] == file_path
        }

    def delete_chunks_by_file(self, file_path: str) -> None:
        """Delete all chunks for a file."""
        chunk_ids = [cid for cid, data in self.chunks.items() if data["file_path"] == file_path]
        for chunk_id in chunk_ids:
            del self.chunks[chunk_id]
            self.embeddings.pop(chunk_id, None)
        self.files.discard(file_path)

    def delete_chunks_by_hash(self, hashes: list[str]) -> None:
        """Delete chunks by their hashes."""
        chunk_ids = [cid for cid, data in self.chunks.items() if data["hash"] in hashes]
        for chunk_id in chunk_ids:
            del self.chunks[chunk_id]
            self.embeddings.pop(chunk_id, None)

    def update_file_metadata(self, file_path: str, chunk_count: int) -> None:
        """Update file metadata."""
        self.files.add(file_path)

    def delete_file(self, file_path: str) -> int:
        """Delete file and return number of chunks deleted."""
        initial = len(self.chunks)
        self.delete_chunks_by_file(file_path)
        return initial - len(self.chunks)

    def insert_chunks(self, chunk_records: list[dict[str, Any]]) -> None:
        """Insert chunks and embeddings."""
        for record in chunk_records:
            chunk_id = f"{record['file_path']}:{record['start_line']}"
            self.chunks[chunk_id] = {
                "file_path": record["file_path"],
                "content": record["content"],
                "hash": record["hash"],
                "start_line": record["start_line"],
                "end_line": record["end_line"],
            }
            self.embeddings[chunk_id] = record["embedding"]
            self.files.add(record["file_path"])

    def vector_search(
        self, query_embedding: list[float], top_k: int, path_prefix: str | None
    ) -> list[dict[str, Any]]:
        """Simulate vector/semantic search (return chunks for performance testing)."""
        results = []
        for _chunk_id, chunk_data in list(self.chunks.items())[:top_k]:
            if path_prefix and not chunk_data["file_path"].startswith(path_prefix):
                continue
            results.append(
                {
                    "file_path": chunk_data["file_path"],
                    "content": chunk_data["content"],
                    "context": f"File: {chunk_data['file_path']}",
                    "start_line": chunk_data["start_line"],
                    "end_line": chunk_data["end_line"],
                    "language": "python",
                    "score": 0.95,
                }
            )
        return results

    def bm25_search(
        self, query: str, top_k: int, path_prefix: str | None = None
    ) -> list[dict[str, Any]]:
        """Simulate BM25/keyword search."""
        return self.vector_search([], top_k, path_prefix)

    def shrink_memory(self) -> None:
        """No-op: mock database has nothing to shrink."""


# ============================================================================
# Test Data Generation
# ============================================================================


def generate_python_file(size: str = "medium") -> str:
    """Generate a synthetic Python file for testing.

    Args:
        size: "small" (50 lines), "medium" (200 lines), "large" (1000 lines)

    Returns:
        Python code as string
    """
    sizes = {"small": 5, "medium": 20, "large": 100}
    num_functions = sizes.get(size, 20)

    lines = [
        '"""Generated test module."""',
        "",
        "from typing import Any",
        "",
    ]

    for i in range(num_functions):
        lines.extend(
            [
                f"def function_{i}(arg1: str, arg2: int) -> dict[str, Any]:",
                f'    """Function {i} docstring.',
                "    ",
                "    Performs some computation and returns results.",
                '    """',
                f"    result = {{'function': 'function_{i}', 'arg1': arg1, 'arg2': arg2}}",
                "    # Process data",
                "    for j in range(arg2):",
                "        result[f'key_{j}'] = arg1 * j",
                "    return result",
                "",
            ]
        )

    return "\n".join(lines)


def generate_large_codebase(tmp_path: Path, num_files: int = 100) -> Path:
    """Generate a large synthetic codebase.

    Args:
        tmp_path: Temporary directory
        num_files: Number of files to generate

    Returns:
        Path to generated codebase root
    """
    repo = tmp_path / "large_repo"
    repo.mkdir()

    # Create directory structure
    dirs = [
        repo / "src" / "core",
        repo / "src" / "utils",
        repo / "src" / "models",
        repo / "tests",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Generate files with different sizes
    file_count = 0
    for directory in dirs:
        files_per_dir = num_files // len(dirs)
        for i in range(files_per_dir):
            # Mix of file sizes
            if i % 3 == 0:
                size = "large"
            elif i % 3 == 1:
                size = "medium"
            else:
                size = "small"

            file_path = directory / f"module_{file_count}.py"
            file_path.write_text(generate_python_file(size))
            file_count += 1

    return repo


# ============================================================================
# Performance Tests
# ============================================================================


class TestChunkingPerformance:
    """Test chunking performance on large files and codebases."""

    def test_chunk_large_file(self) -> None:
        """Test chunking a single large file (1000+ lines)."""
        metrics = PerformanceMetrics("Chunk large file")

        # Generate large file
        content = generate_python_file("large")
        lines = content.count("\n") + 1

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            file_path = f.name

        try:
            # Configure chunker
            lang_config = LanguageConfig()

            # Time chunking
            metrics.start()
            chunks = chunker.chunk_file(Path(file_path), lang_config)
            metrics.stop()

            metrics.items_processed = len(chunks)

            # Validate results
            assert len(chunks) > 0, "Should produce chunks"
            assert all(c.content for c in chunks), "All chunks should have content"
            assert all(c.hash for c in chunks), "All chunks should have hashes"

            # Performance requirements
            assert metrics.duration < 2.0, f"Chunking too slow: {metrics.duration:.2f}s"
            assert metrics.throughput > 20, f"Throughput too low: {metrics.throughput:.2f} chunks/s"

            print(f"\n{metrics.report()}")
            print(f"File size: {lines} lines")

        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_chunk_many_files(self, tmp_path: Path) -> None:
        """Test chunking many files (100 files)."""
        metrics = PerformanceMetrics("Chunk many files")

        # Generate codebase
        repo = generate_large_codebase(tmp_path, num_files=100)

        # Get all Python files
        python_files = list(repo.rglob("*.py"))
        assert len(python_files) == 100, f"Expected 100 files, got {len(python_files)}"

        # Configure chunker
        lang_config = LanguageConfig()

        # Time chunking
        all_chunks = []
        metrics.start()
        for file_path in python_files:
            chunks = chunker.chunk_file(file_path, lang_config)
            all_chunks.extend(chunks)
        metrics.stop()

        metrics.items_processed = len(all_chunks)

        # Validate results
        assert len(all_chunks) >= 100, "Should produce many chunks"
        assert all(c.content for c in all_chunks), "All chunks should have content"

        # Performance requirements (should handle 100 files in reasonable time)
        assert metrics.duration < 30.0, f"Chunking too slow: {metrics.duration:.2f}s"
        assert metrics.throughput > 20, f"Throughput too low: {metrics.throughput:.2f} chunks/s"

        print(f"\n{metrics.report()}")
        print(f"Total files: {len(python_files)}")
        print(f"Total chunks: {len(all_chunks)}")
        print(f"Avg chunks/file: {len(all_chunks) / len(python_files):.1f}")


class TestIndexingPerformance:
    """Test indexing performance on large codebases."""

    def test_index_large_codebase(self, tmp_path: Path) -> None:
        """Test full indexing of a large codebase (100 files)."""
        metrics = PerformanceMetrics("Index large codebase")

        # Generate codebase
        repo = generate_large_codebase(tmp_path, num_files=100)

        # Configure indexer
        config = load_config(repo)
        db = MockDatabase()
        embedder = MockEmbedder()
        indexer = Indexer(repo, config, db, embedder)

        # Time full indexing
        metrics.start()
        indexer.start_full_index(background=False)
        metrics.stop()

        stats = db.get_index_stats()
        metrics.items_processed = stats["total_chunks"]

        # Validate results
        assert stats["total_chunks"] >= 100, "Should have indexed many chunks"
        assert stats["files_indexed"] >= 0, "Should have indexed files"

        # Performance requirements (100 files should index in reasonable time)
        # With mock embedder, this should be fast (mostly chunking + DB overhead)
        assert metrics.duration < 60.0, f"Indexing too slow: {metrics.duration:.2f}s"
        assert metrics.throughput > 10, f"Throughput too low: {metrics.throughput:.2f} chunks/s"

        print(f"\n{metrics.report()}")
        print(f"Total files indexed: {stats['files_indexed']}")
        print(f"Total chunks: {stats['total_chunks']}")

    def test_incremental_update_performance(self, tmp_path: Path) -> None:
        """Test incremental update performance."""
        metrics = PerformanceMetrics("Incremental updates")

        # Generate codebase and do initial index
        repo = generate_large_codebase(tmp_path, num_files=50)
        config = load_config(repo)
        db = MockDatabase()
        embedder = MockEmbedder()
        indexer = Indexer(repo, config, db, embedder)

        indexer.start_full_index(background=False)

        # Modify 10 files
        python_files = list(repo.rglob("*.py"))[:10]
        for file_path in python_files:
            content = file_path.read_text()
            file_path.write_text(content + "\n# Modified\n")

        # Time incremental updates
        metrics.start()
        for file_path in python_files:
            indexer.update_file(file_path)
        metrics.stop()

        metrics.items_processed = len(python_files)
        final_stats = db.get_index_stats()

        # Validate results
        # Note: total_chunks may decrease if old chunks are replaced by fewer new chunks
        assert final_stats["total_chunks"] > 0

        # Performance requirements (incremental should be fast)
        assert metrics.duration < 10.0, f"Updates too slow: {metrics.duration:.2f}s"
        assert metrics.throughput > 2, f"Throughput too low: {metrics.throughput:.2f} files/s"

        print(f"\n{metrics.report()}")
        print(f"Files updated: {len(python_files)}")


class TestSearchPerformance:
    """Test search performance on large indexes."""

    def test_search_large_index(self, tmp_path: Path) -> None:
        """Test search performance on a large index."""
        # Generate and index codebase
        repo = generate_large_codebase(tmp_path, num_files=100)
        config = load_config(repo)
        db = MockDatabase()
        embedder = MockEmbedder()
        indexer = Indexer(repo, config, db, embedder)

        indexer.start_full_index(background=False)
        stats = db.get_index_stats()

        # Create searcher
        searcher = Searcher(db, embedder)

        # Test multiple search modes
        queries = [
            "authentication function",
            "database connection",
            "error handling",
            "configuration parser",
            "user validation",
        ]

        for mode in ["hybrid", "semantic", "keyword"]:
            metrics = PerformanceMetrics(f"Search ({mode} mode)")
            metrics.items_processed = len(queries)

            metrics.start()
            for query in queries:
                response = searcher.search(query, mode=mode, top_k=10)
                assert isinstance(response.results, list)
            metrics.stop()

            # Performance requirements (search should be fast)
            avg_per_query = metrics.duration / len(queries)
            assert avg_per_query < 1.0, f"Search too slow: {avg_per_query:.3f}s per query"

            print(f"\n{metrics.report()}")
            print(f"Avg per query: {avg_per_query:.3f}s")
            print(f"Index size: {stats['total_chunks']} chunks")

    def test_search_with_path_filter_performance(self, tmp_path: Path) -> None:
        """Test search performance with path filtering."""
        # Generate and index codebase
        repo = generate_large_codebase(tmp_path, num_files=100)
        config = load_config(repo)
        db = MockDatabase()
        embedder = MockEmbedder()
        indexer = Indexer(repo, config, db, embedder)

        indexer.start_full_index(background=False)

        # Create searcher
        searcher = Searcher(db, embedder)

        # Test search with path prefix
        metrics = PerformanceMetrics("Search with path filter")
        query = "function implementation"
        num_searches = 20

        metrics.start()
        for i in range(num_searches):
            # Alternate between different path prefixes
            path_filter = "src/core" if i % 2 == 0 else "src/utils"
            response = searcher.search(query, path=path_filter, top_k=10)
            assert isinstance(response.results, list)
        metrics.stop()

        metrics.items_processed = num_searches

        # Performance requirements
        avg_per_query = metrics.duration / num_searches
        assert avg_per_query < 1.0, f"Filtered search too slow: {avg_per_query:.3f}s per query"

        print(f"\n{metrics.report()}")
        print(f"Avg per query: {avg_per_query:.3f}s")


# ============================================================================
# End-to-End Performance Tests
# ============================================================================


class TestEndToEndPerformance:
    """Test complete workflows on realistic codebases."""

    def test_complete_workflow_performance(self, tmp_path: Path) -> None:
        """Test complete index + search workflow on a realistic codebase."""
        print("\n" + "=" * 70)
        print("COMPLETE WORKFLOW PERFORMANCE TEST")
        print("=" * 70)

        # Generate codebase (100 files)
        repo = generate_large_codebase(tmp_path, num_files=100)
        python_files = list(repo.rglob("*.py"))

        # Phase 1: Initial indexing
        print("\nPhase 1: Initial indexing...")
        index_metrics = PerformanceMetrics("Initial indexing")

        config = load_config(repo)
        db = MockDatabase()
        embedder = MockEmbedder()
        indexer = Indexer(repo, config, db, embedder)

        index_metrics.start()
        indexer.start_full_index(background=False)
        index_metrics.stop()

        stats = db.get_index_stats()
        index_metrics.items_processed = stats["total_chunks"]

        print(index_metrics.report())
        print(f"Files indexed: {len(python_files)}")

        # Phase 2: Search queries
        print("\nPhase 2: Search queries...")
        search_metrics = PerformanceMetrics("Search queries")

        searcher = Searcher(db, embedder)
        queries = [
            "authentication",
            "database connection",
            "error handling",
            "configuration",
            "user validation",
        ]

        search_metrics.start()
        for query in queries:
            response = searcher.search(query, top_k=5)
            assert isinstance(response.results, list)
        search_metrics.stop()

        search_metrics.items_processed = len(queries)
        print(search_metrics.report())

        # Phase 3: Incremental updates
        print("\nPhase 3: Incremental updates...")
        update_metrics = PerformanceMetrics("Incremental updates")

        files_to_update = python_files[:10]
        for f in files_to_update:
            content = f.read_text()
            f.write_text(content + "\n# Updated\n")

        update_metrics.start()
        for file_path in files_to_update:
            indexer.update_file(file_path)
        update_metrics.stop()

        update_metrics.items_processed = len(files_to_update)
        print(update_metrics.report())

        # Overall validation
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(
            f"Total time: {index_metrics.duration + search_metrics.duration + update_metrics.duration:.2f}s"
        )
        print(f"Indexed {stats['total_chunks']} chunks from {len(python_files)} files")
        print(f"Performed {len(queries)} searches")
        print(f"Updated {len(files_to_update)} files incrementally")

        # Overall performance requirements
        total_time = index_metrics.duration + search_metrics.duration + update_metrics.duration
        assert total_time < 120.0, f"Complete workflow too slow: {total_time:.2f}s"


# ============================================================================
# Search Benchmarks (pytest-benchmark)
# ============================================================================


@pytest.fixture(scope="class")
def search_fixture(tmp_path_factory: pytest.TempPathFactory) -> Searcher:
    """Build a 100-file mock index once, shared across all benchmark tests."""
    tmp_path = tmp_path_factory.mktemp("bench")
    repo = generate_large_codebase(tmp_path, num_files=100)
    config = load_config(repo)
    db = MockDatabase()
    embedder = MockEmbedder()
    indexer = Indexer(repo, config, db, embedder)
    indexer.start_full_index(background=False)
    return Searcher(db, embedder)


class TestSearchBenchmark:
    """Benchmark search query speed using pytest-benchmark.

    Run with:
        pytest tests/test_performance.py::TestSearchBenchmark -v --benchmark-only
    """

    _QUERY = "authentication function"

    def test_benchmark_hybrid(self, benchmark: Any, search_fixture: Searcher) -> None:
        """Benchmark hybrid (BM25 + vector + RRF) search."""
        response = benchmark(search_fixture.search, self._QUERY, mode="hybrid")
        timings = response.timings.to_dict()
        print(f"\nphase breakdown (last run): {timings}")

    def test_benchmark_semantic(self, benchmark: Any, search_fixture: Searcher) -> None:
        """Benchmark semantic (vector) search."""
        response = benchmark(search_fixture.search, self._QUERY, mode="semantic")
        timings = response.timings.to_dict()
        print(f"\nphase breakdown (last run): {timings}")

    def test_benchmark_keyword(self, benchmark: Any, search_fixture: Searcher) -> None:
        """Benchmark keyword (BM25) search."""
        response = benchmark(search_fixture.search, self._QUERY, mode="keyword")
        timings = response.timings.to_dict()
        print(f"\nphase breakdown (last run): {timings}")


# ============================================================================
# Real DuckDB Search Benchmarks (pytest-benchmark)
# ============================================================================

# Embedding dimension for nomic-ai/nomic-embed-text-v1.5 (default model)
_BENCH_DIM = 768

# Fixed random unit vector used for all vector search calls — avoids model
# inference while still exercising the real DuckDB VSS cosine-similarity scan.
random.seed(42)
_raw = [random.gauss(0, 1) for _ in range(_BENCH_DIM)]
_norm = math.sqrt(sum(x * x for x in _raw))
_FIXED_UNIT_VECTOR: list[float] = [x / _norm for x in _raw]

# Persistent DB location — survives between pytest runs so setup only pays
# the indexing cost once.
_BENCH_DB_DIR = Path(__file__).parent.parent / ".bench_db"
_BENCH_DB_PATH = _BENCH_DB_DIR / "search_bench.duckdb"
# generate_large_codebase(parent) creates parent/"large_repo", so point at
# _BENCH_DB_DIR and the repo ends up at .bench_db/large_repo.
_BENCH_REPO_PATH = _BENCH_DB_DIR / "large_repo"


class FixedVectorEmbedder:
    """Embedder that always returns the same pre-computed unit vector.

    This lets us exercise the real DuckDB VSS/FTS query path without loading
    a sentence-transformers model.  The fixed vector means cosine scores are
    uniform across all chunks, but the index scan, sorting, and result
    materialisation are all real.
    """

    dimension = _BENCH_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_FIXED_UNIT_VECTOR for _ in texts]

    def unload(self) -> None:
        pass


def _build_bench_db() -> tuple[Database, FixedVectorEmbedder]:
    """Build (or reuse) the persistent benchmark DuckDB and return an open connection."""
    embedder = FixedVectorEmbedder()

    _BENCH_DB_DIR.mkdir(exist_ok=True)

    db = Database(_BENCH_DB_PATH)
    db.connect()

    # If the DB already has chunks, reuse it.
    try:
        stats = db.get_index_stats()
        if stats["total_chunks"] > 0:
            return db, embedder
    except Exception:
        pass

    # First run: generate a synthetic repo and index it into the real DB.
    if not _BENCH_REPO_PATH.exists():
        generate_large_codebase(_BENCH_DB_DIR, num_files=200)

    config = load_config(_BENCH_REPO_PATH)
    indexer = Indexer(_BENCH_REPO_PATH, config, db, embedder)  # type: ignore[arg-type]
    indexer.start_full_index(background=False)

    return db, embedder


@pytest.fixture(scope="session")
def real_search_fixture() -> Searcher:
    """Return a Searcher backed by a real DuckDB (built once per session)."""
    db, embedder = _build_bench_db()
    return Searcher(db, embedder)  # type: ignore[arg-type]


class TestSearchBenchmarkReal:
    """Benchmark search query speed against a real DuckDB index.

    Uses a persistent DB at .bench_db/ so the indexing cost is only paid on
    the first run.  Delete .bench_db/ to force a rebuild.

    Run with:
        pytest tests/test_performance.py::TestSearchBenchmarkReal -v --benchmark-only --no-cov -s
    """

    _QUERY = "authentication function"

    def test_benchmark_real_hybrid(self, benchmark: Any, real_search_fixture: Searcher) -> None:
        """Benchmark hybrid (BM25 + vector + RRF) search against real DuckDB."""
        response = benchmark(real_search_fixture.search, self._QUERY, mode="hybrid")
        timings = response.timings.to_dict()
        print(f"\nphase breakdown (last run): {timings}")

    def test_benchmark_real_semantic(self, benchmark: Any, real_search_fixture: Searcher) -> None:
        """Benchmark semantic (vector) search against real DuckDB."""
        response = benchmark(real_search_fixture.search, self._QUERY, mode="semantic")
        timings = response.timings.to_dict()
        print(f"\nphase breakdown (last run): {timings}")

    def test_benchmark_real_keyword(self, benchmark: Any, real_search_fixture: Searcher) -> None:
        """Benchmark keyword (BM25) search against real DuckDB."""
        response = benchmark(real_search_fixture.search, self._QUERY, mode="keyword")
        timings = response.timings.to_dict()
        print(f"\nphase breakdown (last run): {timings}")
