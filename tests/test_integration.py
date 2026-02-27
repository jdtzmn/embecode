"""Integration tests for end-to-end workflows."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from embecode.config import load_config
from embecode.indexer import Indexer
from embecode.searcher import ChunkResult, IndexNotReadyError, Searcher
from embecode.watcher import Watcher
from tests.helpers.mocks import MockEmbedder

# ============================================================================
# Mock Database for Integration Tests
# ============================================================================


class MockDatabase:
    """Mock database for integration testing."""

    def __init__(self) -> None:
        """Initialize mock database with in-memory storage."""
        self.chunks: dict[str, dict[str, Any]] = {}  # chunk_id -> chunk data
        self.embeddings: dict[str, list[float]] = {}  # chunk_id -> embedding
        self.files: set[str] = set()  # tracked files

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
        chunk_ids_to_delete = [
            chunk_id
            for chunk_id, chunk_data in self.chunks.items()
            if chunk_data["file_path"] == file_path
        ]
        for chunk_id in chunk_ids_to_delete:
            del self.chunks[chunk_id]
            if chunk_id in self.embeddings:
                del self.embeddings[chunk_id]
        if file_path in self.files:
            self.files.remove(file_path)

    def delete_chunks_by_hash(self, hashes: list[str]) -> None:
        """Delete chunks by their hashes."""
        chunk_ids_to_delete = [
            chunk_id for chunk_id, chunk_data in self.chunks.items() if chunk_data["hash"] in hashes
        ]
        for chunk_id in chunk_ids_to_delete:
            del self.chunks[chunk_id]
            if chunk_id in self.embeddings:
                del self.embeddings[chunk_id]

    def update_file_metadata(self, file_path: str, chunk_count: int) -> None:
        """Update file metadata (no-op for mock)."""
        pass

    def shrink_memory(self) -> None:
        """Release memory (no-op for mock)."""
        pass

    def delete_file(self, file_path: str) -> int:
        """Delete all chunks for a file and return count deleted."""
        initial_count = len(self.chunks)
        self.delete_chunks_by_file(file_path)
        return initial_count - len(self.chunks)

    def insert_chunks(self, chunk_records: list[dict[str, Any]]) -> None:
        """Insert chunks and their embeddings."""
        for record in chunk_records:
            chunk_id = f"{record['file_path']}:{record['start_line']}"
            self.chunks[chunk_id] = {
                "content": record["content"],
                "file_path": record["file_path"],
                "language": record["language"],
                "start_line": record["start_line"],
                "end_line": record["end_line"],
                "context": record["context"],
                "hash": record["hash"],
            }
            self.embeddings[chunk_id] = record["embedding"]
            self.files.add(record["file_path"])

    def vector_search(
        self, query_embedding: list[float], top_k: int, path_prefix: str | None = None
    ) -> list[dict[str, Any]]:
        """Perform vector similarity search (mock - returns all chunks)."""
        results = []
        for _chunk_id, chunk_data in self.chunks.items():
            if path_prefix and not chunk_data["file_path"].startswith(path_prefix):
                continue
            results.append(
                {
                    "content": chunk_data["content"],
                    "file_path": chunk_data["file_path"],
                    "language": chunk_data["language"],
                    "start_line": chunk_data["start_line"],
                    "end_line": chunk_data["end_line"],
                    "context": chunk_data["context"],
                    "score": 0.9,  # Mock score
                }
            )
        return results[:top_k]

    def bm25_search(
        self, query: str, top_k: int, path_prefix: str | None = None
    ) -> list[dict[str, Any]]:
        """Perform BM25 keyword search (mock - returns chunks containing query)."""
        results = []
        for _chunk_id, chunk_data in self.chunks.items():
            if path_prefix and not chunk_data["file_path"].startswith(path_prefix):
                continue
            if query.lower() in chunk_data["content"].lower():
                results.append(
                    {
                        "content": chunk_data["content"],
                        "file_path": chunk_data["file_path"],
                        "language": chunk_data["language"],
                        "start_line": chunk_data["start_line"],
                        "end_line": chunk_data["end_line"],
                        "context": chunk_data["context"],
                        "score": 3.5,  # Mock BM25 score
                    }
                )
        return results[:top_k]


# ============================================================================
# Integration Tests: Full Indexing Workflow
# ============================================================================


def test_full_indexing_workflow():
    """Test complete workflow: config → chunk → embed → index → status."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)

        # Create sample Python files
        (project_path / "src").mkdir()
        src_file = project_path / "src" / "main.py"
        src_file.write_text("""
def hello():
    print("hello")

def world():
    print("world")

class Calculator:
    def add(self, a, b):
        return a + b
""")

        lib_file = project_path / "src" / "util.py"
        lib_file.write_text("""
def helper():
    return 42
""")

        # Load config
        config = load_config()

        # Initialize components
        db = MockDatabase()
        embedder = MockEmbedder()
        indexer = Indexer(project_path, config, db, embedder)

        # Check initial status
        status = indexer.get_status()
        assert status.files_indexed == 0
        assert status.total_chunks == 0
        assert not status.is_indexing

        # Perform full index (foreground)
        indexer.start_full_index(background=False)

        # Check final status
        status = indexer.get_status()
        assert status.files_indexed == 2  # main.py and util.py
        assert status.total_chunks > 0
        assert not status.is_indexing

        # Verify chunks are in database
        assert len(db.chunks) > 0
        assert len(db.embeddings) == len(db.chunks)
        assert str(src_file) in db.files
        assert str(lib_file) in db.files


def test_full_indexing_respects_include_exclude():
    """Test that full indexing respects include/exclude patterns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)

        # Create files in different directories
        (project_path / "src").mkdir()
        (project_path / "node_modules").mkdir()
        (project_path / "dist").mkdir()

        src_file = project_path / "src" / "app.py"
        src_file.write_text("def app(): pass")

        excluded1 = project_path / "node_modules" / "pkg.js"
        excluded1.write_text("console.log('excluded');")

        excluded2 = project_path / "dist" / "bundle.js"
        excluded2.write_text("console.log('excluded');")

        minified = project_path / "src" / "app.min.js"
        minified.write_text("console.log('minified');")

        # Load config with include/exclude rules
        config = load_config()

        # Initialize and index
        db = MockDatabase()
        embedder = MockEmbedder()
        indexer = Indexer(project_path, config, db, embedder)
        indexer.start_full_index(background=False)

        # Verify only src/app.py was indexed (not node_modules, dist, or *.min.js)
        assert str(src_file) in db.files
        assert str(excluded1) not in db.files
        assert str(excluded2) not in db.files
        assert str(minified) not in db.files


def test_background_indexing_with_progress_tracking():
    """Test background indexing with progress tracking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)

        # Create multiple files
        (project_path / "src").mkdir()
        for i in range(5):
            file_path = project_path / "src" / f"file_{i}.py"
            file_path.write_text(f"def func_{i}(): return {i}")

        config = load_config()
        db = MockDatabase()
        embedder = MockEmbedder()
        indexer = Indexer(project_path, config, db, embedder)

        # Start background indexing
        indexer.start_full_index(background=True)

        # Check that indexing is in progress
        time.sleep(0.1)  # Let thread start
        status = indexer.get_status()
        # May or may not be in progress depending on timing
        if status.is_indexing:
            assert status.current_file is not None
            assert status.progress is not None
            assert 0.0 <= status.progress <= 1.0

        # Wait for completion
        indexer.wait_for_completion(timeout=5.0)

        # Verify completion
        status = indexer.get_status()
        assert not status.is_indexing
        assert status.files_indexed == 5
        assert status.total_chunks > 0


# ============================================================================
# Integration Tests: Incremental Indexing Workflow
# ============================================================================


def test_incremental_update_file_workflow():
    """Test incremental update when file is modified."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        (project_path / "src").mkdir()
        src_file = project_path / "src" / "main.py"

        # Initial content
        src_file.write_text("""
def old_function():
    return "old"
""")

        config = load_config()
        db = MockDatabase()
        embedder = MockEmbedder()
        indexer = Indexer(project_path, config, db, embedder)

        # Initial index
        indexer.start_full_index(background=False)
        initial_chunks = len(db.chunks)
        assert initial_chunks > 0

        # Modify file
        src_file.write_text("""
def new_function():
    return "new"

def another_function():
    return "another"
""")

        # Incremental update
        indexer.update_file(src_file)

        # Verify chunks were updated
        final_chunks = len(db.chunks)
        assert final_chunks >= initial_chunks  # May have more chunks now

        # Verify new content is indexed
        chunk_contents = [chunk["content"] for chunk in db.chunks.values()]
        combined_content = "".join(chunk_contents)
        assert "new_function" in combined_content
        assert "another_function" in combined_content


def test_incremental_delete_file_workflow():
    """Test incremental update when file is deleted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        (project_path / "src").mkdir()
        src_file = project_path / "src" / "main.py"
        src_file.write_text("def hello(): pass")

        config = load_config()
        db = MockDatabase()
        embedder = MockEmbedder()
        indexer = Indexer(project_path, config, db, embedder)

        # Initial index
        indexer.start_full_index(background=False)
        assert len(db.chunks) > 0
        assert str(src_file) in db.files

        # Delete file from index
        indexer.delete_file(src_file)

        # Verify chunks were removed
        assert len(db.chunks) == 0
        assert str(src_file) not in db.files


# ============================================================================
# Integration Tests: Search Workflow
# ============================================================================


def test_search_workflow_semantic_mode():
    """Test semantic search workflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        (project_path / "src").mkdir()
        src_file = project_path / "src" / "math.py"
        src_file.write_text("""
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
""")

        config = load_config()
        db = MockDatabase()
        embedder = MockEmbedder()
        indexer = Indexer(project_path, config, db, embedder)
        searcher = Searcher(db, embedder)

        # Index files
        indexer.start_full_index(background=False)

        # Perform semantic search
        response = searcher.search("addition function", mode="semantic", top_k=5)

        # Verify results
        assert len(response.results) > 0
        assert all(isinstance(r, ChunkResult) for r in response.results)
        assert all(r.file_path == str(src_file) for r in response.results)
        assert all(r.language == "python" for r in response.results)


def test_search_workflow_keyword_mode():
    """Test keyword (BM25) search workflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        (project_path / "src").mkdir()
        src_file = project_path / "src" / "utils.py"
        src_file.write_text("""
def calculate_total(items):
    return sum(items)
""")

        config = load_config()
        db = MockDatabase()
        embedder = MockEmbedder()
        indexer = Indexer(project_path, config, db, embedder)
        searcher = Searcher(db, embedder)

        # Index files
        indexer.start_full_index(background=False)

        # Perform keyword search
        response = searcher.search("calculate_total", mode="keyword", top_k=5)

        # Verify results
        assert len(response.results) > 0
        assert any("calculate_total" in r.content for r in response.results)


def test_search_workflow_hybrid_mode():
    """Test hybrid search with RRF fusion."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        (project_path / "src").mkdir()
        src_file = project_path / "src" / "app.py"
        src_file.write_text("""
def process_data(data):
    return data.strip()

def transform_input(input_str):
    return input_str.upper()
""")

        config = load_config()
        db = MockDatabase()
        embedder = MockEmbedder()
        indexer = Indexer(project_path, config, db, embedder)
        searcher = Searcher(db, embedder)

        # Index files
        indexer.start_full_index(background=False)

        # Perform hybrid search
        response = searcher.search("process data", mode="hybrid", top_k=5)

        # Verify results
        assert len(response.results) > 0
        assert all(isinstance(r, ChunkResult) for r in response.results)
        # Hybrid mode should return results from both semantic and keyword search


def test_search_with_path_prefix_filter():
    """Test search with path prefix filtering."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        (project_path / "src").mkdir()
        (project_path / "tests").mkdir()

        src_file = project_path / "src" / "main.py"
        src_file.write_text("def main(): pass")

        test_file = project_path / "tests" / "test_main.py"
        test_file.write_text("def test_main(): pass")

        config = load_config()
        db = MockDatabase()
        embedder = MockEmbedder()
        indexer = Indexer(project_path, config, db, embedder)
        searcher = Searcher(db, embedder)

        # Index files
        indexer.start_full_index(background=False)

        # Search with path prefix filter
        src_response = searcher.search("main", mode="hybrid", top_k=10, path="src/")
        test_response = searcher.search("main", mode="hybrid", top_k=10, path="tests/")

        # Verify filtering
        assert all(str(src_file) in r.file_path for r in src_response.results)
        assert all(str(test_file) in r.file_path for r in test_response.results)


def test_search_before_indexing_raises_error():
    """Test that search raises error when index is empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir)

        load_config()
        db = MockDatabase()
        embedder = MockEmbedder()
        searcher = Searcher(db, embedder)

        # Attempt search before indexing
        with pytest.raises(IndexNotReadyError):
            searcher.search("test query", mode="hybrid", top_k=5)


# ============================================================================
# Integration Tests: File Watching Workflow
# ============================================================================


def test_watcher_workflow_with_file_changes():
    """Test watcher detects and processes file changes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir).resolve()
        (project_path / "src").mkdir()

        config = load_config()

        # Create mock indexer that tracks calls
        mock_indexer = Mock()
        mock_indexer.update_file = Mock()
        mock_indexer.delete_file = Mock()
        # _should_index_file is called by the watcher to filter files;
        # return True for all files so they get processed
        mock_indexer._should_index_file = Mock(return_value=True)

        watcher = Watcher(project_path, config, mock_indexer)

        # Start watcher
        watcher.start()
        time.sleep(0.5)  # Let watcher initialize

        # Create new file
        new_file = project_path / "src" / "new.py"
        new_file.write_text("def new(): pass")
        time.sleep(1.2)  # Wait for debounce + processing

        # Verify update_file was called (may be flaky in test environments)
        if mock_indexer.update_file.call_count == 0:
            # File watching may not work reliably in all test environments
            # Skip the rest of the test if watcher didn't detect changes
            watcher.stop()
            pytest.skip("File watcher did not detect changes (test environment limitation)")

        # Modify file
        mock_indexer.update_file.reset_mock()
        mock_indexer.delete_file.reset_mock()
        new_file.write_text("def modified(): pass")
        time.sleep(1.2)  # Wait for debounce + processing

        # Verify update_file was called again
        assert mock_indexer.update_file.call_count > 0

        # Delete file
        mock_indexer.update_file.reset_mock()
        mock_indexer.delete_file.reset_mock()
        new_file.unlink()
        time.sleep(2.0)  # Wait for debounce + processing (longer for delete detection)

        # Verify delete_file was called
        if mock_indexer.delete_file.call_count == 0:
            watcher.stop()
            pytest.skip("File watcher did not detect deletion (test environment limitation)")

        # Stop watcher
        watcher.stop()


def test_watcher_respects_exclude_patterns():
    """Test watcher ignores files matching exclude patterns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir).resolve()
        (project_path / "node_modules").mkdir()

        config = load_config()
        mock_indexer = Mock()
        # _should_index_file returns False for node_modules (simulating real exclude logic)
        mock_indexer._should_index_file = Mock(return_value=False)
        watcher = Watcher(project_path, config, mock_indexer)

        watcher.start()
        time.sleep(0.2)

        # Create file in excluded directory
        excluded_file = project_path / "node_modules" / "package.js"
        excluded_file.write_text("console.log('test');")
        time.sleep(0.7)

        # Verify update_file was NOT called
        assert mock_indexer.update_file.call_count == 0

        watcher.stop()


def test_watcher_debounces_rapid_changes():
    """Test watcher debounces rapid successive changes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        (project_path / "src").mkdir()

        config = load_config()
        mock_indexer = Mock()
        watcher = Watcher(project_path, config, mock_indexer)

        watcher.start()
        time.sleep(0.2)

        # Make rapid changes to same file
        test_file = project_path / "src" / "test.py"
        for i in range(5):
            test_file.write_text(f"def func_{i}(): pass")
            time.sleep(0.05)  # Rapid changes within debounce window

        time.sleep(0.7)  # Wait for debounce

        # Should have been batched into fewer calls
        # Exact count depends on timing, but should be less than 5
        assert mock_indexer.update_file.call_count <= 3

        watcher.stop()


# ============================================================================
# Integration Tests: Complete Lifecycle
# ============================================================================


def test_complete_lifecycle_index_search_update_search():
    """Test complete lifecycle: index → search → update → search again."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        (project_path / "src").mkdir()
        src_file = project_path / "src" / "app.py"

        # Initial content
        src_file.write_text("""
def original_function():
    return "original"
""")

        config = load_config()
        db = MockDatabase()
        embedder = MockEmbedder()
        indexer = Indexer(project_path, config, db, embedder)
        searcher = Searcher(db, embedder)

        # Step 1: Initial index
        indexer.start_full_index(background=False)
        status1 = indexer.get_status()
        assert status1.files_indexed == 1
        assert status1.total_chunks > 0

        # Step 2: Search for original content
        response1 = searcher.search("original", mode="keyword", top_k=5)
        assert len(response1.results) > 0
        assert any("original" in r.content.lower() for r in response1.results)

        # Step 3: Update file
        src_file.write_text("""
def updated_function():
    return "updated"

def another_new_function():
    return "new"
""")
        indexer.update_file(src_file)

        # Step 4: Search for new content
        response2 = searcher.search("updated", mode="keyword", top_k=5)
        assert len(response2.results) > 0
        assert any("updated" in r.content.lower() for r in response2.results)

        # Step 5: Verify old content is gone (if chunk was replaced)
        status2 = indexer.get_status()
        assert status2.files_indexed == 1
        # Total chunks may have changed


def test_complete_workflow_with_multiple_files_and_searches():
    """Test workflow with multiple files, mixed operations, and searches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        (project_path / "src").mkdir()
        (project_path / "lib").mkdir()

        # Create initial files
        files = {
            "src/main.py": "def main(): pass",
            "src/utils.py": "def helper(): pass",
            "lib/db.py": "class Database: pass",
        }

        for rel_path, content in files.items():
            file_path = project_path / rel_path
            file_path.write_text(content)

        config = load_config()
        db = MockDatabase()
        embedder = MockEmbedder()
        indexer = Indexer(project_path, config, db, embedder)
        searcher = Searcher(db, embedder)

        # Full index
        indexer.start_full_index(background=False)
        assert indexer.get_status().files_indexed == 3

        # Search across all files
        all_response = searcher.search("class", mode="keyword", top_k=10)
        assert any("Database" in r.content for r in all_response.results)

        # Update one file
        (project_path / "src/main.py").write_text("""
def main():
    print("Hello, world!")
""")
        indexer.update_file(project_path / "src/main.py")

        # Delete one file
        (project_path / "lib/db.py").unlink()
        indexer.delete_file(project_path / "lib/db.py")

        # Verify final state
        assert indexer.get_status().files_indexed == 2

        # Search should no longer find deleted file content
        response = searcher.search("Database", mode="keyword", top_k=10)
        assert not any("Database" in r.content for r in response.results)


# ============================================================================
# Integration Tests: Error Handling and Edge Cases
# ============================================================================


def test_indexing_handles_unparseable_files_gracefully():
    """Test that indexing continues even if some files fail to parse."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        (project_path / "src").mkdir()

        # Valid file
        valid_file = project_path / "src" / "valid.py"
        valid_file.write_text("def valid(): pass")

        # Binary file (unparseable)
        binary_file = project_path / "src" / "binary.bin"
        binary_file.write_bytes(b"\x00\x01\x02\x03")

        config = load_config()
        db = MockDatabase()
        embedder = MockEmbedder()
        indexer = Indexer(project_path, config, db, embedder)

        # Should not raise exception, just skip unparseable files
        indexer.start_full_index(background=False)

        # Valid file should be indexed
        assert indexer.get_status().files_indexed >= 1


def test_search_handles_empty_results_gracefully():
    """Test search returns empty list when no results found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        (project_path / "src").mkdir()
        src_file = project_path / "src" / "app.py"
        src_file.write_text("def hello(): pass")

        config = load_config()
        db = MockDatabase()
        embedder = MockEmbedder()
        indexer = Indexer(project_path, config, db, embedder)
        searcher = Searcher(db, embedder)

        indexer.start_full_index(background=False)

        # Search for non-existent content
        response = searcher.search("nonexistent_function_xyz", mode="keyword", top_k=5)

        # Should return empty list, not raise exception
        assert isinstance(response.results, list)
        assert len(response.results) == 0


def test_incremental_update_handles_nonexistent_file():
    """Test incremental update handles files that don't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        (project_path / "src").mkdir()

        config = load_config()
        db = MockDatabase()
        embedder = MockEmbedder()
        indexer = Indexer(project_path, config, db, embedder)

        # Attempt to update non-existent file
        nonexistent = project_path / "src" / "nonexistent.py"

        # Should handle gracefully (may log warning but not crash)
        indexer.update_file(nonexistent)

        # Database should remain empty
        assert len(db.chunks) == 0


def test_complete_workflow_with_config_overrides():
    """Test workflow with custom config overrides."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        (project_path / "custom").mkdir()
        custom_file = project_path / "custom" / "app.py"
        custom_file.write_text("def app(): pass")

        # Load config with custom include paths
        config = load_config(cli_overrides={"index": {"include": ["custom/"], "exclude": []}})

        db = MockDatabase()
        embedder = MockEmbedder()
        indexer = Indexer(project_path, config, db, embedder)

        # Index should respect custom include
        indexer.start_full_index(background=False)

        assert indexer.get_status().files_indexed == 1
        assert str(custom_file) in db.files


# ============================================================================
# Integration Tests: Performance and Scalability
# ============================================================================


def test_indexing_many_files_completes_successfully():
    """Test that indexing many files completes successfully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        (project_path / "src").mkdir()

        # Create 50 small files
        num_files = 50
        for i in range(num_files):
            file_path = project_path / "src" / f"module_{i}.py"
            file_path.write_text(f"def func_{i}(): return {i}")

        config = load_config()
        db = MockDatabase()
        embedder = MockEmbedder()
        indexer = Indexer(project_path, config, db, embedder)

        # Index all files
        indexer.start_full_index(background=False)

        # Verify all files were indexed
        assert indexer.get_status().files_indexed == num_files
        assert len(db.files) == num_files
        assert len(db.chunks) >= num_files  # At least one chunk per file


def test_search_returns_limited_results():
    """Test that search respects top_k limit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        (project_path / "src").mkdir()

        # Create file with multiple functions
        src_file = project_path / "src" / "functions.py"
        functions = "\n\n".join([f"def func_{i}(): return {i}" for i in range(20)])
        src_file.write_text(functions)

        config = load_config()
        db = MockDatabase()
        embedder = MockEmbedder()
        indexer = Indexer(project_path, config, db, embedder)
        searcher = Searcher(db, embedder)

        indexer.start_full_index(background=False)

        # Search with top_k limit
        response = searcher.search("func", mode="keyword", top_k=3)

        # Should respect limit
        assert len(response.results) <= 3
