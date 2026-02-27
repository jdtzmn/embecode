"""Tests for server.py - FastMCP server implementation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from embecode.indexer import IndexStatus
from embecode.searcher import ChunkResult, IndexNotReadyError, SearchResponse, SearchTimings
from embecode.server import (
    EmbeCodeServer,
    EmbeddingModelChangedError,
    get_server,
    initialize_server,
)


@pytest.fixture
def temp_project(tmp_path: Path) -> Path:
    """Create a temporary project directory with some files."""
    project = tmp_path / "project"
    project.mkdir()

    # Create some Python files
    (project / "main.py").write_text("def hello():\n    print('hello')\n")
    (project / "utils.py").write_text("def add(a, b):\n    return a + b\n")

    return project


@pytest.fixture
def mock_config() -> Mock:
    """Create a mock config."""
    config = Mock()
    config.embeddings.model = "test-model"
    config.daemon.auto_watch = False
    config.daemon.debounce_ms = 500
    return config


@pytest.fixture
def mock_db() -> Mock:
    """Create a mock database."""
    db = Mock()
    db.get_index_stats.return_value = {"total_chunks": 0, "files_indexed": 0, "last_updated": None}
    db.connect.return_value = None
    db.close.return_value = None
    db.get_metadata.return_value = None  # First run by default
    db.set_metadata.return_value = None
    return db


@pytest.fixture
def mock_embedder() -> Mock:
    """Create a mock embedder."""
    embedder = Mock()
    embedder.embed.return_value = [[0.1, 0.2, 0.3]]
    return embedder


@pytest.fixture
def mock_indexer() -> Mock:
    """Create a mock indexer."""
    indexer = Mock()
    status = IndexStatus(
        files_indexed=10,
        total_chunks=50,
        embedding_model="test-model",
        last_updated="2024-01-01T00:00:00",
        is_indexing=False,
    )
    indexer.get_status.return_value = status
    indexer.index_full.return_value = None
    return indexer


@pytest.fixture
def mock_searcher() -> Mock:
    """Create a mock searcher."""
    searcher = Mock()
    return searcher


class TestEmbeCodeServer:
    """Tests for EmbeCodeServer class."""

    @patch("embecode.server.load_config")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.Database")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.threading.Thread")
    def test_initialization_empty_index(
        self,
        mock_thread: Mock,
        mock_indexer_class: Mock,
        mock_searcher_class: Mock,
        mock_embedder_class: Mock,
        mock_db_class: Mock,
        mock_cache_manager_class: Mock,
        mock_load_config: Mock,
        temp_project: Path,
        mock_config: Mock,
        mock_db: Mock,
    ) -> None:
        """Test server initialization with empty index starts background catch-up."""
        # Setup mocks
        mock_load_config.return_value = mock_config
        mock_cache_manager = Mock()
        cache_dir = temp_project / ".cache"
        cache_dir.mkdir()
        mock_cache_manager.get_cache_dir.return_value = cache_dir
        mock_cache_manager_class.return_value = mock_cache_manager

        mock_db.get_index_stats.return_value = {"total_chunks": 0}
        mock_db_class.return_value = mock_db

        # Initialize server
        server = EmbeCodeServer(temp_project)

        # Verify initialization
        assert server.project_path == temp_project.resolve()
        mock_load_config.assert_called_once_with(temp_project.resolve())
        mock_cache_manager.get_cache_dir.assert_called_once()
        mock_db.connect.assert_called_once()

        # Verify model was stored (first run)
        mock_db.set_metadata.assert_called_once_with("embedding_model", "test-model")

        # Verify background thread was started for catch-up indexing
        mock_thread.assert_called_once()
        _args, kwargs = mock_thread.call_args
        assert kwargs["daemon"] is True

    @patch("embecode.server.load_config")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.Database")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.threading.Thread")
    def test_initialization_existing_index(
        self,
        mock_thread: Mock,
        mock_indexer_class: Mock,
        mock_searcher_class: Mock,
        mock_embedder_class: Mock,
        mock_db_class: Mock,
        mock_cache_manager_class: Mock,
        mock_load_config: Mock,
        temp_project: Path,
        mock_config: Mock,
        mock_db: Mock,
    ) -> None:
        """Test server initialization with existing index still spawns catch-up thread."""
        # Setup mocks
        mock_load_config.return_value = mock_config
        mock_config.daemon.auto_watch = True

        mock_cache_manager = Mock()
        cache_dir = temp_project / ".cache"
        cache_dir.mkdir()
        mock_cache_manager.get_cache_dir.return_value = cache_dir
        mock_cache_manager_class.return_value = mock_cache_manager

        mock_db.get_index_stats.return_value = {"total_chunks": 100}
        mock_db.get_metadata.return_value = "test-model"  # Model already stored
        mock_db_class.return_value = mock_db

        # Initialize server
        EmbeCodeServer(temp_project)

        # Verify background catch-up thread was still started (unified path)
        mock_thread.assert_called_once()
        _args, kwargs = mock_thread.call_args
        assert kwargs["daemon"] is True

    @patch("embecode.server.EmbeCodeConfig")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.Database")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.threading.Thread")
    def test_search_code_success(
        self,
        mock_thread: Mock,
        mock_indexer_class: Mock,
        mock_searcher_class: Mock,
        mock_embedder_class: Mock,
        mock_db_class: Mock,
        mock_cache_manager_class: Mock,
        mock_config_class: Mock,
        temp_project: Path,
        mock_config: Mock,
        mock_db: Mock,
        mock_searcher: Mock,
    ) -> None:
        """Test successful code search."""
        # Setup mocks
        mock_config_class.load.return_value = mock_config
        mock_cache_manager = Mock()
        cache_dir = temp_project / ".cache"
        cache_dir.mkdir()
        mock_cache_manager.get_cache_dir.return_value = cache_dir
        mock_cache_manager_class.return_value = mock_cache_manager

        mock_db.get_index_stats.return_value = {"total_chunks": 100}
        mock_db_class.return_value = mock_db

        # Setup searcher mock
        result = ChunkResult(
            content="def hello():\n    print('hello')",
            file_path="main.py",
            language="python",
            start_line=1,
            end_line=2,
            definitions="function hello",
            score=0.95,
        )
        mock_searcher.search.return_value = [result]
        mock_searcher_class.return_value = mock_searcher

        # Initialize server
        server = EmbeCodeServer(temp_project)

        # Search
        results = server.search_code("hello function", mode="semantic", top_k=5)

        # Verify
        assert len(results) == 1
        assert results[0]["file_path"] == "main.py"
        assert results[0]["definitions"] == "function hello"
        assert results[0]["preview"] == "def hello():\n    print('hello')"
        assert "content" not in results[0]
        mock_searcher.search.assert_called_once_with(
            "hello function", mode="semantic", top_k=5, path=None
        )

    @patch("embecode.server.EmbeCodeConfig")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.Database")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.threading.Thread")
    def test_search_code_not_ready(
        self,
        mock_thread: Mock,
        mock_indexer_class: Mock,
        mock_searcher_class: Mock,
        mock_embedder_class: Mock,
        mock_db_class: Mock,
        mock_cache_manager_class: Mock,
        mock_config_class: Mock,
        temp_project: Path,
        mock_config: Mock,
        mock_db: Mock,
        mock_searcher: Mock,
        mock_indexer: Mock,
    ) -> None:
        """Test search when index is not ready."""
        # Setup mocks
        mock_config_class.load.return_value = mock_config
        mock_cache_manager = Mock()
        cache_dir = temp_project / ".cache"
        cache_dir.mkdir()
        mock_cache_manager.get_cache_dir.return_value = cache_dir
        mock_cache_manager_class.return_value = mock_cache_manager

        mock_db.get_index_stats.return_value = {"total_chunks": 0}
        mock_db_class.return_value = mock_db

        # Setup searcher to raise IndexNotReadyError
        mock_searcher.search.side_effect = IndexNotReadyError("Index not ready")
        mock_searcher_class.return_value = mock_searcher

        # Setup indexer status
        status = IndexStatus(
            files_indexed=5,
            total_chunks=0,
            embedding_model="test-model",
            last_updated=None,
            is_indexing=True,
            current_file="test.py",
            progress=0.25,
        )
        mock_indexer.get_status.return_value = status
        mock_indexer_class.return_value = mock_indexer

        # Initialize server
        server = EmbeCodeServer(temp_project)

        # Search should raise with progress info
        with pytest.raises(IndexNotReadyError, match=r"5 files processed.*25% complete"):
            server.search_code("test query")

    @patch("embecode.server.EmbeCodeConfig")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.Database")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.threading.Thread")
    def test_get_index_status(
        self,
        mock_thread: Mock,
        mock_indexer_class: Mock,
        mock_searcher_class: Mock,
        mock_embedder_class: Mock,
        mock_db_class: Mock,
        mock_cache_manager_class: Mock,
        mock_config_class: Mock,
        temp_project: Path,
        mock_config: Mock,
        mock_db: Mock,
        mock_indexer: Mock,
    ) -> None:
        """Test getting index status."""
        # Setup mocks
        mock_config_class.load.return_value = mock_config
        mock_cache_manager = Mock()
        cache_dir = temp_project / ".cache"
        cache_dir.mkdir()
        mock_cache_manager.get_cache_dir.return_value = cache_dir
        mock_cache_manager_class.return_value = mock_cache_manager

        mock_db.get_index_stats.return_value = {"total_chunks": 100}
        mock_db_class.return_value = mock_db

        # Setup indexer status
        status = IndexStatus(
            files_indexed=10,
            total_chunks=50,
            embedding_model="test-model",
            last_updated="2024-01-01T00:00:00",
            is_indexing=False,
        )
        mock_indexer.get_status.return_value = status
        mock_indexer_class.return_value = mock_indexer

        # Initialize server
        server = EmbeCodeServer(temp_project)

        # Get status
        result = server.get_index_status()

        # Verify
        assert result["files_indexed"] == 10
        assert result["total_chunks"] == 50
        assert result["embedding_model"] == "test-model"
        assert result["is_indexing"] is False

    @patch("embecode.server.EmbeCodeConfig")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.Database")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.Watcher")
    @patch("embecode.server.threading.Thread")
    def test_cleanup(
        self,
        mock_thread: Mock,
        mock_watcher_class: Mock,
        mock_indexer_class: Mock,
        mock_searcher_class: Mock,
        mock_embedder_class: Mock,
        mock_db_class: Mock,
        mock_cache_manager_class: Mock,
        mock_config_class: Mock,
        temp_project: Path,
        mock_config: Mock,
        mock_db: Mock,
    ) -> None:
        """Test server cleanup stops watcher and closes database."""
        # Setup mocks
        mock_config_class.load.return_value = mock_config
        mock_config.daemon.auto_watch = True

        mock_cache_manager = Mock()
        cache_dir = temp_project / ".cache"
        cache_dir.mkdir()
        mock_cache_manager.get_cache_dir.return_value = cache_dir
        mock_cache_manager_class.return_value = mock_cache_manager

        mock_db.get_index_stats.return_value = {"total_chunks": 100}
        mock_db_class.return_value = mock_db

        mock_watcher = Mock()
        mock_watcher_class.return_value = mock_watcher

        # Initialize server (thread is mocked, so _catchup_index doesn't run)
        server = EmbeCodeServer(temp_project)

        # Manually run _catchup_index to start watcher (simulating thread completion)
        server._catchup_index()

        # Cleanup
        server.cleanup()

        # Verify
        mock_watcher.stop.assert_called_once()
        mock_db.close.assert_called_once()

    @patch("embecode.server.EmbeCodeConfig")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.Database")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.threading.Thread")
    def test_search_with_path_filter(
        self,
        mock_thread: Mock,
        mock_indexer_class: Mock,
        mock_searcher_class: Mock,
        mock_embedder_class: Mock,
        mock_db_class: Mock,
        mock_cache_manager_class: Mock,
        mock_config_class: Mock,
        temp_project: Path,
        mock_config: Mock,
        mock_db: Mock,
        mock_searcher: Mock,
    ) -> None:
        """Test search with path prefix filter."""
        # Setup mocks
        mock_config_class.load.return_value = mock_config
        mock_cache_manager = Mock()
        cache_dir = temp_project / ".cache"
        cache_dir.mkdir()
        mock_cache_manager.get_cache_dir.return_value = cache_dir
        mock_cache_manager_class.return_value = mock_cache_manager

        mock_db.get_index_stats.return_value = {"total_chunks": 100}
        mock_db_class.return_value = mock_db

        mock_searcher.search.return_value = []
        mock_searcher_class.return_value = mock_searcher

        # Initialize server
        server = EmbeCodeServer(temp_project)

        # Search with path filter
        server.search_code("test", path="src/")

        # Verify path was passed to searcher
        mock_searcher.search.assert_called_once_with("test", mode="hybrid", top_k=10, path="src/")

    @patch("embecode.server.EmbeCodeConfig")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.Database")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.threading.Thread")
    def test_search_code_returns_concise_results(
        self,
        mock_thread: Mock,
        mock_indexer_class: Mock,
        mock_searcher_class: Mock,
        mock_embedder_class: Mock,
        mock_db_class: Mock,
        mock_cache_manager_class: Mock,
        mock_config_class: Mock,
        temp_project: Path,
        mock_config: Mock,
        mock_db: Mock,
        mock_searcher: Mock,
    ) -> None:
        """Test search_code returns concise format with required keys, no content field.

        Allows optional fields like match_lines and file_result_count.
        """
        # Setup mocks
        mock_config_class.load.return_value = mock_config
        mock_cache_manager = Mock()
        cache_dir = temp_project / ".cache"
        cache_dir.mkdir()
        mock_cache_manager.get_cache_dir.return_value = cache_dir
        mock_cache_manager_class.return_value = mock_cache_manager

        mock_db.get_index_stats.return_value = {"total_chunks": 100}
        mock_db_class.return_value = mock_db

        # Setup searcher mock with a result that has content
        result = ChunkResult(
            content="def hello():\n    print('hello')",
            file_path="main.py",
            language="python",
            start_line=1,
            end_line=2,
            definitions="function hello",
            score=0.95,
        )
        mock_searcher.search.return_value = SearchResponse(
            results=[result],
            timings=SearchTimings(),
        )
        mock_searcher_class.return_value = mock_searcher

        # Initialize server and search
        server = EmbeCodeServer(temp_project)
        results = server.search_code("hello function")

        # Verify all required keys are present
        required_keys = {
            "file_path",
            "language",
            "start_line",
            "end_line",
            "definitions",
            "preview",
            "score",
        }
        optional_keys = {"match_lines", "file_result_count"}
        assert required_keys <= set(results[0].keys())
        assert set(results[0].keys()) <= required_keys | optional_keys

        # Verify content field is excluded (concise format)
        assert "content" not in results[0]

    @patch("embecode.server.EmbeCodeConfig")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.Database")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.threading.Thread")
    def test_search_code_default_top_k_is_10(
        self,
        mock_thread: Mock,
        mock_indexer_class: Mock,
        mock_searcher_class: Mock,
        mock_embedder_class: Mock,
        mock_db_class: Mock,
        mock_cache_manager_class: Mock,
        mock_config_class: Mock,
        temp_project: Path,
        mock_config: Mock,
        mock_db: Mock,
        mock_searcher: Mock,
    ) -> None:
        """Test search_code defaults to top_k=10 and can return up to 10 results."""
        # Setup mocks
        mock_config_class.load.return_value = mock_config
        mock_cache_manager = Mock()
        cache_dir = temp_project / ".cache"
        cache_dir.mkdir()
        mock_cache_manager.get_cache_dir.return_value = cache_dir
        mock_cache_manager_class.return_value = mock_cache_manager

        mock_db.get_index_stats.return_value = {"total_chunks": 100}
        mock_db_class.return_value = mock_db

        # Create 10 mock results
        mock_results = [
            ChunkResult(
                content=f"def func_{i}():\n    pass",
                file_path=f"file_{i}.py",
                language="python",
                start_line=1,
                end_line=2,
                definitions=f"function func_{i}",
                score=1.0 - i * 0.1,
            )
            for i in range(10)
        ]
        mock_searcher.search.return_value = mock_results
        mock_searcher_class.return_value = mock_searcher

        # Initialize server and search without explicit top_k
        server = EmbeCodeServer(temp_project)
        results = server.search_code("test query")

        # Verify searcher was called with top_k=10
        mock_searcher.search.assert_called_once_with(
            "test query", mode="hybrid", top_k=10, path=None
        )

        # Verify all 10 results are returned
        assert len(results) == 10

    @patch("embecode.server.EmbeCodeConfig")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.Database")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.threading.Thread")
    def test_search_code_file_result_count_for_duplicates(
        self,
        mock_thread: Mock,
        mock_indexer_class: Mock,
        mock_searcher_class: Mock,
        mock_embedder_class: Mock,
        mock_db_class: Mock,
        mock_cache_manager_class: Mock,
        mock_config_class: Mock,
        temp_project: Path,
        mock_config: Mock,
        mock_db: Mock,
        mock_searcher: Mock,
    ) -> None:
        """Two results from same file get file_result_count, unique file result does not."""
        mock_config_class.load.return_value = mock_config
        mock_cache_manager = Mock()
        cache_dir = temp_project / ".cache"
        cache_dir.mkdir()
        mock_cache_manager.get_cache_dir.return_value = cache_dir
        mock_cache_manager_class.return_value = mock_cache_manager

        mock_db.get_index_stats.return_value = {"total_chunks": 100}
        mock_db_class.return_value = mock_db

        # Two results from auth.py, one from main.py
        results_list = [
            ChunkResult(
                content="def login():\n    pass",
                file_path="src/auth.py",
                language="python",
                start_line=1,
                end_line=5,
                definitions="function login",
                score=0.9,
            ),
            ChunkResult(
                content="def logout():\n    pass",
                file_path="src/auth.py",
                language="python",
                start_line=10,
                end_line=15,
                definitions="function logout",
                score=0.8,
            ),
            ChunkResult(
                content="def main():\n    pass",
                file_path="src/main.py",
                language="python",
                start_line=1,
                end_line=3,
                definitions="function main",
                score=0.7,
            ),
        ]
        mock_searcher.search.return_value = SearchResponse(
            results=results_list,
            timings=SearchTimings(),
        )
        mock_searcher_class.return_value = mock_searcher

        server = EmbeCodeServer(temp_project)
        results = server.search_code("auth functions")

        # Both auth.py results should have file_result_count: 2
        assert results[0]["file_path"] == "src/auth.py"
        assert results[0]["file_result_count"] == 2
        assert results[1]["file_path"] == "src/auth.py"
        assert results[1]["file_result_count"] == 2

        # main.py result should NOT have file_result_count
        assert results[2]["file_path"] == "src/main.py"
        assert "file_result_count" not in results[2]

    @patch("embecode.server.EmbeCodeConfig")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.Database")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.threading.Thread")
    def test_search_code_file_result_count_absent_for_unique_files(
        self,
        mock_thread: Mock,
        mock_indexer_class: Mock,
        mock_searcher_class: Mock,
        mock_embedder_class: Mock,
        mock_db_class: Mock,
        mock_cache_manager_class: Mock,
        mock_config_class: Mock,
        temp_project: Path,
        mock_config: Mock,
        mock_db: Mock,
        mock_searcher: Mock,
    ) -> None:
        """When all results are from different files, no result has file_result_count."""
        mock_config_class.load.return_value = mock_config
        mock_cache_manager = Mock()
        cache_dir = temp_project / ".cache"
        cache_dir.mkdir()
        mock_cache_manager.get_cache_dir.return_value = cache_dir
        mock_cache_manager_class.return_value = mock_cache_manager

        mock_db.get_index_stats.return_value = {"total_chunks": 100}
        mock_db_class.return_value = mock_db

        results_list = [
            ChunkResult(
                content="def foo():\n    pass",
                file_path="a.py",
                language="python",
                start_line=1,
                end_line=2,
                definitions="function foo",
                score=0.9,
            ),
            ChunkResult(
                content="def bar():\n    pass",
                file_path="b.py",
                language="python",
                start_line=1,
                end_line=2,
                definitions="function bar",
                score=0.8,
            ),
        ]
        mock_searcher.search.return_value = SearchResponse(
            results=results_list,
            timings=SearchTimings(),
        )
        mock_searcher_class.return_value = mock_searcher

        server = EmbeCodeServer(temp_project)
        results = server.search_code("functions")

        for r in results:
            assert "file_result_count" not in r

    @patch("embecode.server.EmbeCodeConfig")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.Database")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.threading.Thread")
    def test_search_code_passes_query_to_to_dict(
        self,
        mock_thread: Mock,
        mock_indexer_class: Mock,
        mock_searcher_class: Mock,
        mock_embedder_class: Mock,
        mock_db_class: Mock,
        mock_cache_manager_class: Mock,
        mock_config_class: Mock,
        temp_project: Path,
        mock_config: Mock,
        mock_db: Mock,
        mock_searcher: Mock,
    ) -> None:
        """Verify the query string flows through from search_code to to_dict."""
        mock_config_class.load.return_value = mock_config
        mock_cache_manager = Mock()
        cache_dir = temp_project / ".cache"
        cache_dir.mkdir()
        mock_cache_manager.get_cache_dir.return_value = cache_dir
        mock_cache_manager_class.return_value = mock_cache_manager

        mock_db.get_index_stats.return_value = {"total_chunks": 100}
        mock_db_class.return_value = mock_db

        # Use a real ChunkResult with content that matches the query
        result = ChunkResult(
            content="import os\nimport sys\ndef hello():\n    print('hello world')",
            file_path="main.py",
            language="python",
            start_line=1,
            end_line=4,
            definitions="function hello",
            score=0.95,
        )
        mock_searcher.search.return_value = SearchResponse(
            results=[result],
            timings=SearchTimings(),
        )
        mock_searcher_class.return_value = mock_searcher

        server = EmbeCodeServer(temp_project)
        results = server.search_code("hello")

        # Query "hello" matches line 3 and line 4, so match_lines should be present
        assert "match_lines" in results[0]
        assert 3 in results[0]["match_lines"]  # "def hello():"
        assert 4 in results[0]["match_lines"]  # "print('hello world')"

        # Preview should be match-aware (showing hello lines, not import lines)
        preview = results[0]["preview"]
        assert "hello" in preview


class TestCatchUpStartup:
    """Tests for catch-up indexing startup logic and embedding model detection."""

    @patch("embecode.server.load_config")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.Database")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.threading.Thread")
    def test_startup_always_spawns_catchup_thread(
        self,
        mock_thread: Mock,
        mock_indexer_class: Mock,
        mock_searcher_class: Mock,
        mock_embedder_class: Mock,
        mock_db_class: Mock,
        mock_cache_manager_class: Mock,
        mock_load_config: Mock,
        temp_project: Path,
        mock_config: Mock,
        mock_db: Mock,
    ) -> None:
        """Both empty and non-empty DBs result in a background thread being spawned."""
        mock_load_config.return_value = mock_config
        mock_cache_manager = Mock()
        cache_dir = temp_project / ".cache"
        cache_dir.mkdir()
        mock_cache_manager.get_cache_dir.return_value = cache_dir
        mock_cache_manager_class.return_value = mock_cache_manager

        # Test with empty DB
        mock_db.get_index_stats.return_value = {"total_chunks": 0}
        mock_db_class.return_value = mock_db
        EmbeCodeServer(temp_project)
        assert mock_thread.call_count == 1

        # Test with non-empty DB
        mock_thread.reset_mock()
        mock_db.get_index_stats.return_value = {"total_chunks": 100}
        EmbeCodeServer(temp_project)
        assert mock_thread.call_count == 1

    @patch("embecode.server.load_config")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.Database")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.Watcher")
    @patch("embecode.server.threading.Thread")
    def test_startup_catchup_starts_watcher_after_completion(
        self,
        mock_thread: Mock,
        mock_watcher_class: Mock,
        mock_indexer_class: Mock,
        mock_searcher_class: Mock,
        mock_embedder_class: Mock,
        mock_db_class: Mock,
        mock_cache_manager_class: Mock,
        mock_load_config: Mock,
        temp_project: Path,
        mock_config: Mock,
        mock_db: Mock,
    ) -> None:
        """Catch-up thread completes. Watcher is started afterward."""
        mock_load_config.return_value = mock_config
        mock_config.daemon.auto_watch = True
        mock_cache_manager = Mock()
        cache_dir = temp_project / ".cache"
        cache_dir.mkdir()
        mock_cache_manager.get_cache_dir.return_value = cache_dir
        mock_cache_manager_class.return_value = mock_cache_manager
        mock_db_class.return_value = mock_db

        server = EmbeCodeServer(temp_project)

        # Watcher not started yet (thread is mocked)
        mock_watcher_class.assert_not_called()

        # Manually run _catchup_index to simulate thread completion
        server._catchup_index()

        # Now watcher should be started
        mock_watcher_class.assert_called_once()

    @patch("embecode.server.load_config")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.Database")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.Watcher")
    @patch("embecode.server.threading.Thread")
    def test_startup_catchup_failure_still_starts_watcher(
        self,
        mock_thread: Mock,
        mock_watcher_class: Mock,
        mock_indexer_class: Mock,
        mock_searcher_class: Mock,
        mock_embedder_class: Mock,
        mock_db_class: Mock,
        mock_cache_manager_class: Mock,
        mock_load_config: Mock,
        temp_project: Path,
        mock_config: Mock,
        mock_db: Mock,
    ) -> None:
        """Catch-up thread raises an exception. Watcher is still started."""
        mock_load_config.return_value = mock_config
        mock_config.daemon.auto_watch = True
        mock_cache_manager = Mock()
        cache_dir = temp_project / ".cache"
        cache_dir.mkdir()
        mock_cache_manager.get_cache_dir.return_value = cache_dir
        mock_cache_manager_class.return_value = mock_cache_manager
        mock_db_class.return_value = mock_db

        # Make catch-up raise an exception
        mock_indexer = mock_indexer_class.return_value
        mock_indexer.start_catchup_index.side_effect = RuntimeError("indexing failed")

        server = EmbeCodeServer(temp_project)

        # Manually run _catchup_index (simulating thread)
        server._catchup_index()  # Should not raise

        # Watcher should still be started despite the failure
        mock_watcher_class.assert_called_once()

    @patch("embecode.server.load_config")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.Database")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.threading.Thread")
    def test_startup_embedding_model_match_proceeds(
        self,
        mock_thread: Mock,
        mock_indexer_class: Mock,
        mock_searcher_class: Mock,
        mock_embedder_class: Mock,
        mock_db_class: Mock,
        mock_cache_manager_class: Mock,
        mock_load_config: Mock,
        temp_project: Path,
        mock_config: Mock,
        mock_db: Mock,
    ) -> None:
        """Stored model matches config model. Server initializes normally."""
        mock_load_config.return_value = mock_config
        mock_cache_manager = Mock()
        cache_dir = temp_project / ".cache"
        cache_dir.mkdir()
        mock_cache_manager.get_cache_dir.return_value = cache_dir
        mock_cache_manager_class.return_value = mock_cache_manager

        # Stored model matches configured model
        mock_db.get_metadata.return_value = "test-model"
        mock_db_class.return_value = mock_db

        # Should not raise
        server = EmbeCodeServer(temp_project)
        assert server is not None

        # set_metadata should NOT be called (model already stored and matches)
        mock_db.set_metadata.assert_not_called()

    @patch("embecode.server.load_config")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.Database")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.threading.Thread")
    def test_startup_embedding_model_mismatch_refuses_to_start(
        self,
        mock_thread: Mock,
        mock_indexer_class: Mock,
        mock_searcher_class: Mock,
        mock_embedder_class: Mock,
        mock_db_class: Mock,
        mock_cache_manager_class: Mock,
        mock_load_config: Mock,
        temp_project: Path,
        mock_config: Mock,
        mock_db: Mock,
    ) -> None:
        """Stored model differs from config model. Raises EmbeddingModelChangedError."""
        mock_load_config.return_value = mock_config
        mock_cache_manager = Mock()
        cache_dir = temp_project / ".cache"
        cache_dir.mkdir()
        mock_cache_manager.get_cache_dir.return_value = cache_dir
        mock_cache_manager_class.return_value = mock_cache_manager

        # Stored model differs from configured model ("test-model")
        mock_db.get_metadata.return_value = "model-a"
        mock_db_class.return_value = mock_db

        with pytest.raises(EmbeddingModelChangedError) as exc_info:
            EmbeCodeServer(temp_project)

        error_msg = str(exc_info.value)
        assert "model-a" in error_msg
        assert "test-model" in error_msg
        assert "index.db" in error_msg

        # Thread should NOT have been started
        mock_thread.assert_not_called()

    @patch("embecode.server.load_config")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.Database")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.threading.Thread")
    def test_startup_no_stored_model_stores_current(
        self,
        mock_thread: Mock,
        mock_indexer_class: Mock,
        mock_searcher_class: Mock,
        mock_embedder_class: Mock,
        mock_db_class: Mock,
        mock_cache_manager_class: Mock,
        mock_load_config: Mock,
        temp_project: Path,
        mock_config: Mock,
        mock_db: Mock,
    ) -> None:
        """First run (no metadata). Server stores current model in DB and proceeds."""
        mock_load_config.return_value = mock_config
        mock_cache_manager = Mock()
        cache_dir = temp_project / ".cache"
        cache_dir.mkdir()
        mock_cache_manager.get_cache_dir.return_value = cache_dir
        mock_cache_manager_class.return_value = mock_cache_manager

        # No stored model (first run)
        mock_db.get_metadata.return_value = None
        mock_db_class.return_value = mock_db

        server = EmbeCodeServer(temp_project)
        assert server is not None

        # Model should be stored
        mock_db.set_metadata.assert_called_once_with("embedding_model", "test-model")

        # Catch-up thread should still be spawned
        mock_thread.assert_called_once()

    @patch("embecode.server.load_config")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.Database")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.threading.Thread")
    def test_startup_stores_model_before_catchup(
        self,
        mock_thread: Mock,
        mock_indexer_class: Mock,
        mock_searcher_class: Mock,
        mock_embedder_class: Mock,
        mock_db_class: Mock,
        mock_cache_manager_class: Mock,
        mock_load_config: Mock,
        temp_project: Path,
        mock_config: Mock,
        mock_db: Mock,
    ) -> None:
        """Model is stored during __init__ (main thread), not after catch-up."""
        mock_load_config.return_value = mock_config
        mock_cache_manager = Mock()
        cache_dir = temp_project / ".cache"
        cache_dir.mkdir()
        mock_cache_manager.get_cache_dir.return_value = cache_dir
        mock_cache_manager_class.return_value = mock_cache_manager

        # Track call order
        call_order: list[str] = []
        mock_db.get_metadata.return_value = None
        mock_db.set_metadata.side_effect = lambda k, v: call_order.append("set_metadata")
        mock_db_class.return_value = mock_db

        def thread_start_side_effect():
            call_order.append("thread_start")

        mock_thread_instance = Mock()
        mock_thread_instance.start.side_effect = thread_start_side_effect
        mock_thread.return_value = mock_thread_instance

        EmbeCodeServer(temp_project)

        # set_metadata must happen before thread.start()
        assert call_order == ["set_metadata", "thread_start"]


class TestGlobalServerManagement:
    """Tests for global server instance management."""

    def test_get_server_not_initialized(self) -> None:
        """Test get_server raises when server not initialized."""
        # Reset global server
        import embecode.server

        embecode.server._server = None

        with pytest.raises(RuntimeError, match="Server not initialized"):
            get_server()

    @patch("embecode.server.EmbeCodeServer")
    def test_initialize_server(self, mock_server_class: Mock, temp_project: Path) -> None:
        """Test initialize_server creates global instance."""
        # Reset global server
        import embecode.server

        embecode.server._server = None

        mock_server = Mock()
        mock_server_class.return_value = mock_server

        # Initialize
        result = initialize_server(temp_project)

        # Verify
        assert result == mock_server
        mock_server_class.assert_called_once_with(temp_project)

    @patch("embecode.server.EmbeCodeServer")
    def test_initialize_server_already_initialized(
        self, mock_server_class: Mock, temp_project: Path
    ) -> None:
        """Test initialize_server returns existing instance if already initialized."""
        # Create mock server
        mock_server = Mock()

        # Set global server
        import embecode.server

        embecode.server._server = mock_server

        # Initialize again
        result = initialize_server(temp_project)

        # Verify returns existing instance
        assert result == mock_server
        # Should not create new instance
        mock_server_class.assert_not_called()


class TestMCPToolFunctions:
    """Tests for MCP tool functions."""

    @patch("embecode.server.get_server")
    def test_search_code_tool_success(self, mock_get_server: Mock) -> None:
        """Test search_code tool function."""
        from embecode.server import search_code

        # Setup mock server
        mock_server = Mock()
        mock_server.search_code.return_value = [
            {
                "file_path": "test.py",
                "language": "python",
                "start_line": 1,
                "end_line": 2,
                "definitions": "",
                "preview": "test",
                "score": 0.9,
            }
        ]
        mock_get_server.return_value = mock_server

        # Call tool
        results = search_code("test query", mode="semantic", top_k=10)

        # Verify
        assert len(results) == 1
        assert results[0]["file_path"] == "test.py"
        mock_server.search_code.assert_called_once_with(
            "test query", mode="semantic", top_k=10, path=None
        )

    @patch("embecode.server.get_server")
    def test_search_code_tool_not_ready(self, mock_get_server: Mock) -> None:
        """Test search_code tool when index not ready."""
        from embecode.server import search_code

        # Setup mock server
        mock_server = Mock()
        mock_server.search_code.side_effect = IndexNotReadyError("Index not ready")
        mock_get_server.return_value = mock_server

        # Call tool - should return error dict, not raise
        results = search_code("test query")

        # Verify error response
        assert len(results) == 1
        assert "error" in results[0]
        assert "Index not ready" in results[0]["error"]
        assert results[0]["retry_recommended"] is True

    @patch("embecode.server.get_server")
    def test_index_status_tool(self, mock_get_server: Mock) -> None:
        """Test index_status tool function."""
        from embecode.server import index_status

        # Setup mock server
        mock_server = Mock()
        mock_server.get_index_status.return_value = {
            "files_indexed": 10,
            "total_chunks": 50,
            "embedding_model": "test-model",
            "last_updated": "2024-01-01T00:00:00",
            "is_indexing": False,
            "current_file": None,
            "progress": None,
        }
        mock_get_server.return_value = mock_server

        # Call tool
        result = index_status()

        # Verify
        assert result["files_indexed"] == 10
        assert result["total_chunks"] == 50
        assert result["embedding_model"] == "test-model"
        assert result["is_indexing"] is False
        mock_server.get_index_status.assert_called_once()

    @patch("embecode.server.get_server")
    def test_search_code_default_top_k_is_10(self, mock_get_server: Mock) -> None:
        """Test search_code tool defaults to top_k=10 when not explicitly passed."""
        from embecode.server import search_code

        # Setup mock server
        mock_server = Mock()
        mock_server.search_code.return_value = []
        mock_get_server.return_value = mock_server

        # Call tool without specifying top_k
        search_code("test query")

        # Verify top_k defaults to 10
        mock_server.search_code.assert_called_once_with(
            "test query", mode="hybrid", top_k=10, path=None
        )


class TestRunServer:
    """Tests for run_server function."""

    @patch("embecode.server.initialize_server")
    @patch("embecode.server.mcp")
    def test_run_server_success(
        self, mock_mcp: Mock, mock_initialize: Mock, temp_project: Path
    ) -> None:
        """Test run_server initializes and runs MCP server."""
        from embecode.server import run_server

        # Setup mocks
        mock_server = Mock()
        mock_initialize.return_value = mock_server
        mock_mcp.run.side_effect = KeyboardInterrupt()  # Exit after starting

        # Mock get_server to return our mock server
        with patch("embecode.server.get_server", return_value=mock_server):
            # Run server - KeyboardInterrupt is caught and cleanup should run
            run_server(temp_project)

        # Verify
        mock_initialize.assert_called_once_with(temp_project)
        mock_mcp.run.assert_called_once()
        mock_server.cleanup.assert_called_once()

    @patch("embecode.server.initialize_server")
    def test_run_server_initialization_failure(
        self, mock_initialize: Mock, temp_project: Path
    ) -> None:
        """Test run_server exits on initialization failure."""
        from embecode.server import run_server

        # Setup mock to raise exception
        mock_initialize.side_effect = Exception("Init failed")

        # Run server should exit with code 1
        with pytest.raises(SystemExit) as exc_info:
            run_server(temp_project)

        assert exc_info.value.code == 1
