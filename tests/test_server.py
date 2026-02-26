"""Tests for server.py - FastMCP server implementation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from embecode.indexer import IndexStatus
from embecode.searcher import ChunkResult, IndexNotReadyError
from embecode.server import EmbeCodeServer, get_server, initialize_server


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
        """Test server initialization with empty index starts background indexing."""
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

        # Verify background thread was started for indexing
        mock_thread.assert_called_once()
        _args, kwargs = mock_thread.call_args
        assert kwargs["daemon"] is True

    @patch("embecode.server.EmbeCodeConfig")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.Database")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.Watcher")
    def test_initialization_existing_index(
        self,
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
        """Test server initialization with existing index doesn't reindex."""
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

        # Initialize server
        EmbeCodeServer(temp_project)

        # Verify watcher was started
        mock_watcher_class.assert_called_once()

    @patch("embecode.server.EmbeCodeConfig")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.Database")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Indexer")
    def test_search_code_success(
        self,
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
            context="",
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
        assert results[0]["content"] == "def hello():\n    print('hello')"
        assert results[0]["file_path"] == "main.py"
        mock_searcher.search.assert_called_once_with(
            "hello function", mode="semantic", top_k=5, path=None
        )

    @patch("embecode.server.EmbeCodeConfig")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.Database")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Indexer")
    def test_search_code_not_ready(
        self,
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
    def test_get_index_status(
        self,
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
    def test_cleanup(
        self,
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

        # Initialize server
        server = EmbeCodeServer(temp_project)

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
    def test_search_with_path_filter(
        self,
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
        mock_searcher.search.assert_called_once_with("test", mode="hybrid", top_k=5, path="src/")


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
                "content": "test",
                "file_path": "test.py",
                "language": "python",
                "start_line": 1,
                "end_line": 2,
                "context": "",
                "score": 0.9,
            }
        ]
        mock_get_server.return_value = mock_server

        # Call tool
        results = search_code("test query", mode="semantic", top_k=10)

        # Verify
        assert len(results) == 1
        assert results[0]["content"] == "test"
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
