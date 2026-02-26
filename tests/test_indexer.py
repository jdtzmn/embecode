"""Tests for indexer.py - indexing orchestration."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from embecode.chunker import Chunk
from embecode.config import (
    EmbeCodeConfig,
    EmbeddingsConfig,
    IndexConfig,
    LanguageConfig,
)
from embecode.indexer import Indexer, IndexingError, IndexingInProgressError, IndexStatus


class TestIndexStatus:
    """Test suite for IndexStatus."""

    def test_to_dict(self) -> None:
        """Should convert status to dictionary."""
        status = IndexStatus(
            files_indexed=10,
            total_chunks=100,
            embedding_model="test-model",
            last_updated="2025-02-25T10:00:00",
            is_indexing=False,
        )

        result = status.to_dict()

        assert result["files_indexed"] == 10
        assert result["total_chunks"] == 100
        assert result["embedding_model"] == "test-model"
        assert result["last_updated"] == "2025-02-25T10:00:00"
        assert result["is_indexing"] is False
        assert result["current_file"] is None
        assert result["progress"] is None

    def test_to_dict_with_progress(self) -> None:
        """Should include progress information when indexing."""
        status = IndexStatus(
            files_indexed=5,
            total_chunks=50,
            embedding_model="test-model",
            last_updated="2025-02-25T10:00:00",
            is_indexing=True,
            current_file="src/main.py",
            progress=0.5,
        )

        result = status.to_dict()

        assert result["is_indexing"] is True
        assert result["current_file"] == "src/main.py"
        assert result["progress"] == 0.5


class TestIndexer:
    """Test suite for Indexer."""

    @pytest.fixture
    def temp_project(self) -> Path:
        """Create a temporary project directory with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Create test files
            (project_path / "src").mkdir()
            (project_path / "src" / "main.py").write_text("print('hello')")
            (project_path / "src" / "utils.py").write_text("def foo(): pass")
            (project_path / "tests").mkdir()
            (project_path / "tests" / "test_main.py").write_text("def test_foo(): pass")
            (project_path / "node_modules").mkdir()
            (project_path / "node_modules" / "lib.js").write_text("console.log('lib')")

            yield project_path

    @pytest.fixture
    def mock_config(self) -> EmbeCodeConfig:
        """Create a mock config."""
        config = Mock(spec=EmbeCodeConfig)
        config.index = IndexConfig(
            include=["src/", "tests/"],
            exclude=["node_modules/", "*.min.js"],
            languages=LanguageConfig(python=1500, default=1000),
        )
        config.embeddings = EmbeddingsConfig(model="test-model")
        return config

    @pytest.fixture
    def mock_db(self) -> Mock:
        """Create a mock database."""
        db = Mock()
        db.get_index_stats.return_value = {
            "files_indexed": 0,
            "total_chunks": 0,
            "last_updated": None,
        }
        db.get_chunk_hashes_for_file.return_value = set()
        db.delete_chunks_by_hash.return_value = 0
        db.delete_file.return_value = 0
        db.clear_index.return_value = None
        db.insert_chunks.return_value = None
        db.update_file_metadata.return_value = None
        return db

    @pytest.fixture
    def mock_embedder(self) -> Mock:
        """Create a mock embedder."""
        embedder = Mock()
        embedder.embed.return_value = [[0.1, 0.2, 0.3]]
        return embedder

    def test_get_status_not_indexing(
        self, temp_project: Path, mock_config: EmbeCodeConfig, mock_db: Mock, mock_embedder: Mock
    ) -> None:
        """Should return status when not indexing."""
        mock_db.get_index_stats.return_value = {
            "files_indexed": 10,
            "total_chunks": 100,
            "last_updated": "2025-02-25T10:00:00",
        }

        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)
        status = indexer.get_status()

        assert status.files_indexed == 10
        assert status.total_chunks == 100
        assert status.embedding_model == "test-model"
        assert status.last_updated == "2025-02-25T10:00:00"
        assert status.is_indexing is False

    def test_get_status_while_indexing(
        self, temp_project: Path, mock_config: EmbeCodeConfig, mock_db: Mock, mock_embedder: Mock
    ) -> None:
        """Should return progress when indexing."""
        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)

        # Simulate indexing state
        with indexer._lock:
            indexer._is_indexing = True
            indexer._current_file = "src/main.py"
            indexer._progress = 0.5

        status = indexer.get_status()

        assert status.is_indexing is True
        assert status.current_file == "src/main.py"
        assert status.progress == 0.5

    def test_should_index_file_included(
        self, mock_config: EmbeCodeConfig, mock_db: Mock, mock_embedder: Mock, temp_project: Path
    ) -> None:
        """Should index files matching include patterns."""
        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)

        assert indexer._should_index_file(temp_project / "src" / "main.py") is True
        assert indexer._should_index_file(temp_project / "tests" / "test_main.py") is True

    def test_should_index_file_excluded(
        self, mock_config: EmbeCodeConfig, mock_db: Mock, mock_embedder: Mock, temp_project: Path
    ) -> None:
        """Should not index files matching exclude patterns."""
        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)

        assert indexer._should_index_file(temp_project / "node_modules" / "lib.js") is False

    def test_should_index_file_not_in_includes(
        self, mock_config: EmbeCodeConfig, mock_db: Mock, mock_embedder: Mock, temp_project: Path
    ) -> None:
        """Should not index files not matching any include pattern."""
        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)

        # Create a file not in includes
        (temp_project / "README.md").write_text("# README")

        assert indexer._should_index_file(temp_project / "README.md") is False

    def test_matches_pattern_exact(self) -> None:
        """Should match exact file patterns."""
        assert Indexer._matches_pattern("src/main.py", "src/main.py") is True
        assert Indexer._matches_pattern("src/utils.py", "src/main.py") is False

    def test_matches_pattern_wildcard(self) -> None:
        """Should match wildcard patterns."""
        assert Indexer._matches_pattern("src/main.py", "src/*.py") is True
        assert Indexer._matches_pattern("src/main.js", "src/*.py") is False

    def test_matches_pattern_directory(self) -> None:
        """Should match directory patterns."""
        assert Indexer._matches_pattern("node_modules/lib.js", "node_modules/") is True
        assert Indexer._matches_pattern("src/main.py", "node_modules/") is False

    def test_matches_pattern_recursive(self) -> None:
        """Should match recursive patterns with **."""
        assert Indexer._matches_pattern("src/deep/nested/file.py", "**/*.py") is True
        assert Indexer._matches_pattern("src/file.js", "**/*.py") is False

    def test_collect_files(
        self, mock_config: EmbeCodeConfig, mock_db: Mock, mock_embedder: Mock, temp_project: Path
    ) -> None:
        """Should collect files matching include/exclude rules."""
        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)
        files = indexer._collect_files()

        # Should include src/ and tests/ files, exclude node_modules/
        file_names = {f.name for f in files}
        assert "main.py" in file_names
        assert "utils.py" in file_names
        assert "test_main.py" in file_names
        assert "lib.js" not in file_names

    @patch("embecode.chunker.chunk_file")
    def test_update_file_new_file(
        self,
        mock_chunk_file: Mock,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
        temp_project: Path,
    ) -> None:
        """Should add chunks for a new file."""
        # Setup mock chunk_file
        mock_chunk = Chunk(
            file_path="src/main.py",
            language="python",
            start_line=1,
            end_line=1,
            content="print('hello')",
            context="File: src/main.py",
            hash="abc123",
        )
        mock_chunk_file.return_value = [mock_chunk]

        # No existing chunks
        mock_db.get_chunk_hashes_for_file.return_value = set()

        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)
        indexer.update_file(temp_project / "src" / "main.py")

        # Should embed and insert new chunk
        mock_embedder.embed.assert_called_once()
        mock_db.insert_chunks.assert_called_once()
        mock_db.update_file_metadata.assert_called_once()

    @patch("embecode.chunker.chunk_file")
    def test_update_file_changed_file(
        self,
        mock_chunk_file: Mock,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
        temp_project: Path,
    ) -> None:
        """Should update chunks for a changed file."""
        # Setup mock chunk_file
        new_chunk = Chunk(
            file_path="src/main.py",
            language="python",
            start_line=1,
            end_line=1,
            content="print('hello world')",
            context="File: src/main.py",
            hash="def456",
        )
        mock_chunk_file.return_value = [new_chunk]

        # Existing chunk has different hash
        mock_db.get_chunk_hashes_for_file.return_value = {"abc123"}

        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)
        indexer.update_file(temp_project / "src" / "main.py")

        # Should delete old chunk and insert new chunk
        mock_db.delete_chunks_by_hash.assert_called_once_with(["abc123"])
        mock_embedder.embed.assert_called_once()
        mock_db.insert_chunks.assert_called_once()

    @patch("embecode.chunker.chunk_file")
    def test_update_file_unchanged(
        self,
        mock_chunk_file: Mock,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
        temp_project: Path,
    ) -> None:
        """Should not update chunks if file hasn't changed."""
        # Setup mock chunk_file
        chunk = Chunk(
            file_path="src/main.py",
            language="python",
            start_line=1,
            end_line=1,
            content="print('hello')",
            context="File: src/main.py",
            hash="abc123",
        )
        mock_chunk_file.return_value = [chunk]

        # Existing chunk has same hash
        mock_db.get_chunk_hashes_for_file.return_value = {"abc123"}

        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)
        indexer.update_file(temp_project / "src" / "main.py")

        # Should not delete or insert anything
        mock_db.delete_chunks_by_hash.assert_not_called()
        mock_embedder.embed.assert_not_called()
        mock_db.insert_chunks.assert_not_called()

    def test_delete_file(
        self, mock_config: EmbeCodeConfig, mock_db: Mock, mock_embedder: Mock, temp_project: Path
    ) -> None:
        """Should remove file from index."""
        mock_db.delete_file.return_value = 5

        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)
        indexer.delete_file(temp_project / "src" / "main.py")

        mock_db.delete_file.assert_called_once()

    def test_start_full_index_already_in_progress(
        self, temp_project: Path, mock_config: EmbeCodeConfig, mock_db: Mock, mock_embedder: Mock
    ) -> None:
        """Should raise error if indexing already in progress."""
        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)

        with indexer._lock:
            indexer._is_indexing = True

        with pytest.raises(IndexingInProgressError):
            indexer.start_full_index()

    @patch("embecode.chunker.chunk_file")
    def test_start_full_index_foreground(
        self,
        mock_chunk_file: Mock,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
        temp_project: Path,
    ) -> None:
        """Should run full index in foreground when background=False."""
        # Setup mock chunk_file
        mock_chunk = Chunk(
            file_path="src/main.py",
            language="python",
            start_line=1,
            end_line=1,
            content="print('hello')",
            context="File: src/main.py",
            hash="abc123",
        )
        mock_chunk_file.return_value = [mock_chunk]

        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)
        indexer.start_full_index(background=False)

        # Should have cleared index
        mock_db.clear_index.assert_called_once()

        # Should have processed files
        mock_embedder.embed.assert_called()
        mock_db.insert_chunks.assert_called()

        # Should not be indexing anymore
        assert indexer.is_indexing is False

    @patch("embecode.chunker.chunk_file")
    def test_start_full_index_background(
        self,
        mock_chunk_file: Mock,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
        temp_project: Path,
    ) -> None:
        """Should run full index in background thread when background=True."""
        # Setup mock chunk_file
        mock_chunk = Chunk(
            file_path="src/main.py",
            language="python",
            start_line=1,
            end_line=1,
            content="print('hello')",
            context="File: src/main.py",
            hash="abc123",
        )
        mock_chunk_file.return_value = [mock_chunk]

        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)
        indexer.start_full_index(background=True)

        # Should be indexing
        assert indexer.is_indexing is True

        # Wait for completion
        completed = indexer.wait_for_completion(timeout=5.0)
        assert completed is True

        # Should have processed files
        mock_db.clear_index.assert_called_once()
        mock_embedder.embed.assert_called()
        mock_db.insert_chunks.assert_called()

    def test_wait_for_completion_no_thread(
        self, temp_project: Path, mock_config: EmbeCodeConfig, mock_db: Mock, mock_embedder: Mock
    ) -> None:
        """Should return immediately if no indexing thread."""
        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)
        result = indexer.wait_for_completion(timeout=1.0)
        assert result is True

    def test_store_chunks_mismatched_lengths(
        self, temp_project: Path, mock_config: EmbeCodeConfig, mock_db: Mock, mock_embedder: Mock
    ) -> None:
        """Should raise error if chunk and embedding counts don't match."""
        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)

        chunks = [
            Chunk(
                file_path="test.py",
                language="python",
                start_line=1,
                end_line=1,
                content="test",
                context="File: test.py",
                hash="abc",
            )
        ]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]  # Wrong count

        with pytest.raises(IndexingError, match=r"Chunk count .* does not match"):
            indexer._store_chunks(Path("test.py"), chunks, embeddings)
