"""Tests for indexer.py - indexing orchestration."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
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

    def test_to_dict_includes_new_fields(self) -> None:
        """Should include indexing_type and files_to_process keys, defaulting to None."""
        status = IndexStatus(
            files_indexed=10,
            total_chunks=100,
            embedding_model="test-model",
            last_updated="2025-02-25T10:00:00",
            is_indexing=False,
        )

        result = status.to_dict()

        assert "indexing_type" in result
        assert "files_to_process" in result
        assert result["indexing_type"] is None
        assert result["files_to_process"] is None

        # With values set
        status_active = IndexStatus(
            files_indexed=5,
            total_chunks=50,
            embedding_model="test-model",
            last_updated="2025-02-25T10:00:00",
            is_indexing=True,
            indexing_type="catchup",
            files_to_process=15,
        )

        result_active = status_active.to_dict()

        assert result_active["indexing_type"] == "catchup"
        assert result_active["files_to_process"] == 15


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


class TestCatchUpIndex:
    """Test suite for catch-up indexing."""

    @pytest.fixture
    def temp_project(self) -> Path:
        """Create a temporary project directory with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Create test files
            (project_path / "src").mkdir()
            (project_path / "src" / "main.py").write_text("print('hello')")
            (project_path / "src" / "utils.py").write_text("def foo(): pass")
            (project_path / "src" / "helpers.py").write_text("def bar(): pass")
            (project_path / "tests").mkdir()
            (project_path / "tests" / "test_main.py").write_text("def test_foo(): pass")
            (project_path / "tests" / "test_utils.py").write_text("def test_bar(): pass")

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
        db.get_indexed_files_with_timestamps.return_value = {}
        return db

    @pytest.fixture
    def mock_embedder(self) -> Mock:
        """Create a mock embedder."""
        embedder = Mock()
        embedder.embed.return_value = [[0.1, 0.2, 0.3]]
        return embedder

    @patch("embecode.chunker.chunk_file")
    def test_catchup_empty_db_indexes_all_files(
        self,
        mock_chunk_file: Mock,
        temp_project: Path,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """DB is empty - catch-up discovers all files as missing, indexes them all."""
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

        # Empty DB
        mock_db.get_indexed_files_with_timestamps.return_value = {}

        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)
        indexer.start_catchup_index(background=False)

        # clear_index should NOT be called
        mock_db.clear_index.assert_not_called()

        # update_file_metadata should be called for each file
        assert mock_db.update_file_metadata.call_count == 5  # 3 src + 2 tests

        # Should not be indexing anymore
        assert indexer.is_indexing is False

    @patch("embecode.chunker.chunk_file")
    def test_catchup_partial_db_indexes_missing_files(
        self,
        mock_chunk_file: Mock,
        temp_project: Path,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """DB has 2 of 5 files indexed - catch-up indexes the 3 missing files."""
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

        # DB has 2 files already indexed (with future timestamp so they aren't "modified")
        future_ts = datetime(2099, 1, 1, tzinfo=UTC)
        mock_db.get_indexed_files_with_timestamps.return_value = {
            str(temp_project / "src" / "main.py"): future_ts,
            str(temp_project / "src" / "utils.py"): future_ts,
        }

        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)
        indexer.start_catchup_index(background=False)

        # Should index only the 3 missing files
        assert mock_db.update_file_metadata.call_count == 3

    @patch("embecode.chunker.chunk_file")
    def test_catchup_removes_stale_files(
        self,
        mock_chunk_file: Mock,
        temp_project: Path,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """DB has a file that no longer exists on disk - catch-up calls delete_file."""
        mock_chunk_file.return_value = []

        # DB has a stale file that doesn't exist on disk
        future_ts = datetime(2099, 1, 1, tzinfo=UTC)
        stale_path = str(temp_project / "src" / "deleted_file.py")
        indexed = {stale_path: future_ts}
        # Also add all real files so they don't appear as missing
        for f in [
            "src/main.py",
            "src/utils.py",
            "src/helpers.py",
            "tests/test_main.py",
            "tests/test_utils.py",
        ]:
            indexed[str(temp_project / f)] = future_ts
        mock_db.get_indexed_files_with_timestamps.return_value = indexed

        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)
        indexer.start_catchup_index(background=False)

        # delete_file should be called for the stale path
        mock_db.delete_file.assert_called_once_with(stale_path)

    @patch("embecode.chunker.chunk_file")
    def test_catchup_reindexes_modified_files(
        self,
        mock_chunk_file: Mock,
        temp_project: Path,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """DB has a file with last_indexed older than mtime - catch-up re-indexes it."""
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

        # Set all files as indexed, but one with old timestamp
        future_ts = datetime(2099, 1, 1, tzinfo=UTC)
        old_ts = datetime(2000, 1, 1, tzinfo=UTC)
        indexed = {}
        for f in [
            "src/main.py",
            "src/utils.py",
            "src/helpers.py",
            "tests/test_main.py",
            "tests/test_utils.py",
        ]:
            full_path = str(temp_project / f)
            indexed[full_path] = future_ts

        # Make main.py appear modified (old last_indexed)
        main_path = str(temp_project / "src" / "main.py")
        indexed[main_path] = old_ts

        mock_db.get_indexed_files_with_timestamps.return_value = indexed

        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)
        indexer.start_catchup_index(background=False)

        # update_file_metadata should be called for the modified file
        assert mock_db.update_file_metadata.call_count == 1
        mock_db.update_file_metadata.assert_called_once_with(main_path, 1)

    @patch("embecode.chunker.chunk_file")
    def test_catchup_skips_unmodified_files(
        self,
        mock_chunk_file: Mock,
        temp_project: Path,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """DB has a file with last_indexed newer than mtime - not re-processed."""
        mock_chunk_file.return_value = []

        # All files indexed with future timestamp (not modified)
        future_ts = datetime(2099, 1, 1, tzinfo=UTC)
        indexed = {}
        for f in [
            "src/main.py",
            "src/utils.py",
            "src/helpers.py",
            "tests/test_main.py",
            "tests/test_utils.py",
        ]:
            indexed[str(temp_project / f)] = future_ts

        mock_db.get_indexed_files_with_timestamps.return_value = indexed

        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)
        indexer.start_catchup_index(background=False)

        # No files should be processed
        mock_db.update_file_metadata.assert_not_called()
        mock_embedder.embed.assert_not_called()

    @patch("embecode.chunker.chunk_file")
    def test_catchup_complete_index_noop(
        self,
        mock_chunk_file: Mock,
        temp_project: Path,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """All files on disk are in DB and unmodified - _is_indexing is never set True."""
        mock_chunk_file.return_value = []

        future_ts = datetime(2099, 1, 1, tzinfo=UTC)
        indexed = {}
        for f in [
            "src/main.py",
            "src/utils.py",
            "src/helpers.py",
            "tests/test_main.py",
            "tests/test_utils.py",
        ]:
            indexed[str(temp_project / f)] = future_ts

        mock_db.get_indexed_files_with_timestamps.return_value = indexed

        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)

        # Track if _is_indexing was ever set to True
        was_indexing = []
        original_run = indexer._run_catchup_index

        def tracking_run() -> None:
            was_indexing.append(indexer._is_indexing)
            original_run()
            was_indexing.append(indexer._is_indexing)

        indexer._run_catchup_index = tracking_run
        indexer.start_catchup_index(background=False)

        # _is_indexing should never have been True
        assert all(v is False for v in was_indexing)

    @patch("embecode.chunker.chunk_file")
    def test_catchup_sets_is_indexing_when_work_exists(
        self,
        mock_chunk_file: Mock,
        temp_project: Path,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """DB is missing files - during catch-up, _is_indexing is True."""
        was_indexing_during = []

        def tracking_embed(texts: list[str]) -> list[list[float]]:
            was_indexing_during.append(indexer._is_indexing)
            return [[0.1, 0.2, 0.3]] * len(texts)

        mock_embedder.embed.side_effect = tracking_embed

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
        mock_db.get_indexed_files_with_timestamps.return_value = {}

        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)
        indexer.start_catchup_index(background=False)

        # During indexing, _is_indexing should have been True
        assert len(was_indexing_during) > 0
        assert all(v is True for v in was_indexing_during)

        # After completion, should be False
        assert indexer.is_indexing is False

    @patch("embecode.chunker.chunk_file")
    def test_catchup_does_not_set_is_indexing_when_no_work(
        self,
        mock_chunk_file: Mock,
        temp_project: Path,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """Index is complete - _is_indexing remains False throughout."""
        future_ts = datetime(2099, 1, 1, tzinfo=UTC)
        indexed = {}
        for f in [
            "src/main.py",
            "src/utils.py",
            "src/helpers.py",
            "tests/test_main.py",
            "tests/test_utils.py",
        ]:
            indexed[str(temp_project / f)] = future_ts

        mock_db.get_indexed_files_with_timestamps.return_value = indexed

        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)
        indexer.start_catchup_index(background=False)

        assert indexer.is_indexing is False

    @patch("embecode.chunker.chunk_file")
    def test_catchup_progress_tracking(
        self,
        mock_chunk_file: Mock,
        temp_project: Path,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """With 3 missing files, _progress updates from 0 to 1 as files are processed."""
        progress_values = []

        def tracking_embed(texts: list[str]) -> list[list[float]]:
            with indexer._lock:
                progress_values.append(indexer._progress)
            return [[0.1, 0.2, 0.3]] * len(texts)

        mock_embedder.embed.side_effect = tracking_embed

        mock_chunk = Chunk(
            file_path="test.py",
            language="python",
            start_line=1,
            end_line=1,
            content="test",
            context="File: test.py",
            hash="abc123",
        )
        mock_chunk_file.return_value = [mock_chunk]

        # Only include 3 files as missing (by having an empty DB)
        # But we need exactly 3 files. Use a smaller project.
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            (project_path / "src").mkdir()
            (project_path / "src" / "a.py").write_text("a")
            (project_path / "src" / "b.py").write_text("b")
            (project_path / "src" / "c.py").write_text("c")

            mock_db.get_indexed_files_with_timestamps.return_value = {}

            indexer = Indexer(project_path, mock_config, mock_db, mock_embedder)
            indexer.start_catchup_index(background=False)

        # Progress should have been tracked
        assert len(progress_values) == 3
        # Progress increases: 0/3, 1/3, 2/3
        assert progress_values[0] == pytest.approx(0.0)
        assert progress_values[1] == pytest.approx(1 / 3)
        assert progress_values[2] == pytest.approx(2 / 3)

        # After completion, progress should be cleared
        assert indexer._progress is None

    def test_catchup_already_in_progress_raises(
        self,
        temp_project: Path,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """_is_indexing is already True - start_catchup_index raises IndexingInProgressError."""
        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)

        with indexer._lock:
            indexer._is_indexing = True

        with pytest.raises(IndexingInProgressError):
            indexer.start_catchup_index()

    @patch("embecode.chunker.chunk_file")
    def test_catchup_background_thread(
        self,
        mock_chunk_file: Mock,
        temp_project: Path,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """start_catchup_index(background=True) runs in background thread."""
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
        mock_db.get_indexed_files_with_timestamps.return_value = {}

        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)
        indexer.start_catchup_index(background=True)

        completed = indexer.wait_for_completion(timeout=5.0)
        assert completed is True
        assert indexer.is_indexing is False

    @patch("embecode.chunker.chunk_file")
    def test_catchup_individual_file_failure_continues(
        self,
        mock_chunk_file: Mock,
        temp_project: Path,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """One file fails to index - catch-up continues with remaining files."""
        call_count = 0

        def failing_chunk_file(path: Path, languages: object) -> list[Chunk]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Parse error")
            return [
                Chunk(
                    file_path=str(path),
                    language="python",
                    start_line=1,
                    end_line=1,
                    content="test",
                    context="File: test",
                    hash=f"hash_{call_count}",
                )
            ]

        mock_chunk_file.side_effect = failing_chunk_file
        mock_db.get_indexed_files_with_timestamps.return_value = {}

        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)
        indexer.start_catchup_index(background=False)

        # Should have processed files despite one failure
        # 5 files total, 1 fails, 4 succeed
        assert mock_db.update_file_metadata.call_count == 4
        assert indexer.is_indexing is False

    @patch("embecode.chunker.chunk_file")
    def test_catchup_status_shows_indexing_type(
        self,
        mock_chunk_file: Mock,
        temp_project: Path,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """During catch-up with work, get_status() returns indexing_type='catchup'."""
        status_during = []

        def tracking_embed(texts: list[str]) -> list[list[float]]:
            status_during.append(indexer.get_status())
            return [[0.1, 0.2, 0.3]] * len(texts)

        mock_embedder.embed.side_effect = tracking_embed

        mock_chunk = Chunk(
            file_path="test.py",
            language="python",
            start_line=1,
            end_line=1,
            content="test",
            context="File: test",
            hash="abc123",
        )
        mock_chunk_file.return_value = [mock_chunk]
        mock_db.get_indexed_files_with_timestamps.return_value = {}

        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)
        indexer.start_catchup_index(background=False)

        assert len(status_during) > 0
        assert status_during[0].indexing_type == "catchup"
        assert status_during[0].files_to_process == 5  # All 5 files are missing

    @patch("embecode.chunker.chunk_file")
    def test_catchup_noop_status_shows_no_indexing_type(
        self,
        mock_chunk_file: Mock,
        temp_project: Path,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """When catch-up finds nothing to do, indexing_type is None."""
        future_ts = datetime(2099, 1, 1, tzinfo=UTC)
        indexed = {}
        for f in [
            "src/main.py",
            "src/utils.py",
            "src/helpers.py",
            "tests/test_main.py",
            "tests/test_utils.py",
        ]:
            indexed[str(temp_project / f)] = future_ts

        mock_db.get_indexed_files_with_timestamps.return_value = indexed

        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)
        indexer.start_catchup_index(background=False)

        status = indexer.get_status()
        assert status.indexing_type is None
        assert status.files_to_process is None

    @patch("embecode.chunker.chunk_file")
    def test_full_index_status_shows_indexing_type(
        self,
        mock_chunk_file: Mock,
        temp_project: Path,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """During start_full_index(), get_status() returns indexing_type='full'."""
        status_during = []

        def tracking_embed(texts: list[str]) -> list[list[float]]:
            status_during.append(indexer.get_status())
            return [[0.1, 0.2, 0.3]] * len(texts)

        mock_embedder.embed.side_effect = tracking_embed

        mock_chunk = Chunk(
            file_path="test.py",
            language="python",
            start_line=1,
            end_line=1,
            content="test",
            context="File: test",
            hash="abc123",
        )
        mock_chunk_file.return_value = [mock_chunk]

        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)
        indexer.start_full_index(background=False)

        assert len(status_during) > 0
        assert status_during[0].indexing_type == "full"
        assert status_during[0].files_to_process == 5  # Total files

    @patch("embecode.chunker.chunk_file")
    def test_status_cleared_after_indexing(
        self,
        mock_chunk_file: Mock,
        temp_project: Path,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """After both full and catch-up indexing complete, indexing_type and files_to_process are None."""
        mock_chunk = Chunk(
            file_path="test.py",
            language="python",
            start_line=1,
            end_line=1,
            content="test",
            context="File: test",
            hash="abc123",
        )
        mock_chunk_file.return_value = [mock_chunk]
        mock_db.get_indexed_files_with_timestamps.return_value = {}

        indexer = Indexer(temp_project, mock_config, mock_db, mock_embedder)

        # After full index
        indexer.start_full_index(background=False)
        status = indexer.get_status()
        assert status.indexing_type is None
        assert status.files_to_process is None

        # After catch-up index
        mock_db.get_indexed_files_with_timestamps.return_value = {}
        indexer.start_catchup_index(background=False)
        status = indexer.get_status()
        assert status.indexing_type is None
        assert status.files_to_process is None
