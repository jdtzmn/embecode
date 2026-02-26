"""Tests for .gitignore support in indexer.py."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from embecode.config import (
    EmbeCodeConfig,
    EmbeddingsConfig,
    IndexConfig,
    LanguageConfig,
)
from embecode.indexer import Indexer


class TestGitignoreBasic:
    """Test suite for basic .gitignore functionality."""

    @pytest.fixture
    def mock_config(self) -> EmbeCodeConfig:
        """Create a mock config with empty include (index everything)."""
        config = Mock(spec=EmbeCodeConfig)
        config.index = IndexConfig(
            include=[],  # Empty = index everything
            exclude=["node_modules/", ".git/"],
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
        return db

    @pytest.fixture
    def mock_embedder(self) -> Mock:
        """Create a mock embedder."""
        return Mock()

    def test_no_gitignore_indexes_all_files(
        self,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """Project with no .gitignore should index all files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Create test files without .gitignore
            (project_path / "main.py").write_text("print('hello')")
            (project_path / "utils.py").write_text("def foo(): pass")
            (project_path / "data.json").write_text('{"key": "value"}')

            # Create subdirectory with files
            subdir = project_path / "src"
            subdir.mkdir()
            (subdir / "module.py").write_text("class MyClass: pass")
            (subdir / "helper.py").write_text("def helper(): return 42")

            # Create indexer and collect files
            indexer = Indexer(project_path, mock_config, mock_db, mock_embedder)
            files = indexer._collect_files()

            # Convert to relative paths for easier assertion
            relative_files = sorted([f.relative_to(project_path) for f in files])

            # All files should be collected
            expected = [
                Path("data.json"),
                Path("main.py"),
                Path("src/helper.py"),
                Path("src/module.py"),
                Path("utils.py"),
            ]

            assert relative_files == expected

    def test_empty_gitignore_has_no_effect(
        self,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """Completely empty .gitignore file should have no effect on indexing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Create empty .gitignore file
            (project_path / ".gitignore").write_text("")

            # Create test files
            (project_path / "main.py").write_text("print('hello')")
            (project_path / "utils.py").write_text("def foo(): pass")
            (project_path / "data.json").write_text('{"key": "value"}')

            # Create subdirectory with files
            subdir = project_path / "src"
            subdir.mkdir()
            (subdir / "module.py").write_text("class MyClass: pass")
            (subdir / "helper.py").write_text("def helper(): return 42")

            # Create indexer and collect files
            indexer = Indexer(project_path, mock_config, mock_db, mock_embedder)
            files = indexer._collect_files()

            # Convert to relative paths for easier assertion
            relative_files = sorted([f.relative_to(project_path) for f in files])

            # All files should be collected (empty .gitignore has no effect)
            expected = [
                Path("data.json"),
                Path("main.py"),
                Path("src/helper.py"),
                Path("src/module.py"),
                Path("utils.py"),
            ]

            assert relative_files == expected

    def test_no_git_directory_still_respects_gitignore(
        self,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """Project with .gitignore but no .git directory should still respect gitignore rules."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Create .gitignore without creating .git directory
            (project_path / ".gitignore").write_text("*.log\n*.tmp\n")

            # Create test files - some that should be ignored, some that should not
            (project_path / "main.py").write_text("print('hello')")
            (project_path / "debug.log").write_text("log content")  # Should be ignored
            (project_path / "temp.tmp").write_text("temp data")  # Should be ignored
            (project_path / "data.json").write_text('{"key": "value"}')

            # Create subdirectory with files
            subdir = project_path / "src"
            subdir.mkdir()
            (subdir / "module.py").write_text("class MyClass: pass")
            (subdir / "output.log").write_text("output")  # Should be ignored

            # Create indexer and collect files
            indexer = Indexer(project_path, mock_config, mock_db, mock_embedder)
            files = indexer._collect_files()

            # Convert to relative paths for easier assertion
            relative_files = sorted([f.relative_to(project_path) for f in files])

            # Only non-ignored files should be collected
            expected = [
                Path("data.json"),
                Path("main.py"),
                Path("src/module.py"),
            ]

            assert relative_files == expected
