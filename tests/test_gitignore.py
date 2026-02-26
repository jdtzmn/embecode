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


class TestGitignorePatternAnchoring:
    """Test suite for gitignore pattern anchoring behavior."""

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

    def test_unanchored_pattern_matches_at_any_depth(
        self,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """Pattern with no slash (e.g., *.log) should match at any depth."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Create .gitignore with unanchored pattern
            (project_path / ".gitignore").write_text("*.log\n")

            # Create test files at various depths
            (project_path / "root.py").write_text("print('root')")
            (project_path / "root.log").write_text("root log")  # Should be ignored

            # Create nested directory structure
            a = project_path / "a"
            a.mkdir()
            (a / "file.py").write_text("print('a')")
            (a / "file.log").write_text("a log")  # Should be ignored

            b = a / "b"
            b.mkdir()
            (b / "file.py").write_text("print('b')")
            (b / "file.log").write_text("b log")  # Should be ignored

            c = b / "c"
            c.mkdir()
            (c / "file.py").write_text("print('c')")
            (c / "file.log").write_text("c log")  # Should be ignored

            # Create indexer and collect files
            indexer = Indexer(project_path, mock_config, mock_db, mock_embedder)
            files = indexer._collect_files()

            # Convert to relative paths for easier assertion
            relative_files = sorted([f.relative_to(project_path) for f in files])

            # Only .py files should be collected, all .log files should be ignored
            expected = [
                Path("a/b/c/file.py"),
                Path("a/b/file.py"),
                Path("a/file.py"),
                Path("root.py"),
            ]

            assert relative_files == expected

    def test_leading_slash_anchors_to_gitignore_dir(
        self,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """Pattern with leading slash (e.g., /foo.txt) should only match at gitignore directory level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Create .gitignore with leading slash pattern
            (project_path / ".gitignore").write_text("/foo.txt\n")

            # Create foo.txt at root (should be ignored)
            (project_path / "foo.txt").write_text("root foo")
            (project_path / "bar.txt").write_text("root bar")

            # Create subdirectory with another foo.txt (should NOT be ignored)
            subdir = project_path / "sub"
            subdir.mkdir()
            (subdir / "foo.txt").write_text("sub foo")
            (subdir / "baz.txt").write_text("sub baz")

            # Create deeper nested directory with foo.txt (should NOT be ignored)
            nested = subdir / "nested"
            nested.mkdir()
            (nested / "foo.txt").write_text("nested foo")
            (nested / "other.txt").write_text("nested other")

            # Create indexer and collect files
            indexer = Indexer(project_path, mock_config, mock_db, mock_embedder)
            files = indexer._collect_files()

            # Convert to relative paths for easier assertion
            relative_files = sorted([f.relative_to(project_path) for f in files])

            # Root foo.txt should be excluded, but foo.txt in subdirectories should be included
            expected = [
                Path("bar.txt"),
                Path("sub/baz.txt"),
                Path("sub/foo.txt"),  # NOT ignored (pattern is anchored to root)
                Path("sub/nested/foo.txt"),  # NOT ignored
                Path("sub/nested/other.txt"),
            ]

            assert relative_files == expected

    def test_middle_slash_anchors_to_gitignore_dir(
        self,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """Pattern with slash in middle (e.g., foo/bar.txt) should be anchored to gitignore directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Create .gitignore with middle-slash pattern
            (project_path / ".gitignore").write_text("foo/bar.txt\n")

            # Create foo directory at root with bar.txt (should be ignored)
            foo_dir = project_path / "foo"
            foo_dir.mkdir()
            (foo_dir / "bar.txt").write_text("should be ignored")
            (foo_dir / "other.txt").write_text("not ignored")

            # Create subdirectory with its own foo/bar.txt (should NOT be ignored - pattern is anchored to root)
            subdir = project_path / "sub"
            subdir.mkdir()
            sub_foo_dir = subdir / "foo"
            sub_foo_dir.mkdir()
            (sub_foo_dir / "bar.txt").write_text("should not be ignored")
            (sub_foo_dir / "other.txt").write_text("also not ignored")

            # Create some other files
            (project_path / "main.py").write_text("print('hello')")
            (subdir / "module.py").write_text("class MyClass: pass")

            # Create indexer and collect files
            indexer = Indexer(project_path, mock_config, mock_db, mock_embedder)
            files = indexer._collect_files()

            # Convert to relative paths for easier assertion
            relative_files = sorted([f.relative_to(project_path) for f in files])

            # Only root-level foo/bar.txt should be excluded
            expected = [
                Path("foo/other.txt"),
                Path("main.py"),
                Path("sub/foo/bar.txt"),  # NOT ignored (pattern is anchored to root)
                Path("sub/foo/other.txt"),
                Path("sub/module.py"),
            ]

            assert relative_files == expected

    def test_trailing_slash_matches_directories_only(
        self,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """Pattern with trailing slash (e.g., build/) should match directories only, not files with same name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Create .gitignore with trailing slash pattern
            (project_path / ".gitignore").write_text("dist/\n")

            # Create a directory named 'dist' with files inside (should all be ignored)
            dist_dir = project_path / "dist"
            dist_dir.mkdir()
            (dist_dir / "output.js").write_text("console.log('build');")
            (dist_dir / "bundle.js").write_text("console.log('bundle');")

            # Create a nested file inside dist directory (should also be ignored)
            nested_dist = dist_dir / "nested"
            nested_dist.mkdir()
            (nested_dist / "deep.js").write_text("console.log('deep');")

            # Create a FILE literally named 'dist' in a different location (should NOT be ignored - pattern only matches directories)
            # Put it in a subdirectory to avoid name collision
            src_dir = project_path / "src"
            src_dir.mkdir()
            (src_dir / "dist").write_text("this is a file named dist")
            (src_dir / "module.py").write_text("class MyClass: pass")

            # Create some other files that should be indexed
            (project_path / "main.py").write_text("print('hello')")

            # Create indexer and collect files
            indexer = Indexer(project_path, mock_config, mock_db, mock_embedder)
            files = indexer._collect_files()

            # Convert to relative paths for easier assertion
            relative_files = sorted([f.relative_to(project_path) for f in files])

            # The file named 'dist' should be included, but everything in dist/ directory should be excluded
            expected = [
                Path("main.py"),
                Path("src/dist"),  # File, not directory - should be included
                Path("src/module.py"),
            ]

            assert relative_files == expected

    def test_double_star_prefix_matches_any_depth(
        self,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """Pattern with **/ prefix (e.g., **/logs) should match at any depth."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Create .gitignore with **/ prefix pattern
            (project_path / ".gitignore").write_text("**/logs\n")

            # Create 'logs' directory at root level (should be ignored)
            root_logs = project_path / "logs"
            root_logs.mkdir()
            (root_logs / "app.log").write_text("root level log")
            (root_logs / "error.log").write_text("root level error")

            # Create 'logs' directory in a subdirectory (should also be ignored)
            sub = project_path / "sub"
            sub.mkdir()
            sub_logs = sub / "logs"
            sub_logs.mkdir()
            (sub_logs / "debug.log").write_text("sub level log")

            # Create 'logs' directory deeply nested (should also be ignored)
            a = project_path / "a"
            a.mkdir()
            b = a / "b"
            b.mkdir()
            ab_logs = b / "logs"
            ab_logs.mkdir()
            (ab_logs / "trace.log").write_text("deeply nested log")

            # Create some non-logs files at various levels (should be indexed)
            (project_path / "main.py").write_text("print('root')")
            (sub / "module.py").write_text("print('sub')")
            (b / "helper.py").write_text("print('nested')")

            # Create indexer and collect files
            indexer = Indexer(project_path, mock_config, mock_db, mock_embedder)
            files = indexer._collect_files()

            # Convert to relative paths for easier assertion
            relative_files = sorted([f.relative_to(project_path) for f in files])

            # All files in 'logs' directories at any depth should be excluded
            expected = [
                Path("a/b/helper.py"),
                Path("main.py"),
                Path("sub/module.py"),
            ]

            assert relative_files == expected
