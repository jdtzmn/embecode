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

    def test_double_star_suffix_matches_subtree(
        self,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """Pattern with /** suffix (e.g., abc/**) should exclude all files inside abc/ at any depth."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Create .gitignore with /** suffix pattern
            (project_path / ".gitignore").write_text("abc/**\n")

            # Create 'abc' directory with files at various depths (all should be ignored)
            abc = project_path / "abc"
            abc.mkdir()
            (abc / "file1.txt").write_text("file at abc root")
            (abc / "file2.py").write_text("python file at abc root")

            # Create subdirectory inside abc (should also be ignored)
            abc_sub = abc / "sub"
            abc_sub.mkdir()
            (abc_sub / "deep1.txt").write_text("file in abc/sub")

            # Create deeply nested directory inside abc (should also be ignored)
            abc_nested = abc_sub / "nested"
            abc_nested.mkdir()
            (abc_nested / "deep2.js").write_text("file in abc/sub/nested")

            # Create files outside abc directory (should be indexed)
            (project_path / "main.py").write_text("print('root')")
            (project_path / "readme.md").write_text("# README")

            # Create another directory with files (should be indexed)
            other = project_path / "other"
            other.mkdir()
            (other / "module.py").write_text("print('other')")
            (other / "data.json").write_text('{"key": "value"}')

            # Create indexer and collect files
            indexer = Indexer(project_path, mock_config, mock_db, mock_embedder)
            files = indexer._collect_files()

            # Convert to relative paths for easier assertion
            relative_files = sorted([f.relative_to(project_path) for f in files])

            # All files inside 'abc/' at any depth should be excluded
            expected = [
                Path("main.py"),
                Path("other/data.json"),
                Path("other/module.py"),
                Path("readme.md"),
            ]

            assert relative_files == expected

    def test_double_star_middle_matches_zero_or_more_dirs(
        self,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """Pattern with /** / in middle (e.g., a/**/b) should match zero or more directory levels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Create .gitignore with /**/ middle pattern
            (project_path / ".gitignore").write_text("a/**/b\n")

            # Create files that match at various depths
            # Zero directories between: a/b (should be ignored)
            a_dir = project_path / "a"
            a_dir.mkdir()
            (a_dir / "b").write_text("zero levels")  # Should be ignored
            (a_dir / "other.txt").write_text("not matched")  # Should NOT be ignored

            # One directory between: a/x/b (should be ignored)
            a_x = a_dir / "x"
            a_x.mkdir()
            (a_x / "b").write_text("one level")  # Should be ignored
            (a_x / "other.txt").write_text("not matched")  # Should NOT be ignored

            # Two directories between: a/x/y/b (should be ignored)
            a_x_y = a_x / "y"
            a_x_y.mkdir()
            (a_x_y / "b").write_text("two levels")  # Should be ignored
            (a_x_y / "other.txt").write_text("not matched")  # Should NOT be ignored

            # Create files that should NOT match the pattern
            (project_path / "b").write_text("root b")  # Should NOT be ignored (no 'a/' prefix)
            (project_path / "main.py").write_text("print('main')")  # Should NOT be ignored

            # Create another directory that doesn't match
            other = project_path / "other"
            other.mkdir()
            (other / "b").write_text("other/b")  # Should NOT be ignored (no 'a/' prefix)

            # Create indexer and collect files
            indexer = Indexer(project_path, mock_config, mock_db, mock_embedder)
            files = indexer._collect_files()

            # Convert to relative paths for easier assertion
            relative_files = sorted([f.relative_to(project_path) for f in files])

            # Pattern a/**/b should match a/b, a/x/b, a/x/y/b but NOT b or other/b
            expected = [
                Path("a/other.txt"),  # NOT ignored
                Path("a/x/other.txt"),  # NOT ignored
                Path("a/x/y/other.txt"),  # NOT ignored
                Path("b"),  # NOT ignored (no 'a/' prefix)
                Path("main.py"),  # NOT ignored
                Path("other/b"),  # NOT ignored (no 'a/' prefix)
            ]

            assert relative_files == expected

    def test_single_star_does_not_cross_slash(
        self,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """Pattern with single * (e.g., foo/*) matches direct children including subdirectories.

        The * wildcard doesn't match / within a filename, but foo/* matches foo/sub/ (a subdirectory).
        Once foo/sub/ is matched, everything under it is also excluded.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Create .gitignore with single * pattern
            # This matches all direct children of foo/, including files and subdirectories
            (project_path / ".gitignore").write_text("foo/*\n")

            # Create foo/ directory with files directly inside (should be ignored)
            foo_dir = project_path / "foo"
            foo_dir.mkdir()
            (foo_dir / "bar.txt").write_text("direct child")  # Should be ignored
            (foo_dir / "baz.py").write_text("direct child")  # Should be ignored

            # Create foo/ subdirectory - foo/* matches foo/sub/, so everything under it is ignored
            foo_sub = foo_dir / "sub"
            foo_sub.mkdir()
            (foo_sub / "deep.txt").write_text(
                "nested file"
            )  # Should be ignored (parent dir matched)
            (foo_sub / "nested.py").write_text(
                "nested file"
            )  # Should be ignored (parent dir matched)

            # Create another level deeper (also ignored because foo/sub/ is matched)
            foo_sub_deep = foo_sub / "deeper"
            foo_sub_deep.mkdir()
            (foo_sub_deep / "very_deep.txt").write_text("very nested")  # Should be ignored

            # Create some other files at root (should be indexed)
            (project_path / "main.py").write_text("print('main')")
            (project_path / "readme.md").write_text("# Readme")

            # Create indexer and collect files
            indexer = Indexer(project_path, mock_config, mock_db, mock_embedder)
            files = indexer._collect_files()

            # Convert to relative paths for easier assertion
            relative_files = sorted([f.relative_to(project_path) for f in files])

            # Pattern foo/* matches all direct children of foo/, including the sub/ directory
            # Once sub/ is matched, git doesn't recurse into it, so all files under foo/ are excluded
            expected = [
                Path("main.py"),  # NOT ignored
                Path("readme.md"),  # NOT ignored
            ]

            assert relative_files == expected

    def test_question_mark_matches_single_non_slash_char(
        self,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """Pattern with ? (e.g., foo?.txt) should match exactly one character except /."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Create .gitignore with ? pattern
            (project_path / ".gitignore").write_text("foo?.txt\n")

            # Create files that should match (exactly one char after 'foo')
            (project_path / "fooa.txt").write_text("matches - one char")  # Should be ignored
            (project_path / "food.txt").write_text("matches - one char")  # Should be ignored
            (project_path / "foo1.txt").write_text("matches - one char")  # Should be ignored

            # Create files that should NOT match
            (project_path / "foo.txt").write_text("no char after foo")  # NOT ignored (no char)
            (project_path / "foooo.txt").write_text("too many chars")  # NOT ignored (three chars)
            (project_path / "foobar.txt").write_text("too many chars")  # NOT ignored (three chars)

            # Create a file with / in the position (should NOT match - ? doesn't match /)
            subdir = project_path / "foo"
            subdir.mkdir()
            (subdir / ".txt").write_text("slash in position")  # NOT ignored (? doesn't match /)

            # Create other normal files
            (project_path / "main.py").write_text("print('hello')")
            (project_path / "readme.md").write_text("# Readme")

            # Create indexer and collect files
            indexer = Indexer(project_path, mock_config, mock_db, mock_embedder)
            files = indexer._collect_files()

            # Convert to relative paths for easier assertion
            relative_files = sorted([f.relative_to(project_path) for f in files])

            # Only foo?.txt pattern (exactly one char) should be excluded
            expected = [
                Path("foo/.txt"),  # NOT ignored (? doesn't match /)
                Path("foo.txt"),  # NOT ignored (no char after foo)
                Path("foobar.txt"),  # NOT ignored (three chars)
                Path("foooo.txt"),  # NOT ignored (three chars)
                Path("main.py"),  # NOT ignored
                Path("readme.md"),  # NOT ignored
            ]

            assert relative_files == expected

    def test_character_range_pattern(
        self,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """Pattern with character range (e.g., [abc].py) should match files with one character in the range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Create .gitignore with character range pattern
            (project_path / ".gitignore").write_text("[abc].py\n")

            # Create files that should match (single char from range)
            (project_path / "a.py").write_text("# file a")  # Should be ignored
            (project_path / "b.py").write_text("# file b")  # Should be ignored
            (project_path / "c.py").write_text("# file c")  # Should be ignored

            # Create files that should NOT match
            (project_path / "d.py").write_text("# file d")  # NOT ignored (not in range)
            (project_path / "x.py").write_text("# file x")  # NOT ignored (not in range)
            (project_path / "main.py").write_text("# main")  # NOT ignored (multiple chars)
            (project_path / "ab.py").write_text("# ab")  # NOT ignored (two chars)

            # Create other normal files
            (project_path / "readme.md").write_text("# Readme")
            (project_path / "config.json").write_text("{}")

            # Create indexer and collect files
            indexer = Indexer(project_path, mock_config, mock_db, mock_embedder)
            files = indexer._collect_files()

            # Convert to relative paths for easier assertion
            relative_files = sorted([f.relative_to(project_path) for f in files])

            # Only [abc].py pattern (single char from range) should be excluded
            expected = [
                Path("ab.py"),  # NOT ignored (two chars)
                Path("config.json"),  # NOT ignored
                Path("d.py"),  # NOT ignored (not in range)
                Path("main.py"),  # NOT ignored (multiple chars)
                Path("readme.md"),  # NOT ignored
                Path("x.py"),  # NOT ignored (not in range)
            ]

            assert relative_files == expected


class TestGitignoreNegation:
    """Test suite for .gitignore negation patterns (!)."""

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

    def test_negation_re_includes_within_same_file(
        self,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """Negation pattern (!) should re-include previously excluded files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Create .gitignore with pattern then negation
            # Pattern: *.log (exclude all .log files)
            # Negation: !important.log (re-include important.log)
            gitignore = project_path / ".gitignore"
            gitignore.write_text("*.log\n!important.log\n")

            # Create test files
            (project_path / "important.log").write_text("critical data")
            (project_path / "other.log").write_text("regular log")
            (project_path / "debug.log").write_text("debug info")
            (project_path / "main.py").write_text("print('hello')")

            # Create indexer and collect files
            indexer = Indexer(project_path, mock_config, mock_db, mock_embedder)
            files = indexer._collect_files()

            # Convert to relative paths for easier assertion
            relative_files = sorted([f.relative_to(project_path) for f in files])

            # important.log should be re-included by negation
            # other.log and debug.log should still be excluded
            expected = [
                Path("important.log"),  # Re-included by !important.log
                Path("main.py"),  # NOT ignored
            ]

            assert relative_files == expected

    def test_negation_order_matters(
        self,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """Last matching pattern wins: negation then broader pattern should exclude all."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Create .gitignore with NEGATION FIRST, then broader pattern
            # Pattern: !important.log (re-include important.log - but nothing was excluded yet)
            # Pattern: *.log (exclude all .log files - this comes LAST so it wins)
            gitignore = project_path / ".gitignore"
            gitignore.write_text("!important.log\n*.log\n")

            # Create test files
            (project_path / "important.log").write_text("critical data")
            (project_path / "other.log").write_text("regular log")
            (project_path / "debug.log").write_text("debug info")
            (project_path / "main.py").write_text("print('hello')")

            # Create indexer and collect files
            indexer = Indexer(project_path, mock_config, mock_db, mock_embedder)
            files = indexer._collect_files()

            # Convert to relative paths for easier assertion
            relative_files = sorted([f.relative_to(project_path) for f in files])

            # Since *.log comes AFTER !important.log, the last matching rule wins
            # All .log files (including important.log) should be excluded
            expected = [
                Path("main.py"),  # Only non-.log file
            ]

            assert relative_files == expected

    def test_negation_in_child_overrides_parent_ignore(
        self,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """Child .gitignore can use negation to re-include files excluded by parent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Create root .gitignore that excludes all .log files
            root_gitignore = project_path / ".gitignore"
            root_gitignore.write_text("*.log\n")

            # Create subdirectory with its own .gitignore that re-includes keep.log
            sub = project_path / "sub"
            sub.mkdir()
            sub_gitignore = sub / ".gitignore"
            sub_gitignore.write_text("!keep.log\n")

            # Create test files in root directory
            (project_path / "root.py").write_text("print('root')")
            (project_path / "other.log").write_text(
                "root log"
            )  # Should be excluded by root .gitignore

            # Create test files in subdirectory
            (sub / "keep.log").write_text(
                "important log"
            )  # Should be re-included by sub/.gitignore
            (sub / "other.log").write_text("regular log")  # Should be excluded by root .gitignore
            (sub / "module.py").write_text("print('sub')")

            # Create indexer and collect files
            indexer = Indexer(project_path, mock_config, mock_db, mock_embedder)
            files = indexer._collect_files()

            # Convert to relative paths for easier assertion
            relative_files = sorted([f.relative_to(project_path) for f in files])

            # sub/keep.log should be re-included by sub/.gitignore negation
            # root/other.log and sub/other.log should be excluded
            expected = [
                Path("root.py"),  # NOT ignored
                Path("sub/keep.log"),  # Re-included by sub/.gitignore !keep.log
                Path("sub/module.py"),  # NOT ignored
            ]

            assert relative_files == expected

    def test_negation_cannot_re_include_inside_excluded_dir(
        self,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """Negation patterns cannot re-include files inside an excluded directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Create root .gitignore that excludes the entire build/ directory
            root_gitignore = project_path / ".gitignore"
            root_gitignore.write_text("build/\n")

            # Create build directory
            build = project_path / "build"
            build.mkdir()

            # Create .gitignore inside build directory trying to re-include important.txt
            # This should NOT work because the parent directory is excluded
            build_gitignore = build / ".gitignore"
            build_gitignore.write_text("!important.txt\n")

            # Create files inside build directory
            (build / "important.txt").write_text("trying to re-include this")
            (build / "output.js").write_text("console.log('build');")
            (build / "bundle.js").write_text("console.log('bundle');")

            # Create files outside build directory that should be indexed
            (project_path / "main.py").write_text("print('hello')")
            (project_path / "readme.md").write_text("# Readme")

            # Create indexer and collect files
            indexer = Indexer(project_path, mock_config, mock_db, mock_embedder)
            files = indexer._collect_files()

            # Convert to relative paths for easier assertion
            relative_files = sorted([f.relative_to(project_path) for f in files])

            # All files in build/ should be excluded, even with negation pattern
            # The build/ directory pattern excludes the directory itself, so git doesn't recurse into it
            expected = [
                Path("main.py"),  # NOT ignored
                Path("readme.md"),  # NOT ignored
            ]

            assert relative_files == expected

    def test_escaped_exclamation_mark(
        self,
        mock_config: EmbeCodeConfig,
        mock_db: Mock,
        mock_embedder: Mock,
    ) -> None:
        """Pattern \\!readme should match a file literally named !readme (not a negation)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Create .gitignore with escaped exclamation mark pattern
            # This should match a file literally named "!readme", not act as a negation
            (project_path / ".gitignore").write_text("\\!readme\n")

            # Create a file literally named "!readme" (should be excluded)
            (project_path / "!readme").write_text("this file has an exclamation mark in the name")

            # Create other files (should be indexed)
            (project_path / "readme.txt").write_text("normal readme")
            (project_path / "main.py").write_text("print('hello')")

            # Create indexer and collect files
            indexer = Indexer(project_path, mock_config, mock_db, mock_embedder)
            files = indexer._collect_files()

            # Convert to relative paths for easier assertion
            relative_files = sorted([f.relative_to(project_path) for f in files])

            # The file named "!readme" should be excluded (matched by escaped pattern)
            # Other files should be included
            expected = [
                Path("main.py"),
                Path("readme.txt"),
            ]

            assert relative_files == expected
