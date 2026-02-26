"""Tests for CLI cache commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from embecode.cli import _cache_clean, _cache_purge, _cache_status, main


class TestCacheStatus:
    """Tests for cache status command."""

    def test_cache_status_empty(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test cache status with no cached projects."""
        mock_cache = MagicMock()
        mock_cache.get_cache_status.return_value = {
            "total_size_human": "0 B",
            "size_limit_human": "2.0 GB",
            "project_count": 0,
            "projects": [],
        }

        _cache_status(mock_cache)

        captured = capsys.readouterr()
        assert "Cache Status:" in captured.out
        assert "Total size: 0 B / 2.0 GB" in captured.out
        assert "Projects cached: 0" in captured.out
        assert "No cached projects." in captured.out

    def test_cache_status_with_projects(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test cache status with cached projects."""
        mock_cache = MagicMock()
        mock_cache.get_cache_status.return_value = {
            "total_size_human": "1.2 GB",
            "size_limit_human": "2.0 GB",
            "project_count": 2,
            "projects": [
                {
                    "hash": "a3f9b2c1",
                    "project_path": "/home/user/project1",
                    "last_accessed": "2025-02-24T10:30:00",
                    "size_human": "800.0 MB",
                },
                {
                    "hash": "7e4d1a8f",
                    "project_path": "/home/user/project2",
                    "last_accessed": "2025-02-23T09:15:00",
                    "size_human": "400.0 MB",
                },
            ],
        }

        _cache_status(mock_cache)

        captured = capsys.readouterr()
        assert "Cache Status:" in captured.out
        assert "Total size: 1.2 GB / 2.0 GB" in captured.out
        assert "Projects cached: 2" in captured.out
        assert "Cached projects (most recent first):" in captured.out
        assert "/home/user/project1" in captured.out
        assert "Size: 800.0 MB" in captured.out
        assert "Last accessed: 2025-02-24T10:30:00" in captured.out
        assert "Hash: a3f9b2c1" in captured.out
        assert "/home/user/project2" in captured.out


class TestCacheClean:
    """Tests for cache clean command."""

    def test_cache_clean_no_eviction(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test cache clean when no eviction is needed."""
        mock_cache = MagicMock()
        mock_cache.evict_lru.return_value = []

        _cache_clean(mock_cache)

        captured = capsys.readouterr()
        assert "Cache is already under size limit. No eviction needed." in captured.out

    def test_cache_clean_with_eviction(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test cache clean with evicted projects."""
        mock_cache = MagicMock()
        mock_cache.evict_lru.return_value = [
            "/home/user/old_project1",
            "/home/user/old_project2",
        ]

        _cache_clean(mock_cache)

        captured = capsys.readouterr()
        assert "Evicted 2 project(s):" in captured.out
        assert "/home/user/old_project1" in captured.out
        assert "/home/user/old_project2" in captured.out


class TestCachePurge:
    """Tests for cache purge command."""

    def test_cache_purge_all(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test purging all caches."""
        mock_cache = MagicMock()
        mock_cache.purge_all.return_value = 3

        _cache_purge(mock_cache, None)

        captured = capsys.readouterr()
        assert "Purged all caches (3 project(s))." in captured.out

    def test_cache_purge_specific_exists(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test purging a specific project that exists."""
        mock_cache = MagicMock()
        mock_cache.purge_project.return_value = True

        project_path = tmp_path / "myproject"
        project_path.mkdir()

        _cache_purge(mock_cache, str(project_path))

        captured = capsys.readouterr()
        assert f"Purged cache for: {project_path}" in captured.out

    def test_cache_purge_specific_not_exists(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test purging a specific project that doesn't exist."""
        mock_cache = MagicMock()
        mock_cache.purge_project.return_value = False

        project_path = tmp_path / "nonexistent"

        with pytest.raises(SystemExit) as exc_info:
            _cache_purge(mock_cache, str(project_path))

        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert f"No cache found for: {project_path}" in captured.out

    def test_cache_purge_current_directory(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test purging cache for current directory using '.'."""
        mock_cache = MagicMock()
        mock_cache.purge_project.return_value = True

        with patch("embecode.cli.Path.resolve", return_value=tmp_path):
            _cache_purge(mock_cache, ".")

        captured = capsys.readouterr()
        assert f"Purged cache for: {tmp_path}" in captured.out


class TestMainCacheCommands:
    """Integration tests for cache commands via main()."""

    @patch("embecode.cache.CacheManager")
    def test_cache_status_via_main(
        self, mock_cache_class: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test 'embecode cache status' command."""
        mock_cache = MagicMock()
        mock_cache.get_cache_status.return_value = {
            "total_size_human": "0 B",
            "size_limit_human": "2.0 GB",
            "project_count": 0,
            "projects": [],
        }
        mock_cache_class.return_value = mock_cache

        with patch("sys.argv", ["embecode", "cache", "status"]):
            main()

        captured = capsys.readouterr()
        assert "Cache Status:" in captured.out

    @patch("embecode.cache.CacheManager")
    def test_cache_clean_via_main(
        self, mock_cache_class: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test 'embecode cache clean' command."""
        mock_cache = MagicMock()
        mock_cache.evict_lru.return_value = []
        mock_cache_class.return_value = mock_cache

        with patch("sys.argv", ["embecode", "cache", "clean"]):
            main()

        captured = capsys.readouterr()
        assert "Cache is already under size limit" in captured.out

    @patch("embecode.cache.CacheManager")
    def test_cache_purge_all_via_main(
        self, mock_cache_class: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test 'embecode cache purge' command (purge all)."""
        mock_cache = MagicMock()
        mock_cache.purge_all.return_value = 2
        mock_cache_class.return_value = mock_cache

        with patch("sys.argv", ["embecode", "cache", "purge"]):
            main()

        captured = capsys.readouterr()
        assert "Purged all caches (2 project(s))." in captured.out

    @patch("embecode.cache.CacheManager")
    def test_cache_purge_specific_via_main(
        self, mock_cache_class: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test 'embecode cache purge <path>' command."""
        mock_cache = MagicMock()
        mock_cache.purge_project.return_value = True
        mock_cache_class.return_value = mock_cache

        project_path = tmp_path / "project"
        project_path.mkdir()

        with patch("sys.argv", ["embecode", "cache", "purge", str(project_path)]):
            main()

        captured = capsys.readouterr()
        assert f"Purged cache for: {project_path}" in captured.out

    @patch("embecode.cache.CacheManager")
    def test_cache_no_subcommand(
        self, mock_cache_class: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test 'embecode cache' with no subcommand shows help."""
        mock_cache = MagicMock()
        mock_cache_class.return_value = mock_cache

        with patch("sys.argv", ["embecode", "cache"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

        captured = capsys.readouterr()
        # Help text should be shown
        assert "Cache operations" in captured.out or "usage:" in captured.out


class TestMainServerBehavior:
    """Tests for server command behavior."""

    def test_default_path_when_no_command(self) -> None:
        """Test default server behavior when no command is specified."""
        with patch("sys.argv", ["embecode"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1

    def test_server_subcommand_with_path(self) -> None:
        """Test explicit server subcommand with --path."""
        with patch("sys.argv", ["embecode", "server", "--path", "/custom/path"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1

    def test_version_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test --version flag."""
        with patch("sys.argv", ["embecode", "--version"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            # Version prints and exits with 0
            assert exc_info.value.code == 0

            captured = capsys.readouterr()
            assert "embecode" in captured.out


class TestCLIEdgeCases:
    """Edge case tests for CLI."""

    @patch("embecode.cache.CacheManager")
    def test_cache_purge_relative_path_resolution(
        self, mock_cache_class: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that relative paths are resolved to absolute paths."""
        mock_cache = MagicMock()
        mock_cache.purge_project.return_value = True
        mock_cache_class.return_value = mock_cache

        # Change to tmp_path directory
        monkeypatch.chdir(tmp_path)

        with patch("sys.argv", ["embecode", "cache", "purge", "."]):
            main()

        # Verify purge_project was called with resolved absolute path
        mock_cache.purge_project.assert_called_once()
        called_path = mock_cache.purge_project.call_args[0][0]
        assert called_path == tmp_path

    @patch("embecode.cache.CacheManager")
    def test_cache_status_handles_exception(
        self, mock_cache_class: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test that cache status handles exceptions gracefully."""
        mock_cache = MagicMock()
        mock_cache.get_cache_status.side_effect = Exception("Test error")
        mock_cache_class.return_value = mock_cache

        with patch("sys.argv", ["embecode", "cache", "status"]):
            with pytest.raises(Exception, match="Test error"):
                main()

    @patch("embecode.cache.CacheManager")
    def test_multiple_cache_operations(
        self, mock_cache_class: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test running multiple cache operations in sequence."""
        mock_cache = MagicMock()
        mock_cache_class.return_value = mock_cache

        # First: status
        mock_cache.get_cache_status.return_value = {
            "total_size_human": "1.5 GB",
            "size_limit_human": "2.0 GB",
            "project_count": 1,
            "projects": [],
        }

        with patch("sys.argv", ["embecode", "cache", "status"]):
            main()

        # Second: clean
        mock_cache.evict_lru.return_value = []

        with patch("sys.argv", ["embecode", "cache", "clean"]):
            main()

        # Verify both operations were called
        assert mock_cache.get_cache_status.called
        assert mock_cache.evict_lru.called
