"""Tests for cache.py - cache management."""

import shutil
from datetime import datetime
from pathlib import Path

import pytest

from embecode.cache import CacheEntry, CacheManager


@pytest.fixture
def temp_cache_root(tmp_path: Path) -> Path:
    """Create a temporary cache root directory."""
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    return cache_root


@pytest.fixture
def cache_manager(temp_cache_root: Path) -> CacheManager:
    """Create a CacheManager with a temporary cache root."""
    return CacheManager(cache_root=temp_cache_root, size_limit_bytes=1024 * 1024)  # 1MB


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Create a temporary project directory."""
    project = tmp_path / "project"
    project.mkdir()
    return project


class TestCacheManager:
    """Tests for CacheManager."""

    def test_initialization_creates_cache_root(self, tmp_path: Path) -> None:
        """Test that cache root is created if it doesn't exist."""
        cache_root = tmp_path / "new_cache"
        assert not cache_root.exists()

        manager = CacheManager(cache_root=cache_root)
        assert cache_root.exists()
        assert manager.cache_root == cache_root

    def test_initialization_default_cache_root(self) -> None:
        """Test that default cache root is ~/.cache/embecode."""
        manager = CacheManager()
        expected = Path.home() / ".cache" / "embecode"
        assert manager.cache_root == expected

    def test_get_cache_dir(self, cache_manager: CacheManager, project_dir: Path) -> None:
        """Test getting cache directory for a project."""
        cache_dir = cache_manager.get_cache_dir(project_dir)

        # Should be a subdirectory of cache root
        assert cache_dir.parent == cache_manager.cache_root

        # Should exist
        assert cache_dir.exists()

        # Hash should be first 8 chars of SHA1
        import hashlib

        expected_hash = hashlib.sha1(str(project_dir.resolve()).encode()).hexdigest()[:8]
        assert cache_dir.name == expected_hash

    def test_get_cache_dir_idempotent(self, cache_manager: CacheManager, project_dir: Path) -> None:
        """Test that get_cache_dir returns the same directory on multiple calls."""
        dir1 = cache_manager.get_cache_dir(project_dir)
        dir2 = cache_manager.get_cache_dir(project_dir)
        assert dir1 == dir2

    def test_update_access_time(self, cache_manager: CacheManager, project_dir: Path) -> None:
        """Test updating access time for a project."""
        cache_dir = cache_manager.get_cache_dir(project_dir)

        # Create a file in cache dir to give it some size
        (cache_dir / "test.db").write_text("test data")

        # Update access time
        before = datetime.now()
        cache_manager.update_access_time(project_dir)
        after = datetime.now()

        # Check registry
        registry = cache_manager._load_registry()
        cache_hash = cache_dir.name

        assert cache_hash in registry
        entry = registry[cache_hash]

        assert entry.project_path == str(project_dir.resolve())
        assert entry.size_bytes > 0

        # Check timestamp is between before and after
        timestamp = datetime.fromisoformat(entry.last_accessed)
        assert before <= timestamp <= after

    def test_update_access_time_nonexistent_cache(
        self, cache_manager: CacheManager, project_dir: Path
    ) -> None:
        """Test that update_access_time does nothing if cache doesn't exist."""
        # Don't create cache dir
        cache_manager.update_access_time(project_dir)

        # Registry should be empty
        registry = cache_manager._load_registry()
        assert len(registry) == 0

    def test_get_cache_status_empty(self, cache_manager: CacheManager) -> None:
        """Test getting cache status with no projects."""
        status = cache_manager.get_cache_status()

        assert status["total_size_bytes"] == 0
        assert status["project_count"] == 0
        assert status["projects"] == []
        assert status["size_limit_bytes"] == cache_manager.size_limit_bytes

    def test_get_cache_status_with_projects(
        self, cache_manager: CacheManager, tmp_path: Path
    ) -> None:
        """Test getting cache status with multiple projects."""
        # Create two projects
        project1 = tmp_path / "project1"
        project1.mkdir()
        project2 = tmp_path / "project2"
        project2.mkdir()

        # Create cache dirs with files
        cache_dir1 = cache_manager.get_cache_dir(project1)
        cache_dir2 = cache_manager.get_cache_dir(project2)

        (cache_dir1 / "test1.db").write_text("a" * 100)
        (cache_dir2 / "test2.db").write_text("b" * 200)

        # Update access times
        cache_manager.update_access_time(project1)
        cache_manager.update_access_time(project2)

        # Get status
        status = cache_manager.get_cache_status()

        assert status["project_count"] == 2
        assert status["total_size_bytes"] == 300
        assert len(status["projects"]) == 2

        # Projects should be sorted by last_accessed (most recent first)
        assert status["projects"][0]["project_path"] == str(project2.resolve())
        assert status["projects"][1]["project_path"] == str(project1.resolve())

    def test_evict_lru_under_limit(self, cache_manager: CacheManager, project_dir: Path) -> None:
        """Test that evict_lru does nothing when under size limit."""
        cache_dir = cache_manager.get_cache_dir(project_dir)
        (cache_dir / "test.db").write_text("small")
        cache_manager.update_access_time(project_dir)

        evicted = cache_manager.evict_lru()

        assert len(evicted) == 0
        assert cache_dir.exists()

    def test_evict_lru_over_limit(self, cache_manager: CacheManager, tmp_path: Path) -> None:
        """Test that evict_lru evicts oldest projects when over limit."""
        # Create three projects
        project1 = tmp_path / "project1"
        project1.mkdir()
        project2 = tmp_path / "project2"
        project2.mkdir()
        project3 = tmp_path / "project3"
        project3.mkdir()

        # Create cache dirs with large files
        cache_dir1 = cache_manager.get_cache_dir(project1)
        cache_dir2 = cache_manager.get_cache_dir(project2)
        cache_dir3 = cache_manager.get_cache_dir(project3)

        # Each file is 400KB, total 1.2MB (over 1MB limit)
        (cache_dir1 / "test1.db").write_text("a" * 400_000)
        (cache_dir2 / "test2.db").write_text("b" * 400_000)
        (cache_dir3 / "test3.db").write_text("c" * 400_000)

        # Update access times in order (oldest to newest)
        cache_manager.update_access_time(project1)
        cache_manager.update_access_time(project2)
        cache_manager.update_access_time(project3)

        # Evict
        evicted = cache_manager.evict_lru()

        # Should evict oldest project (project1)
        assert str(project1.resolve()) in evicted
        assert not cache_dir1.exists()

        # Newer projects should remain
        assert cache_dir2.exists()
        assert cache_dir3.exists()

    def test_purge_all(self, cache_manager: CacheManager, tmp_path: Path) -> None:
        """Test purging all caches."""
        # Create two projects
        project1 = tmp_path / "project1"
        project1.mkdir()
        project2 = tmp_path / "project2"
        project2.mkdir()

        cache_dir1 = cache_manager.get_cache_dir(project1)
        cache_dir2 = cache_manager.get_cache_dir(project2)

        (cache_dir1 / "test1.db").write_text("test1")
        (cache_dir2 / "test2.db").write_text("test2")

        cache_manager.update_access_time(project1)
        cache_manager.update_access_time(project2)

        # Purge
        count = cache_manager.purge_all()

        assert count == 2
        assert not cache_dir1.exists()
        assert not cache_dir2.exists()

        # Registry should be empty
        registry = cache_manager._load_registry()
        assert len(registry) == 0

    def test_purge_project(self, cache_manager: CacheManager, project_dir: Path) -> None:
        """Test purging cache for a specific project."""
        cache_dir = cache_manager.get_cache_dir(project_dir)
        (cache_dir / "test.db").write_text("test data")
        cache_manager.update_access_time(project_dir)

        # Purge
        result = cache_manager.purge_project(project_dir)

        assert result is True
        assert not cache_dir.exists()

        # Registry should not have this project
        registry = cache_manager._load_registry()
        assert cache_dir.name not in registry

    def test_purge_project_nonexistent(
        self, cache_manager: CacheManager, project_dir: Path
    ) -> None:
        """Test purging cache for a project that doesn't exist."""
        result = cache_manager.purge_project(project_dir)
        assert result is False

    def test_clean_stale_entries(self, cache_manager: CacheManager, tmp_path: Path) -> None:
        """Test cleaning stale entries for deleted projects."""
        # Create a project and cache it
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        cache_dir = cache_manager.get_cache_dir(project_dir)
        (cache_dir / "test.db").write_text("test data")
        cache_manager.update_access_time(project_dir)

        # Verify cache exists
        registry = cache_manager._load_registry()
        assert len(registry) == 1

        # Delete the project directory
        shutil.rmtree(project_dir)

        # Clean stale entries
        registry = cache_manager._load_registry()
        cache_manager._clean_stale_entries(registry)

        # Registry should be empty now
        registry = cache_manager._load_registry()
        assert len(registry) == 0

        # Cache directory should be deleted
        assert not cache_dir.exists()

    def test_load_registry_empty(self, cache_manager: CacheManager) -> None:
        """Test loading empty registry."""
        registry = cache_manager._load_registry()
        assert len(registry) == 0

    def test_load_registry_corrupted(
        self, cache_manager: CacheManager, temp_cache_root: Path
    ) -> None:
        """Test loading corrupted registry returns empty dict."""
        # Write invalid JSON
        (temp_cache_root / "registry.json").write_text("not valid json")

        registry = cache_manager._load_registry()
        assert len(registry) == 0

    def test_save_and_load_registry(self, cache_manager: CacheManager) -> None:
        """Test saving and loading registry."""
        registry = {
            "abc123": CacheEntry(
                project_path="/path/to/project",
                last_accessed="2025-02-24T10:30:00",
                size_bytes=1000000,
            )
        }

        cache_manager._save_registry(registry)

        loaded = cache_manager._load_registry()
        assert len(loaded) == 1
        assert "abc123" in loaded

        entry = loaded["abc123"]
        assert entry.project_path == "/path/to/project"
        assert entry.last_accessed == "2025-02-24T10:30:00"
        assert entry.size_bytes == 1000000

    def test_hash_path_deterministic(self) -> None:
        """Test that hash_path is deterministic."""
        path = "/path/to/project"
        hash1 = CacheManager._hash_path(path)
        hash2 = CacheManager._hash_path(path)

        assert hash1 == hash2
        assert len(hash1) == 8

    def test_hash_path_different_paths(self) -> None:
        """Test that different paths produce different hashes."""
        hash1 = CacheManager._hash_path("/path/to/project1")
        hash2 = CacheManager._hash_path("/path/to/project2")

        assert hash1 != hash2

    def test_calculate_dir_size(self, tmp_path: Path) -> None:
        """Test calculating directory size."""
        test_dir = tmp_path / "test"
        test_dir.mkdir()

        (test_dir / "file1.txt").write_text("a" * 100)
        (test_dir / "file2.txt").write_text("b" * 200)

        subdir = test_dir / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").write_text("c" * 300)

        size = CacheManager._calculate_dir_size(test_dir)
        assert size == 600

    def test_calculate_dir_size_empty(self, tmp_path: Path) -> None:
        """Test calculating size of empty directory."""
        test_dir = tmp_path / "empty"
        test_dir.mkdir()

        size = CacheManager._calculate_dir_size(test_dir)
        assert size == 0

    def test_human_readable_size(self) -> None:
        """Test human-readable size formatting."""
        assert CacheManager._human_readable_size(0) == "0.0 B"
        assert CacheManager._human_readable_size(500) == "500.0 B"
        assert CacheManager._human_readable_size(1024) == "1.0 KB"
        assert CacheManager._human_readable_size(1536) == "1.5 KB"
        assert CacheManager._human_readable_size(1024 * 1024) == "1.0 MB"
        assert CacheManager._human_readable_size(1024 * 1024 * 1024) == "1.0 GB"
        assert CacheManager._human_readable_size(1024 * 1024 * 1024 * 1024) == "1.0 TB"


class TestCacheIntegration:
    """Integration tests for cache management."""

    def test_complete_workflow(self, cache_manager: CacheManager, tmp_path: Path) -> None:
        """Test complete cache workflow: create, update, check status, evict."""
        # Create projects
        project1 = tmp_path / "project1"
        project1.mkdir()
        project2 = tmp_path / "project2"
        project2.mkdir()

        # Get cache dirs
        cache_dir1 = cache_manager.get_cache_dir(project1)
        cache_dir2 = cache_manager.get_cache_dir(project2)

        # Add files
        (cache_dir1 / "index.db").write_text("a" * 500_000)
        (cache_dir2 / "index.db").write_text("b" * 600_000)

        # Update access times
        cache_manager.update_access_time(project1)
        cache_manager.update_access_time(project2)

        # Check status
        status = cache_manager.get_cache_status()
        assert status["project_count"] == 2
        assert status["total_size_bytes"] == 1_100_000

        # Trigger eviction (over 1MB limit)
        evicted = cache_manager.evict_lru()

        # Should evict project1 (oldest)
        assert len(evicted) == 1
        assert str(project1.resolve()) in evicted

        # Check status again
        status = cache_manager.get_cache_status()
        assert status["project_count"] == 1
        assert status["total_size_bytes"] == 600_000

    def test_stale_cleanup_on_status(self, cache_manager: CacheManager, tmp_path: Path) -> None:
        """Test that stale entries are cleaned when checking status."""
        # Create a project
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        cache_dir = cache_manager.get_cache_dir(project_dir)
        (cache_dir / "test.db").write_text("test")
        cache_manager.update_access_time(project_dir)

        # Verify it exists
        status = cache_manager.get_cache_status()
        assert status["project_count"] == 1

        # Delete the project
        shutil.rmtree(project_dir)

        # Check status again - should clean stale entry
        status = cache_manager.get_cache_status()
        assert status["project_count"] == 0

    def test_multiple_updates_preserve_latest(
        self, cache_manager: CacheManager, project_dir: Path
    ) -> None:
        """Test that multiple updates preserve the latest access time."""
        cache_dir = cache_manager.get_cache_dir(project_dir)
        (cache_dir / "test.db").write_text("test")

        # First update
        cache_manager.update_access_time(project_dir)
        registry1 = cache_manager._load_registry()
        time1 = registry1[cache_dir.name].last_accessed

        # Second update (should be later)
        import time

        time.sleep(0.01)
        cache_manager.update_access_time(project_dir)
        registry2 = cache_manager._load_registry()
        time2 = registry2[cache_dir.name].last_accessed

        assert time2 > time1


class TestCacheEdgeCases:
    """Edge case tests for cache management."""

    def test_get_cache_dir_with_relative_path(
        self, cache_manager: CacheManager, tmp_path: Path, monkeypatch
    ) -> None:
        """Test that relative paths are resolved to absolute."""
        project = tmp_path / "project"
        project.mkdir()

        # Change to project directory
        monkeypatch.chdir(project)

        # Use relative path
        cache_dir1 = cache_manager.get_cache_dir(".")
        cache_dir2 = cache_manager.get_cache_dir(project)

        assert cache_dir1 == cache_dir2

    def test_evict_lru_with_no_registry(self, cache_manager: CacheManager) -> None:
        """Test eviction with no registry file."""
        evicted = cache_manager.evict_lru()
        assert len(evicted) == 0

    def test_registry_survives_corrupted_reads(
        self, cache_manager: CacheManager, temp_cache_root: Path, project_dir: Path
    ) -> None:
        """Test that corrupted registry doesn't break subsequent operations."""
        # Create valid entry
        cache_dir = cache_manager.get_cache_dir(project_dir)
        (cache_dir / "test.db").write_text("test")
        cache_manager.update_access_time(project_dir)

        # Corrupt registry
        (temp_cache_root / "registry.json").write_text("corrupted")

        # Load should return empty
        registry = cache_manager._load_registry()
        assert len(registry) == 0

        # But we can still use the cache manager
        cache_manager.update_access_time(project_dir)

        # Should create new valid registry
        registry = cache_manager._load_registry()
        assert len(registry) == 1

    def test_calculate_dir_size_with_inaccessible_files(self, tmp_path: Path, monkeypatch) -> None:
        """Test that inaccessible files don't crash dir size calculation."""
        test_dir = tmp_path / "test"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("test")

        # Mock rglob to raise OSError
        def mock_rglob(*args, **kwargs):
            raise OSError("Permission denied")

        monkeypatch.setattr(Path, "rglob", mock_rglob)

        size = CacheManager._calculate_dir_size(test_dir)
        assert size == 0

    def test_save_registry_failure(
        self, cache_manager: CacheManager, temp_cache_root: Path, monkeypatch
    ) -> None:
        """Test that registry save failure doesn't crash."""
        # Mock open to raise OSError
        import builtins

        original_open = builtins.open

        def mock_open(path, *args, **kwargs):
            if str(path).endswith("registry.json") and "w" in args:
                raise OSError("Permission denied")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", mock_open)

        # Should not crash
        registry = {"test": CacheEntry("/path", "2025-02-24T10:30:00", 1000)}
        cache_manager._save_registry(registry)

    def test_purge_project_with_registry_entry_but_no_dir(
        self, cache_manager: CacheManager, tmp_path: Path
    ) -> None:
        """Test purging when registry has entry but cache dir doesn't exist."""
        project = tmp_path / "project"
        project.mkdir()

        cache_dir = cache_manager.get_cache_dir(project)
        (cache_dir / "test.db").write_text("test")
        cache_manager.update_access_time(project)

        # Delete cache dir manually (but leave registry entry)
        shutil.rmtree(cache_dir)

        # Purge should return True (registry entry existed)
        result = cache_manager.purge_project(project)
        assert result is True

        # Registry entry should be removed
        registry = cache_manager._load_registry()
        assert cache_dir.name not in registry

    def test_get_lock_path(self, cache_manager: CacheManager, project_dir: Path) -> None:
        """Test getting lock file path for a project."""
        lock_path = cache_manager.get_lock_path(project_dir)

        # Should be daemon.lock inside the project's cache directory
        assert lock_path.name == "daemon.lock"
        assert lock_path.parent == cache_manager.get_cache_dir(project_dir)

    def test_get_socket_path(self, cache_manager: CacheManager, project_dir: Path) -> None:
        """Test getting socket file path for a project."""
        socket_path = cache_manager.get_socket_path(project_dir)

        # Should be daemon.sock inside the project's cache directory
        assert socket_path.name == "daemon.sock"
        assert socket_path.parent == cache_manager.get_cache_dir(project_dir)

    def test_get_socket_path_same_cache_dir_as_lock(
        self, cache_manager: CacheManager, project_dir: Path
    ) -> None:
        """Test that socket and lock paths share the same cache directory."""
        lock_path = cache_manager.get_lock_path(project_dir)
        socket_path = cache_manager.get_socket_path(project_dir)

        assert lock_path.parent == socket_path.parent

    def test_get_socket_path_deterministic(
        self, cache_manager: CacheManager, project_dir: Path
    ) -> None:
        """Test that socket path is deterministic for the same project."""
        path1 = cache_manager.get_socket_path(project_dir)
        path2 = cache_manager.get_socket_path(project_dir)
        assert path1 == path2
