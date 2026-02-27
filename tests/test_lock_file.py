"""Unit tests for the lock-file daemon protocol.

Covers:
- Lock file creation (atomic, first-wins)
- Stale lock detection (dead PID → cleaned up)
- Reader role (no DB connection, no indexer/watcher started)
- Thread safety (concurrent DB operations)
- Cleanup (daemon.lock removed on clean owner shutdown)
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from embecode.server import (
    EmbeCodeServer,
    _cleanup_lock,
    is_pid_alive,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_cache_manager_mock(cache_dir: Path) -> Mock:
    """Return a CacheManager mock whose helpers point at *cache_dir*."""
    cm = Mock()
    cm.get_cache_dir.return_value = cache_dir
    cm.get_lock_path.return_value = cache_dir / "daemon.lock"
    cm.get_socket_path.return_value = cache_dir / "daemon.sock"
    cm.update_access_time.return_value = None
    return cm


def _make_config_mock(model: str = "test-model", auto_watch: bool = False) -> Mock:
    cfg = Mock()
    cfg.embeddings.model = model
    cfg.daemon.auto_watch = auto_watch
    cfg.daemon.debounce_ms = 500
    return cfg


def _make_db_mock(stored_model: str | None = None) -> Mock:
    db = Mock()
    db.get_metadata.return_value = stored_model
    db.set_metadata.return_value = None
    db.connect.return_value = None
    db.close.return_value = None
    db._conn = object()  # non-None so mid-promotion guard passes
    return db


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Isolated cache directory for each test."""
    d = tmp_path / "cache"
    d.mkdir()
    return d


@pytest.fixture
def temp_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    return project


# ---------------------------------------------------------------------------
# is_pid_alive
# ---------------------------------------------------------------------------


class TestIsPidAlive:
    """Tests for the is_pid_alive() utility."""

    def test_current_process_is_alive(self) -> None:
        assert is_pid_alive(os.getpid()) is True

    def test_dead_pid_returns_false(self) -> None:
        # PID 1 is init; we cannot guarantee it's dead, so use a PID that
        # definitely does not exist: find a free one the hard way.
        # os.kill with signal 0 raises ProcessLookupError for truly dead PIDs.
        dead_pid = 999_999_999  # absurdly large, will not exist
        assert is_pid_alive(dead_pid) is False

    def test_permission_error_treated_as_alive(self) -> None:
        """PermissionError from os.kill means the process exists → True."""
        with patch("os.kill", side_effect=PermissionError):
            assert is_pid_alive(1) is True


# ---------------------------------------------------------------------------
# _cleanup_lock
# ---------------------------------------------------------------------------


class TestCleanupLock:
    """Tests for the _cleanup_lock() module-level function."""

    def test_removes_lock_when_pid_matches(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "daemon.lock"
        socket_path = tmp_path / "daemon.sock"
        pid = os.getpid()
        lock_path.write_text(json.dumps({"pid": pid}))

        _cleanup_lock(lock_path, socket_path, pid)

        assert not lock_path.exists()

    def test_removes_socket_file(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "daemon.lock"
        socket_path = tmp_path / "daemon.sock"
        pid = os.getpid()
        lock_path.write_text(json.dumps({"pid": pid}))
        socket_path.write_text("")  # simulate leftover socket file

        _cleanup_lock(lock_path, socket_path, pid)

        assert not lock_path.exists()
        assert not socket_path.exists()

    def test_does_not_remove_when_pid_differs(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "daemon.lock"
        socket_path = tmp_path / "daemon.sock"
        other_pid = os.getpid() + 1  # just a different number
        lock_path.write_text(json.dumps({"pid": other_pid}))

        _cleanup_lock(lock_path, socket_path, os.getpid())

        assert lock_path.exists()

    def test_safe_when_lock_missing(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "nonexistent.lock"
        socket_path = tmp_path / "nonexistent.sock"
        # Should not raise
        _cleanup_lock(lock_path, socket_path, os.getpid())

    def test_safe_when_lock_corrupt(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "daemon.lock"
        socket_path = tmp_path / "daemon.sock"
        lock_path.write_text("not-json")
        # Should not raise
        _cleanup_lock(lock_path, socket_path, os.getpid())


# ---------------------------------------------------------------------------
# Lock file acquisition — owner path
# ---------------------------------------------------------------------------


class TestLockAcquisitionOwner:
    """First process creates lock file and becomes owner."""

    @patch("embecode.server.IPCServer")
    @patch("embecode.server.threading.Thread")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Database")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.load_config")
    def test_first_process_becomes_owner(
        self,
        mock_load_config: Mock,
        mock_cm_cls: Mock,
        mock_db_cls: Mock,
        mock_embedder_cls: Mock,
        mock_searcher_cls: Mock,
        mock_indexer_cls: Mock,
        mock_thread: Mock,
        mock_ipc_server_cls: Mock,
        temp_project: Path,
        temp_cache_dir: Path,
    ) -> None:
        mock_load_config.return_value = _make_config_mock()
        mock_cm_cls.return_value = _make_cache_manager_mock(temp_cache_dir)
        mock_db_cls.return_value = _make_db_mock()

        server = EmbeCodeServer(temp_project)

        assert server._role == "owner"

    @patch("embecode.server.IPCServer")
    @patch("embecode.server.threading.Thread")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Database")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.load_config")
    def test_owner_creates_lock_file_with_pid(
        self,
        mock_load_config: Mock,
        mock_cm_cls: Mock,
        mock_db_cls: Mock,
        mock_embedder_cls: Mock,
        mock_searcher_cls: Mock,
        mock_indexer_cls: Mock,
        mock_thread: Mock,
        mock_ipc_server_cls: Mock,
        temp_project: Path,
        temp_cache_dir: Path,
    ) -> None:
        mock_load_config.return_value = _make_config_mock()
        mock_cm_cls.return_value = _make_cache_manager_mock(temp_cache_dir)
        mock_db_cls.return_value = _make_db_mock()

        EmbeCodeServer(temp_project)

        lock_path = temp_cache_dir / "daemon.lock"
        assert lock_path.exists()
        data = json.loads(lock_path.read_text())
        assert data["pid"] == os.getpid()

    @patch("embecode.server.IPCServer")
    @patch("embecode.server.threading.Thread")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Database")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.load_config")
    def test_owner_opens_db_read_write(
        self,
        mock_load_config: Mock,
        mock_cm_cls: Mock,
        mock_db_cls: Mock,
        mock_embedder_cls: Mock,
        mock_searcher_cls: Mock,
        mock_indexer_cls: Mock,
        mock_thread: Mock,
        mock_ipc_server_cls: Mock,
        temp_project: Path,
        temp_cache_dir: Path,
    ) -> None:
        mock_load_config.return_value = _make_config_mock()
        mock_cm_cls.return_value = _make_cache_manager_mock(temp_cache_dir)
        db_mock = _make_db_mock()
        mock_db_cls.return_value = db_mock

        EmbeCodeServer(temp_project)

        db_mock.connect.assert_called_once_with()

    @patch("embecode.server.IPCServer")
    @patch("embecode.server.threading.Thread")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Database")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.load_config")
    def test_owner_spawns_catch_up_thread(
        self,
        mock_load_config: Mock,
        mock_cm_cls: Mock,
        mock_db_cls: Mock,
        mock_embedder_cls: Mock,
        mock_searcher_cls: Mock,
        mock_indexer_cls: Mock,
        mock_thread: Mock,
        mock_ipc_server_cls: Mock,
        temp_project: Path,
        temp_cache_dir: Path,
    ) -> None:
        mock_load_config.return_value = _make_config_mock()
        mock_cm_cls.return_value = _make_cache_manager_mock(temp_cache_dir)
        mock_db_cls.return_value = _make_db_mock()

        EmbeCodeServer(temp_project)

        # At least one daemon thread must have been started (catch-up indexer)
        assert mock_thread.call_count >= 1
        calls = mock_thread.call_args_list
        assert any(c.kwargs.get("daemon") is True for c in calls)

    @patch("embecode.server.IPCServer")
    @patch("embecode.server.threading.Thread")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Database")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.load_config")
    def test_owner_starts_ipc_server(
        self,
        mock_load_config: Mock,
        mock_cm_cls: Mock,
        mock_db_cls: Mock,
        mock_embedder_cls: Mock,
        mock_searcher_cls: Mock,
        mock_indexer_cls: Mock,
        mock_thread: Mock,
        mock_ipc_server_cls: Mock,
        temp_project: Path,
        temp_cache_dir: Path,
    ) -> None:
        """Owner must start the IPC server on daemon.sock."""
        mock_load_config.return_value = _make_config_mock()
        mock_cm_cls.return_value = _make_cache_manager_mock(temp_cache_dir)
        mock_db_cls.return_value = _make_db_mock()

        server = EmbeCodeServer(temp_project)

        # IPCServer should have been instantiated with the socket path
        mock_ipc_server_cls.assert_called_once_with(
            temp_cache_dir / "daemon.sock",
            server,
        )
        # start() should have been called
        mock_ipc_server_cls.return_value.start.assert_called_once()


# ---------------------------------------------------------------------------
# Lock file acquisition — reader path
# ---------------------------------------------------------------------------


class TestLockAcquisitionReader:
    """Second process sees existing live lock and becomes reader."""

    def _write_lock(self, cache_dir: Path, pid: int) -> None:
        lock_path = cache_dir / "daemon.lock"
        lock_path.write_text(json.dumps({"pid": pid}))

    @patch("embecode.server.IPCClient")
    @patch("embecode.server.threading.Thread")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Database")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.load_config")
    @patch("embecode.server.is_pid_alive", return_value=True)
    def test_second_process_becomes_reader(
        self,
        mock_is_alive: Mock,
        mock_load_config: Mock,
        mock_cm_cls: Mock,
        mock_db_cls: Mock,
        mock_embedder_cls: Mock,
        mock_searcher_cls: Mock,
        mock_indexer_cls: Mock,
        mock_thread: Mock,
        mock_ipc_client_cls: Mock,
        temp_project: Path,
        temp_cache_dir: Path,
    ) -> None:
        # Simulate existing live owner
        self._write_lock(temp_cache_dir, pid=12345)

        mock_load_config.return_value = _make_config_mock()
        mock_cm_cls.return_value = _make_cache_manager_mock(temp_cache_dir)
        mock_db_cls.return_value = _make_db_mock()

        server = EmbeCodeServer(temp_project)

        assert server._role == "reader"

    @patch("embecode.server.IPCClient")
    @patch("embecode.server.threading.Thread")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Database")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.load_config")
    @patch("embecode.server.is_pid_alive", return_value=True)
    def test_reader_does_not_open_db(
        self,
        mock_is_alive: Mock,
        mock_load_config: Mock,
        mock_cm_cls: Mock,
        mock_db_cls: Mock,
        mock_embedder_cls: Mock,
        mock_searcher_cls: Mock,
        mock_indexer_cls: Mock,
        mock_thread: Mock,
        mock_ipc_client_cls: Mock,
        temp_project: Path,
        temp_cache_dir: Path,
    ) -> None:
        """Reader must not open DuckDB — queries are proxied via IPC."""
        self._write_lock(temp_cache_dir, pid=12345)

        mock_load_config.return_value = _make_config_mock()
        mock_cm_cls.return_value = _make_cache_manager_mock(temp_cache_dir)
        db_mock = _make_db_mock()
        mock_db_cls.return_value = db_mock

        EmbeCodeServer(temp_project)

        db_mock.connect.assert_not_called()

    @patch("embecode.server.IPCClient")
    @patch("embecode.server.threading.Thread")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Database")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.load_config")
    @patch("embecode.server.is_pid_alive", return_value=True)
    def test_reader_does_not_start_indexer_or_file_watcher(
        self,
        mock_is_alive: Mock,
        mock_load_config: Mock,
        mock_cm_cls: Mock,
        mock_db_cls: Mock,
        mock_embedder_cls: Mock,
        mock_searcher_cls: Mock,
        mock_indexer_cls: Mock,
        mock_thread: Mock,
        mock_ipc_client_cls: Mock,
        temp_project: Path,
        temp_cache_dir: Path,
    ) -> None:
        """Reader must not launch catch-up indexing or file watching threads."""
        self._write_lock(temp_cache_dir, pid=12345)

        mock_load_config.return_value = _make_config_mock(auto_watch=True)
        mock_cm_cls.return_value = _make_cache_manager_mock(temp_cache_dir)
        mock_db_cls.return_value = _make_db_mock()

        mock_indexer = mock_indexer_cls.return_value

        server = EmbeCodeServer(temp_project)

        # Catch-up indexer must never be called
        mock_indexer.start_catchup_index.assert_not_called()
        # Watcher must not be set on the reader
        assert server.watcher is None

    @patch("embecode.server.IPCClient")
    @patch("embecode.server.threading.Thread")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Database")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.load_config")
    @patch("embecode.server.is_pid_alive", return_value=True)
    def test_reader_starts_lock_file_watcher_thread(
        self,
        mock_is_alive: Mock,
        mock_load_config: Mock,
        mock_cm_cls: Mock,
        mock_db_cls: Mock,
        mock_embedder_cls: Mock,
        mock_searcher_cls: Mock,
        mock_indexer_cls: Mock,
        mock_thread: Mock,
        mock_ipc_client_cls: Mock,
        temp_project: Path,
        temp_cache_dir: Path,
    ) -> None:
        """Reader must start the LockFileWatcher background thread."""
        self._write_lock(temp_cache_dir, pid=12345)

        mock_load_config.return_value = _make_config_mock()
        mock_cm_cls.return_value = _make_cache_manager_mock(temp_cache_dir)
        mock_db_cls.return_value = _make_db_mock()

        EmbeCodeServer(temp_project)

        # LockFileWatcher thread must be started
        thread_names = [c.kwargs.get("name") for c in mock_thread.call_args_list]
        assert "LockFileWatcher" in thread_names

    @patch("embecode.server.IPCClient")
    @patch("embecode.server.threading.Thread")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Database")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.load_config")
    @patch("embecode.server.is_pid_alive", return_value=True)
    def test_reader_connects_ipc_client(
        self,
        mock_is_alive: Mock,
        mock_load_config: Mock,
        mock_cm_cls: Mock,
        mock_db_cls: Mock,
        mock_embedder_cls: Mock,
        mock_searcher_cls: Mock,
        mock_indexer_cls: Mock,
        mock_thread: Mock,
        mock_ipc_client_cls: Mock,
        temp_project: Path,
        temp_cache_dir: Path,
    ) -> None:
        """Reader must connect IPC client to the owner's socket."""
        self._write_lock(temp_cache_dir, pid=12345)

        mock_load_config.return_value = _make_config_mock()
        mock_cm_cls.return_value = _make_cache_manager_mock(temp_cache_dir)
        mock_db_cls.return_value = _make_db_mock()

        EmbeCodeServer(temp_project)

        # IPCClient should have been instantiated with the socket path
        mock_ipc_client_cls.assert_called_once_with(temp_cache_dir / "daemon.sock")
        # connect() should have been called
        mock_ipc_client_cls.return_value.connect.assert_called_once()


# ---------------------------------------------------------------------------
# Stale lock detection
# ---------------------------------------------------------------------------


class TestStaleLockDetection:
    """Process with dead PID in lock file is treated as stale."""

    def _write_lock(self, cache_dir: Path, pid: int) -> None:
        lock_path = cache_dir / "daemon.lock"
        lock_path.write_text(json.dumps({"pid": pid}))

    @patch("embecode.server.IPCServer")
    @patch("embecode.server.threading.Thread")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Database")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.load_config")
    @patch("embecode.server.is_pid_alive", return_value=False)
    def test_stale_lock_removed_and_process_becomes_owner(
        self,
        mock_is_alive: Mock,
        mock_load_config: Mock,
        mock_cm_cls: Mock,
        mock_db_cls: Mock,
        mock_embedder_cls: Mock,
        mock_searcher_cls: Mock,
        mock_indexer_cls: Mock,
        mock_thread: Mock,
        mock_ipc_server_cls: Mock,
        temp_project: Path,
        temp_cache_dir: Path,
    ) -> None:
        dead_pid = 999_999_998
        self._write_lock(temp_cache_dir, pid=dead_pid)

        mock_load_config.return_value = _make_config_mock()
        mock_cm_cls.return_value = _make_cache_manager_mock(temp_cache_dir)
        mock_db_cls.return_value = _make_db_mock()

        server = EmbeCodeServer(temp_project)

        # Must have claimed ownership after removing stale lock
        assert server._role == "owner"

    @patch("embecode.server.IPCServer")
    @patch("embecode.server.threading.Thread")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Database")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.load_config")
    @patch("embecode.server.is_pid_alive", return_value=False)
    def test_stale_lock_replaced_with_current_pid(
        self,
        mock_is_alive: Mock,
        mock_load_config: Mock,
        mock_cm_cls: Mock,
        mock_db_cls: Mock,
        mock_embedder_cls: Mock,
        mock_searcher_cls: Mock,
        mock_indexer_cls: Mock,
        mock_thread: Mock,
        mock_ipc_server_cls: Mock,
        temp_project: Path,
        temp_cache_dir: Path,
    ) -> None:
        dead_pid = 999_999_997
        self._write_lock(temp_cache_dir, pid=dead_pid)

        mock_load_config.return_value = _make_config_mock()
        mock_cm_cls.return_value = _make_cache_manager_mock(temp_cache_dir)
        mock_db_cls.return_value = _make_db_mock()

        EmbeCodeServer(temp_project)

        lock_path = temp_cache_dir / "daemon.lock"
        assert lock_path.exists()
        data = json.loads(lock_path.read_text())
        assert data["pid"] == os.getpid()


# ---------------------------------------------------------------------------
# Role visibility in get_index_status
# ---------------------------------------------------------------------------


class TestRoleInIndexStatus:
    """get_index_status() includes the role field."""

    def _build_server(
        self,
        temp_project: Path,
        temp_cache_dir: Path,
        *,
        role: str,
        mock_load_config: Mock,
        mock_cm_cls: Mock,
        mock_db_cls: Mock,
        mock_indexer_cls: Mock,
    ) -> EmbeCodeServer:
        from embecode.indexer import IndexStatus

        mock_load_config.return_value = _make_config_mock()
        mock_cm_cls.return_value = _make_cache_manager_mock(temp_cache_dir)
        mock_db_cls.return_value = _make_db_mock()

        status = IndexStatus(
            files_indexed=0,
            total_chunks=0,
            embedding_model="test-model",
            last_updated=None,
            is_indexing=False,
        )
        mock_indexer_cls.return_value.get_status.return_value = status

        if role == "reader":
            lock_path = temp_cache_dir / "daemon.lock"
            lock_path.write_text(json.dumps({"pid": 12345}))

        return None  # defer actual creation to individual tests

    @patch("embecode.server.IPCServer")
    @patch("embecode.server.threading.Thread")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Database")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.load_config")
    def test_owner_role_in_status(
        self,
        mock_load_config: Mock,
        mock_cm_cls: Mock,
        mock_db_cls: Mock,
        mock_embedder_cls: Mock,
        mock_searcher_cls: Mock,
        mock_indexer_cls: Mock,
        mock_thread: Mock,
        mock_ipc_server_cls: Mock,
        temp_project: Path,
        temp_cache_dir: Path,
    ) -> None:
        from embecode.indexer import IndexStatus

        mock_load_config.return_value = _make_config_mock()
        mock_cm_cls.return_value = _make_cache_manager_mock(temp_cache_dir)
        mock_db_cls.return_value = _make_db_mock()

        status = IndexStatus(
            files_indexed=0,
            total_chunks=0,
            embedding_model="test-model",
            last_updated=None,
            is_indexing=False,
        )
        mock_indexer_cls.return_value.get_status.return_value = status

        server = EmbeCodeServer(temp_project)
        result = server.get_index_status()

        assert result["role"] == "owner"

    @patch("embecode.server.IPCClient")
    @patch("embecode.server.threading.Thread")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Database")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.load_config")
    @patch("embecode.server.is_pid_alive", return_value=True)
    def test_reader_role_in_status(
        self,
        mock_is_alive: Mock,
        mock_load_config: Mock,
        mock_cm_cls: Mock,
        mock_db_cls: Mock,
        mock_embedder_cls: Mock,
        mock_searcher_cls: Mock,
        mock_indexer_cls: Mock,
        mock_thread: Mock,
        mock_ipc_client_cls: Mock,
        temp_project: Path,
        temp_cache_dir: Path,
    ) -> None:
        from embecode.indexer import IndexStatus

        # Write live lock
        lock_path = temp_cache_dir / "daemon.lock"
        lock_path.write_text(json.dumps({"pid": 12345}))

        mock_load_config.return_value = _make_config_mock()
        mock_cm_cls.return_value = _make_cache_manager_mock(temp_cache_dir)
        mock_db_cls.return_value = _make_db_mock()

        status = IndexStatus(
            files_indexed=0,
            total_chunks=0,
            embedding_model="test-model",
            last_updated=None,
            is_indexing=False,
        )
        mock_indexer_cls.return_value.get_status.return_value = status

        # Mock IPC client to return a status dict
        mock_ipc_client_cls.return_value.index_status.return_value = status.to_dict()
        mock_ipc_client_cls.return_value.is_connected = True

        server = EmbeCodeServer(temp_project)
        result = server.get_index_status()

        assert result["role"] == "reader"


# ---------------------------------------------------------------------------
# Cleanup — owner removes lock
# ---------------------------------------------------------------------------


class TestOwnerCleanup:
    """Owner removes daemon.lock and daemon.sock on cleanup()."""

    @patch("embecode.server.IPCServer")
    @patch("embecode.server.threading.Thread")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Database")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.load_config")
    def test_cleanup_removes_lock_file(
        self,
        mock_load_config: Mock,
        mock_cm_cls: Mock,
        mock_db_cls: Mock,
        mock_embedder_cls: Mock,
        mock_searcher_cls: Mock,
        mock_indexer_cls: Mock,
        mock_thread: Mock,
        mock_ipc_server_cls: Mock,
        temp_project: Path,
        temp_cache_dir: Path,
    ) -> None:
        mock_load_config.return_value = _make_config_mock()
        mock_cm_cls.return_value = _make_cache_manager_mock(temp_cache_dir)
        mock_db_cls.return_value = _make_db_mock()

        server = EmbeCodeServer(temp_project)

        lock_path = temp_cache_dir / "daemon.lock"
        assert lock_path.exists(), "Lock must exist after __init__"

        server.cleanup()

        assert not lock_path.exists(), "Lock must be removed after cleanup()"

    @patch("embecode.server.IPCServer")
    @patch("embecode.server.threading.Thread")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Database")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.load_config")
    def test_cleanup_stops_ipc_server(
        self,
        mock_load_config: Mock,
        mock_cm_cls: Mock,
        mock_db_cls: Mock,
        mock_embedder_cls: Mock,
        mock_searcher_cls: Mock,
        mock_indexer_cls: Mock,
        mock_thread: Mock,
        mock_ipc_server_cls: Mock,
        temp_project: Path,
        temp_cache_dir: Path,
    ) -> None:
        """Owner cleanup must stop the IPC server."""
        mock_load_config.return_value = _make_config_mock()
        mock_cm_cls.return_value = _make_cache_manager_mock(temp_cache_dir)
        mock_db_cls.return_value = _make_db_mock()

        server = EmbeCodeServer(temp_project)
        server.cleanup()

        # IPC server's stop() must have been called
        mock_ipc_server_cls.return_value.stop.assert_called_once()

    @patch("embecode.server.IPCClient")
    @patch("embecode.server.threading.Thread")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Database")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.load_config")
    @patch("embecode.server.is_pid_alive", return_value=True)
    def test_reader_cleanup_does_not_remove_lock(
        self,
        mock_is_alive: Mock,
        mock_load_config: Mock,
        mock_cm_cls: Mock,
        mock_db_cls: Mock,
        mock_embedder_cls: Mock,
        mock_searcher_cls: Mock,
        mock_indexer_cls: Mock,
        mock_thread: Mock,
        mock_ipc_client_cls: Mock,
        temp_project: Path,
        temp_cache_dir: Path,
    ) -> None:
        # Write lock belonging to different process
        lock_path = temp_cache_dir / "daemon.lock"
        lock_path.write_text(json.dumps({"pid": 12345}))

        mock_load_config.return_value = _make_config_mock()
        mock_cm_cls.return_value = _make_cache_manager_mock(temp_cache_dir)
        mock_db_cls.return_value = _make_db_mock()

        server = EmbeCodeServer(temp_project)
        assert server._role == "reader"

        server.cleanup()

        # Lock must still exist (owned by PID 12345, not us)
        assert lock_path.exists()

    @patch("embecode.server.IPCClient")
    @patch("embecode.server.threading.Thread")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Database")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.load_config")
    @patch("embecode.server.is_pid_alive", return_value=True)
    def test_reader_cleanup_disconnects_ipc_client(
        self,
        mock_is_alive: Mock,
        mock_load_config: Mock,
        mock_cm_cls: Mock,
        mock_db_cls: Mock,
        mock_embedder_cls: Mock,
        mock_searcher_cls: Mock,
        mock_indexer_cls: Mock,
        mock_thread: Mock,
        mock_ipc_client_cls: Mock,
        temp_project: Path,
        temp_cache_dir: Path,
    ) -> None:
        """Reader cleanup must disconnect the IPC client."""
        lock_path = temp_cache_dir / "daemon.lock"
        lock_path.write_text(json.dumps({"pid": 12345}))

        mock_load_config.return_value = _make_config_mock()
        mock_cm_cls.return_value = _make_cache_manager_mock(temp_cache_dir)
        mock_db_cls.return_value = _make_db_mock()

        server = EmbeCodeServer(temp_project)
        assert server._role == "reader"

        server.cleanup()

        # IPC client's close() must have been called
        mock_ipc_client_cls.return_value.close.assert_called()


# ---------------------------------------------------------------------------
# Search mid-reconnect guard
# ---------------------------------------------------------------------------


class TestMidReconnectGuard:
    """search_code() returns a retriable error while DB connection is None."""

    @patch("embecode.server.IPCServer")
    @patch("embecode.server.threading.Thread")
    @patch("embecode.server.Indexer")
    @patch("embecode.server.Searcher")
    @patch("embecode.server.Embedder")
    @patch("embecode.server.Database")
    @patch("embecode.server.CacheManager")
    @patch("embecode.server.load_config")
    def test_returns_retriable_error_when_conn_is_none(
        self,
        mock_load_config: Mock,
        mock_cm_cls: Mock,
        mock_db_cls: Mock,
        mock_embedder_cls: Mock,
        mock_searcher_cls: Mock,
        mock_indexer_cls: Mock,
        mock_thread: Mock,
        mock_ipc_server_cls: Mock,
        temp_project: Path,
        temp_cache_dir: Path,
    ) -> None:
        mock_load_config.return_value = _make_config_mock()
        mock_cm_cls.return_value = _make_cache_manager_mock(temp_cache_dir)
        db_mock = _make_db_mock()
        mock_db_cls.return_value = db_mock

        server = EmbeCodeServer(temp_project)

        # Simulate mid-reconnect state
        db_mock._conn = None

        results = server.search_code("hello")

        assert len(results) == 1
        assert results[0].get("retry_recommended") is True
        assert "error" in results[0]


# ---------------------------------------------------------------------------
# Thread safety: concurrent DB operations
# ---------------------------------------------------------------------------


class TestDatabaseThreadSafety:
    """Concurrent access from multiple threads must not raise."""

    def test_concurrent_inserts_do_not_raise(self, tmp_path: Path) -> None:
        """Multiple threads inserting chunks concurrently must all succeed."""
        from embecode.db import Database

        db_path = tmp_path / "concurrent.db"
        db = Database(db_path)
        db.connect()

        errors: list[Exception] = []

        def insert_chunks(thread_id: int) -> None:
            records = [
                {
                    "file_path": f"thread_{thread_id}.py",
                    "language": "python",
                    "start_line": i,
                    "end_line": i + 4,
                    "content": f"def func_{thread_id}_{i}(): pass",
                    "context": f"thread_{thread_id}.py",
                    "hash": f"hash_{thread_id}_{i}",
                    "embedding": [float(thread_id)] * 3,
                }
                for i in range(1, 6)
            ]
            try:
                db.insert_chunks(records)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=insert_chunks, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Thread safety errors: {errors}"
        stats = db.get_index_stats()
        assert stats["total_chunks"] == 25  # 5 threads × 5 chunks each
        db.close()

    def test_concurrent_reads_and_writes_do_not_raise(self, tmp_path: Path) -> None:
        """Readers and writers sharing one DB object must not corrupt state."""
        from embecode.db import Database

        db_path = tmp_path / "rw_concurrent.db"
        db = Database(db_path)
        db.connect()

        # Pre-populate
        db.insert_chunks(
            [
                {
                    "file_path": "base.py",
                    "language": "python",
                    "start_line": 1,
                    "end_line": 5,
                    "content": "def base(): pass",
                    "context": "base.py",
                    "hash": "base_hash",
                    "embedding": [0.1, 0.2, 0.3],
                }
            ]
        )

        errors: list[Exception] = []

        def reader() -> None:
            for _ in range(10):
                try:
                    db.get_index_stats()
                except Exception as exc:
                    errors.append(exc)

        def writer(thread_id: int) -> None:
            for i in range(5):
                try:
                    db.insert_chunks(
                        [
                            {
                                "file_path": f"writer_{thread_id}.py",
                                "language": "python",
                                "start_line": i,
                                "end_line": i + 4,
                                "content": f"def w{thread_id}_{i}(): pass",
                                "context": f"writer_{thread_id}.py",
                                "hash": f"w_hash_{thread_id}_{i}",
                                "embedding": [float(thread_id), float(i), 0.0],
                            }
                        ]
                    )
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(3)] + [
            threading.Thread(target=writer, args=(i,)) for i in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert errors == [], f"Concurrency errors: {errors}"
        db.close()
