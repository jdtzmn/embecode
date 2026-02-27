"""Integration tests for multi-process owner/reader IPC architecture.

Verifies that the IPC layer correctly proxies search_code and index_status
requests from reader processes to the owner process over a real Unix domain
socket.  Also covers owner exit detection, concurrent reader load, and crash
recovery (stale lock with dead PID).

The first test group (``TestIPCOwnerReaderIntegration``) uses a **real**
``IPCServer`` and ``IPCClient`` connected via a real AF_UNIX socket, backed
by a lightweight mock server.  This exercises the full IPC transport path
without the cost of real DuckDB indexing.

The second test (``test_reader_sees_owner_written_data``) validates sequential
cross-process data visibility: the owner indexes, closes the DB, and a reader
process opens it and searches.

Run with:
    pytest tests/test_concurrent.py -v
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from embecode.db import Database
from embecode.ipc import IPCClient, IPCServer

# ---------------------------------------------------------------------------
# FixedVectorEmbedder — deterministic embedder for testing
# ---------------------------------------------------------------------------

_DIM = 768
random.seed(42)
_raw = [random.gauss(0, 1) for _ in range(_DIM)]
_norm = math.sqrt(sum(x * x for x in _raw))
_FIXED_UNIT_VECTOR: list[float] = [x / _norm for x in _raw]


class FixedVectorEmbedder:
    """Embedder that always returns the same pre-computed unit vector."""

    dimension = _DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_FIXED_UNIT_VECTOR[:] for _ in texts]

    def unload(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def short_tmp():
    """Create a short temporary directory under /tmp for Unix socket tests.

    macOS limits AF_UNIX paths to 104 bytes; pytest's ``tmp_path`` is often
    too long.  This fixture yields a short ``/tmp/ipc_XXXX`` path and cleans
    up after the test.
    """
    d = tempfile.mkdtemp(prefix="ipc_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def _make_fake_server(
    search_results: list[dict[str, Any]] | None = None,
    index_status: dict[str, Any] | None = None,
) -> Mock:
    """Return a mock EmbeCodeServer with predictable search and status responses.

    The mock has ``search_code`` and ``get_index_status`` methods that the
    ``IPCServer._dispatch`` machinery calls.
    """
    server = Mock()
    server.search_code.return_value = search_results or [
        {
            "file_path": "hello.py",
            "language": "python",
            "start_line": 1,
            "end_line": 3,
            "definitions": "function greet",
            "preview": "def greet():\n    print('hello')",
            "score": 0.95,
        }
    ]
    server.get_index_status.return_value = index_status or {
        "files_indexed": 10,
        "total_chunks": 50,
        "embedding_model": "test-model",
        "is_indexing": False,
        "last_updated": "2025-01-01T00:00:00",
        "current_file": None,
        "progress": None,
        "role": "owner",
    }
    return server


# ---------------------------------------------------------------------------
# IPC owner/reader integration tests (real sockets, mock server backend)
# ---------------------------------------------------------------------------


class TestIPCOwnerReaderIntegration:
    """Real IPC server ↔ client over Unix domain socket with mock backend.

    These tests verify the full message-framing + dispatch round-trip without
    requiring a real DuckDB database.
    """

    def test_reader_search_via_ipc(self, short_tmp: Path) -> None:
        """Reader sends search_code request to owner via IPC and gets results."""
        socket_path = short_tmp / "daemon.sock"
        fake_server = _make_fake_server()

        ipc_server = IPCServer(socket_path, fake_server)
        ipc_server.start()
        try:
            client = IPCClient(socket_path)
            client.connect()
            try:
                results = client.search_code(
                    query="greet function", mode="semantic", top_k=5, path="src/"
                )
                assert isinstance(results, list)
                assert len(results) == 1
                assert results[0]["file_path"] == "hello.py"
                assert results[0]["score"] == 0.95

                # Verify the server was called with the correct parameters
                fake_server.search_code.assert_called_once_with(
                    query="greet function", mode="semantic", top_k=5, path="src/"
                )
            finally:
                client.close()
        finally:
            ipc_server.stop()

    def test_reader_index_status_via_ipc(self, short_tmp: Path) -> None:
        """Reader sends index_status request to owner via IPC and gets status."""
        socket_path = short_tmp / "daemon.sock"
        fake_server = _make_fake_server()

        ipc_server = IPCServer(socket_path, fake_server)
        ipc_server.start()
        try:
            client = IPCClient(socket_path)
            client.connect()
            try:
                status = client.index_status()
                assert status["files_indexed"] == 10
                assert status["total_chunks"] == 50
                assert status["embedding_model"] == "test-model"
                assert status["is_indexing"] is False
                fake_server.get_index_status.assert_called_once()
            finally:
                client.close()
        finally:
            ipc_server.stop()

    def test_reader_detects_owner_exit(self, short_tmp: Path) -> None:
        """When the owner process exits, the reader gets ConnectionError.

        In production, the OS closes all file descriptors when the owner
        process exits.  Here we simulate this by explicitly shutting down
        the client-side socket from the server's handler threads, then
        stopping the server.
        """
        import socket as socket_mod

        socket_path = short_tmp / "daemon.sock"

        # Track handler connections so we can forcibly close them
        handler_connections: list[socket_mod.socket] = []
        orig_handle = IPCServer._handle_connection

        def tracking_handle(self_ipc: IPCServer, conn: socket_mod.socket) -> None:
            handler_connections.append(conn)
            return orig_handle(self_ipc, conn)

        fake_server = _make_fake_server()
        ipc_server = IPCServer(socket_path, fake_server)

        # Monkey-patch to capture handler connections
        ipc_server._handle_connection = lambda conn: tracking_handle(ipc_server, conn)  # type: ignore[assignment]

        ipc_server.start()

        client = IPCClient(socket_path)
        client.connect()

        # Verify connection works first
        results = client.search_code(query="test")
        assert len(results) == 1

        # Simulate owner process exit: forcibly close all handler connections
        # (this is what the OS does when the process terminates)
        for conn in handler_connections:
            try:
                conn.shutdown(socket_mod.SHUT_RDWR)
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass

        ipc_server.stop()

        # Reader's next request should raise ConnectionError
        with pytest.raises(ConnectionError):
            client.search_code(query="test after owner exit")

        client.close()

    def test_multiple_sequential_requests(self, short_tmp: Path) -> None:
        """A single reader connection can send many requests in sequence."""
        socket_path = short_tmp / "daemon.sock"
        fake_server = _make_fake_server()

        ipc_server = IPCServer(socket_path, fake_server)
        ipc_server.start()
        try:
            client = IPCClient(socket_path)
            client.connect()
            try:
                for i in range(20):
                    results = client.search_code(query=f"query_{i}")
                    assert len(results) == 1
                    assert results[0]["file_path"] == "hello.py"
                assert fake_server.search_code.call_count == 20
            finally:
                client.close()
        finally:
            ipc_server.stop()

    def test_concurrent_reader_clients(self, short_tmp: Path) -> None:
        """Multiple readers connect simultaneously and all get correct results."""
        socket_path = short_tmp / "daemon.sock"
        fake_server = _make_fake_server()

        ipc_server = IPCServer(socket_path, fake_server)
        ipc_server.start()
        try:
            num_clients = 5
            errors: list[Exception] = []
            results_per_client: list[list[dict[str, Any]]] = [[] for _ in range(num_clients)]

            def reader_worker(idx: int) -> None:
                try:
                    client = IPCClient(socket_path)
                    client.connect()
                    try:
                        for _ in range(10):
                            results = client.search_code(query=f"client_{idx}")
                            results_per_client[idx].extend(results)
                    finally:
                        client.close()
                except Exception as exc:
                    errors.append(exc)

            threads = [
                threading.Thread(target=reader_worker, args=(i,)) for i in range(num_clients)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=15)

            assert not errors, f"Reader threads raised errors: {errors}"
            for idx in range(num_clients):
                assert len(results_per_client[idx]) == 10, (
                    f"Client {idx} got {len(results_per_client[idx])} results, expected 10"
                )
            assert fake_server.search_code.call_count == num_clients * 10
        finally:
            ipc_server.stop()

    def test_reader_search_with_owner_error(self, short_tmp: Path) -> None:
        """When the owner's search_code raises, the IPC layer returns an error."""
        socket_path = short_tmp / "daemon.sock"
        fake_server = _make_fake_server()
        fake_server.search_code.side_effect = RuntimeError("Index corrupt")

        ipc_server = IPCServer(socket_path, fake_server)
        ipc_server.start()
        try:
            client = IPCClient(socket_path)
            client.connect()
            try:
                with pytest.raises(RuntimeError, match="Index corrupt"):
                    client.search_code(query="broken query")
            finally:
                client.close()
        finally:
            ipc_server.stop()

    def test_reader_index_status_with_owner_error(self, short_tmp: Path) -> None:
        """When the owner's get_index_status raises, the IPC layer returns an error."""
        socket_path = short_tmp / "daemon.sock"
        fake_server = _make_fake_server()
        fake_server.get_index_status.side_effect = RuntimeError("DB closed")

        ipc_server = IPCServer(socket_path, fake_server)
        ipc_server.start()
        try:
            client = IPCClient(socket_path)
            client.connect()
            try:
                with pytest.raises(RuntimeError, match="DB closed"):
                    client.index_status()
            finally:
                client.close()
        finally:
            ipc_server.stop()

    def test_reader_mixed_search_and_status_requests(self, short_tmp: Path) -> None:
        """Reader interleaves search_code and index_status on one connection."""
        socket_path = short_tmp / "daemon.sock"
        fake_server = _make_fake_server()

        ipc_server = IPCServer(socket_path, fake_server)
        ipc_server.start()
        try:
            client = IPCClient(socket_path)
            client.connect()
            try:
                # search → status → search → status
                r1 = client.search_code(query="first")
                assert r1[0]["file_path"] == "hello.py"

                s1 = client.index_status()
                assert s1["files_indexed"] == 10

                r2 = client.search_code(query="second", mode="keyword", top_k=3)
                assert len(r2) == 1

                s2 = client.index_status()
                assert s2["is_indexing"] is False
            finally:
                client.close()
        finally:
            ipc_server.stop()


# ---------------------------------------------------------------------------
# Crash recovery integration tests
# ---------------------------------------------------------------------------


class TestCrashRecovery:
    """Integration tests for stale lock file detection and recovery."""

    def test_stale_lock_with_dead_pid_allows_new_owner(self, short_tmp: Path) -> None:
        """A lock file referencing a dead PID is cleaned up by _acquire_lock.

        This test creates a lock file with a non-existent PID, then
        instantiates an EmbeCodeServer that should detect the stale lock,
        remove it, and become the owner.
        """
        from unittest.mock import patch

        from embecode.server import EmbeCodeServer

        cache_dir = short_tmp / "cache"
        cache_dir.mkdir()

        # Create a project directory
        project_dir = short_tmp / "project"
        project_dir.mkdir()
        (project_dir / "main.py").write_text("x = 1\n")

        lock_path = cache_dir / "daemon.lock"
        socket_path = cache_dir / "daemon.sock"

        # Write a stale lock with a definitely-dead PID
        dead_pid = 999_999_999
        lock_path.write_text(json.dumps({"pid": dead_pid}))
        assert lock_path.exists()

        # Also leave a stale socket file (simulating unclean shutdown)
        socket_path.touch()

        # Mock CacheManager to point at our cache dir
        mock_cm = Mock()
        mock_cm.get_cache_dir.return_value = cache_dir
        mock_cm.get_lock_path.return_value = lock_path
        mock_cm.get_socket_path.return_value = socket_path
        mock_cm.update_access_time.return_value = None

        mock_config = Mock()
        mock_config.embeddings.model = "test-model"
        mock_config.daemon.auto_watch = False
        mock_config.daemon.debounce_ms = 500

        mock_db = Mock()
        mock_db.get_metadata.return_value = None
        mock_db.set_metadata.return_value = None
        mock_db.connect.return_value = None
        mock_db.close.return_value = None
        mock_db._conn = object()

        with (
            patch("embecode.server.CacheManager", return_value=mock_cm),
            patch("embecode.server.load_config", return_value=mock_config),
            patch("embecode.server.Database", return_value=mock_db),
            patch("embecode.server.Embedder"),
            patch("embecode.server.Searcher"),
            patch("embecode.server.Indexer"),
            patch("embecode.server.IPCServer"),
            patch("embecode.server.threading.Thread"),
        ):
            server = EmbeCodeServer(project_dir)

            # The stale lock should have been removed and re-created
            assert server._role == "owner"

            # The lock file should now contain our PID
            with open(lock_path) as f:
                data = json.load(f)
            assert data["pid"] == os.getpid()

    def test_stale_socket_file_cleaned_on_ipc_server_start(self, short_tmp: Path) -> None:
        """IPCServer.start() removes a stale socket file before binding."""
        socket_path = short_tmp / "daemon.sock"
        # Create a stale socket file (just a regular file — bind would fail)
        socket_path.touch()
        assert socket_path.exists()

        fake_server = _make_fake_server()
        ipc_server = IPCServer(socket_path, fake_server)
        ipc_server.start()
        try:
            # Verify we can connect — the stale file was removed
            client = IPCClient(socket_path)
            client.connect()
            results = client.search_code(query="test")
            assert len(results) == 1
            client.close()
        finally:
            ipc_server.stop()


# ---------------------------------------------------------------------------
# Sequential cross-process data visibility (retained from v1)
# ---------------------------------------------------------------------------

_INDEXER_PROCESS_MODULE = "tests.helpers.indexer_process"
_INDEXER_TIMEOUT = 60  # seconds


def _run_indexer(db_path: Path, project_dir: Path) -> None:
    """Run the indexer subprocess and wait for it to finish.

    Raises ``RuntimeError`` if the subprocess exits with a non-zero code.
    """
    proc = subprocess.run(
        [sys.executable, "-m", _INDEXER_PROCESS_MODULE, str(db_path), str(project_dir)],
        capture_output=True,
        text=True,
        timeout=_INDEXER_TIMEOUT,
        cwd=str(Path(__file__).resolve().parent.parent),
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "src")},
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Indexer process failed (rc={proc.returncode}).\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    assert "done" in proc.stdout, f"Indexer did not print 'done'. stdout: {proc.stdout!r}"


@pytest.mark.integration
@pytest.mark.slow
def test_reader_sees_owner_written_data(tmp_path: Path) -> None:
    """After the owner process indexes a file and exits, a reader
    can open the same database and search for the indexed content.

    This validates sequential cross-process data visibility: the owner
    indexes (RW), closes its connection, and then a reader opens the DB
    and finds all committed data.
    """
    # -- create a project with a distinctive source file --------------------
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    src_file = project_dir / "unique_module.py"
    src_file.write_text(
        textwrap.dedent("""\
            def fibonacci_sequence(n: int) -> list[int]:
                \"\"\"Generate the first n numbers in the Fibonacci sequence.\"\"\"
                if n <= 0:
                    return []
                if n == 1:
                    return [0]
                seq = [0, 1]
                for _ in range(2, n):
                    seq.append(seq[-1] + seq[-2])
                return seq

            class QuantumCalculator:
                \"\"\"A calculator that performs quantum-inspired computations.\"\"\"

                def superposition(self, a: float, b: float) -> float:
                    return (a + b) / 2.0
        """),
    )

    db_path = tmp_path / "index.db"

    # -- Phase 1: owner subprocess indexes the project ----------------------
    _run_indexer(db_path, project_dir)

    # Verify the DB file was created
    assert db_path.exists(), "index.db was not created by the indexer subprocess"

    # -- Phase 2: reader opens and searches (no separate owner process here) -
    reader = Database(db_path)
    reader.connect()
    try:
        # Verify data was indexed
        stats = reader.get_index_stats()
        assert stats["total_chunks"] > 0, "No chunks found — indexer wrote nothing"
        assert stats["files_indexed"] > 0, "No files found — indexer wrote nothing"

        # Vector search — may return [] if the VSS extension cannot resolve
        # array_cosine_similarity(FLOAT[], FLOAT[]) due to a known DuckDB
        # type-cast limitation with dynamic-size arrays.  We still call it
        # to verify it does not raise an unhandled exception.
        results = reader.vector_search(_FIXED_UNIT_VECTOR, top_k=10)
        assert isinstance(results, list)
        if results:
            # If VSS works, verify content comes from our indexed file
            all_content = " ".join(r["content"] for r in results)
            assert "fibonacci_sequence" in all_content or "QuantumCalculator" in all_content, (
                f"Expected indexed content from unique_module.py, got: {all_content[:200]}"
            )

        # BM25 / keyword search for a distinctive term — the primary
        # assertion for cross-process data visibility
        results_bm25 = reader.bm25_search("fibonacci", top_k=5)
        assert isinstance(results_bm25, list)
        assert len(results_bm25) > 0, "bm25_search returned no results for 'fibonacci'"

        # Verify the BM25 results reference the correct file
        bm25_paths = [r["file_path"] for r in results_bm25]
        assert any("unique_module" in p for p in bm25_paths), (
            f"Expected unique_module.py in BM25 results, got paths: {bm25_paths}"
        )
    finally:
        reader.close()
