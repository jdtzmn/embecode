"""Unit tests for the IPC server/client module.

Covers:
- Message framing round-trip (send_message / recv_message) with various payload sizes
- IPCServer request dispatch (search_code, index_status)
- IPCServer unknown method returns error
- IPCClient connection drop raises ConnectionError
- IPCClient search_code and index_status proxying
- IPCServer start/stop lifecycle
"""

from __future__ import annotations

import socket
import struct
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from embecode.ipc import (
    _HEADER_FORMAT,
    _MAX_MESSAGE_SIZE,
    IPCClient,
    IPCServer,
    recv_message,
    send_message,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def short_tmp(tmp_path: Path) -> Path:
    """Return a short temp directory suitable for Unix socket paths.

    macOS limits AF_UNIX paths to 104 bytes. pytest's ``tmp_path`` can
    exceed this, so we create a short-named directory under ``/tmp``.
    """
    d = Path(tempfile.mkdtemp(prefix="ipc_"))
    yield d  # type: ignore[misc]
    import shutil

    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _socketpair() -> tuple[socket.socket, socket.socket]:
    """Create a connected pair of Unix domain sockets for testing."""
    return socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)


def _make_embecode_server_mock(
    search_results: list[dict[str, Any]] | None = None,
    index_status: dict[str, Any] | None = None,
) -> Mock:
    """Build a mock EmbeCodeServer with search_code and get_index_status."""
    srv = Mock()
    srv.search_code.return_value = search_results or []
    srv.get_index_status.return_value = index_status or {
        "files_indexed": 10,
        "total_chunks": 100,
        "role": "owner",
    }
    return srv


# ---------------------------------------------------------------------------
# Message framing tests
# ---------------------------------------------------------------------------


class TestMessageFraming:
    """Tests for send_message / recv_message helpers."""

    def test_round_trip_small(self):
        """Small payload round-trips correctly."""
        a, b = _socketpair()
        try:
            data = {"method": "search_code", "params": {"query": "hello"}}
            send_message(a, data)
            received = recv_message(b)
            assert received == data
        finally:
            a.close()
            b.close()

    def test_round_trip_empty_dict(self):
        """An empty dictionary round-trips correctly."""
        a, b = _socketpair()
        try:
            data: dict[str, Any] = {}
            send_message(a, data)
            received = recv_message(b)
            assert received == data
        finally:
            a.close()
            b.close()

    def test_round_trip_large_payload(self):
        """A payload with many results round-trips correctly.

        Uses a sender thread because the payload (~112 KB) exceeds the
        default Unix socket buffer size, so sendall blocks until the
        receiver starts draining.
        """
        a, b = _socketpair()
        try:
            results = [
                {
                    "file_path": f"src/module_{i}.py",
                    "content": f"def func_{i}(): pass\n" * 50,
                    "score": 0.95 - i * 0.01,
                }
                for i in range(100)
            ]
            data = {"result": results}

            errors: list[Exception] = []

            def _sender() -> None:
                try:
                    send_message(a, data)
                except Exception as exc:
                    errors.append(exc)

            t = threading.Thread(target=_sender)
            t.start()
            received = recv_message(b)
            t.join(timeout=5)

            assert not errors, f"Sender error: {errors}"
            assert received == data
            assert len(received["result"]) == 100
        finally:
            a.close()
            b.close()

    def test_round_trip_unicode(self):
        """Unicode content in payload round-trips correctly."""
        a, b = _socketpair()
        try:
            data = {"result": [{"content": "// comment: hello world"}]}
            send_message(a, data)
            received = recv_message(b)
            assert received == data
        finally:
            a.close()
            b.close()

    def test_recv_message_clean_disconnect(self):
        """recv_message returns None when the peer closes cleanly."""
        a, b = _socketpair()
        try:
            a.close()
            result = recv_message(b)
            assert result is None
        finally:
            b.close()

    def test_recv_message_mid_read_disconnect(self):
        """recv_message raises ConnectionError when connection drops mid-payload."""
        a, b = _socketpair()
        try:
            # Send a header claiming 1000 bytes, then close before sending payload
            header = struct.pack(_HEADER_FORMAT, 1000)
            a.sendall(header)
            a.close()
            with pytest.raises(ConnectionError, match="closed while reading"):
                recv_message(b)
        finally:
            b.close()

    def test_recv_message_oversized_payload(self):
        """recv_message raises ValueError for payload exceeding max size."""
        a, b = _socketpair()
        try:
            header = struct.pack(_HEADER_FORMAT, _MAX_MESSAGE_SIZE + 1)
            a.sendall(header)
            with pytest.raises(ValueError, match="too large"):
                recv_message(b)
        finally:
            a.close()
            b.close()

    def test_recv_message_invalid_json(self):
        """recv_message raises ValueError for non-JSON payload."""
        a, b = _socketpair()
        try:
            payload = b"not valid json {"
            header = struct.pack(_HEADER_FORMAT, len(payload))
            a.sendall(header + payload)
            with pytest.raises(ValueError, match="Invalid IPC message"):
                recv_message(b)
        finally:
            a.close()
            b.close()

    def test_send_message_to_closed_socket(self):
        """send_message raises ConnectionError when the socket is closed."""
        a, b = _socketpair()
        b.close()
        try:
            with pytest.raises(ConnectionError, match="Failed to send"):
                # May need to send enough data to trigger SIGPIPE/EPIPE
                for _ in range(100):
                    send_message(a, {"data": "x" * 10000})
        finally:
            a.close()

    def test_multiple_messages_in_sequence(self):
        """Multiple messages on the same socket round-trip correctly."""
        a, b = _socketpair()
        try:
            messages = [
                {"method": "search_code", "params": {"query": "first"}},
                {"method": "index_status"},
                {"result": [1, 2, 3]},
            ]
            for msg in messages:
                send_message(a, msg)
            for expected in messages:
                received = recv_message(b)
                assert received == expected
        finally:
            a.close()
            b.close()


# ---------------------------------------------------------------------------
# IPCServer dispatch tests
# ---------------------------------------------------------------------------


class TestIPCServerDispatch:
    """Test IPCServer._dispatch directly (without sockets)."""

    def _make_server(self, **kwargs: Any) -> IPCServer:
        mock_embecode = _make_embecode_server_mock(**kwargs)
        srv = IPCServer(Path("/tmp/test.sock"), mock_embecode)
        return srv

    def test_dispatch_search_code(self):
        results = [{"file_path": "a.py", "score": 0.9}]
        srv = self._make_server(search_results=results)
        response = srv._dispatch(
            {
                "method": "search_code",
                "params": {"query": "test", "mode": "hybrid", "top_k": 5},
            }
        )
        assert response == {"result": results}
        srv._server.search_code.assert_called_once_with(
            query="test",
            mode="hybrid",
            top_k=5,
            path=None,
        )

    def test_dispatch_search_code_default_params(self):
        srv = self._make_server()
        srv._dispatch({"method": "search_code", "params": {}})
        srv._server.search_code.assert_called_once_with(
            query="",
            mode="hybrid",
            top_k=10,
            path=None,
        )

    def test_dispatch_index_status(self):
        status = {"files_indexed": 42, "role": "owner"}
        srv = self._make_server(index_status=status)
        response = srv._dispatch({"method": "index_status"})
        assert response == {"result": status}
        srv._server.get_index_status.assert_called_once()

    def test_dispatch_unknown_method(self):
        srv = self._make_server()
        response = srv._dispatch({"method": "unknown_method"})
        assert "error" in response
        assert "Unknown IPC method" in response["error"]
        assert "unknown_method" in response["error"]

    def test_dispatch_no_method(self):
        srv = self._make_server()
        response = srv._dispatch({})
        assert "error" in response
        assert "Unknown IPC method" in response["error"]

    def test_dispatch_search_code_exception(self):
        srv = self._make_server()
        srv._server.search_code.side_effect = RuntimeError("DB error")
        response = srv._dispatch(
            {
                "method": "search_code",
                "params": {"query": "test"},
            }
        )
        assert "error" in response
        assert "DB error" in response["error"]

    def test_dispatch_index_status_exception(self):
        srv = self._make_server()
        srv._server.get_index_status.side_effect = RuntimeError("Status error")
        response = srv._dispatch({"method": "index_status"})
        assert "error" in response
        assert "Status error" in response["error"]


# ---------------------------------------------------------------------------
# IPCServer lifecycle tests (start/stop with real sockets)
# ---------------------------------------------------------------------------


class TestIPCServerLifecycle:
    """Integration-style tests for IPCServer start/stop with real Unix sockets."""

    def test_start_stop(self, short_tmp: Path):
        """Server starts, binds socket, and stops cleanly."""
        sock_path = short_tmp / "daemon.sock"
        mock_embecode = _make_embecode_server_mock()
        srv = IPCServer(sock_path, mock_embecode)

        srv.start()
        assert sock_path.exists()

        srv.stop()
        assert not sock_path.exists()

    def test_client_connects_and_searches(self, short_tmp: Path):
        """A client can connect and execute search_code via IPC."""
        sock_path = short_tmp / "daemon.sock"
        results = [{"file_path": "main.py", "score": 0.85}]
        mock_embecode = _make_embecode_server_mock(search_results=results)
        srv = IPCServer(sock_path, mock_embecode)
        srv.start()

        try:
            # Connect raw client socket and send request
            client_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client_sock.connect(str(sock_path))

            request = {
                "method": "search_code",
                "params": {"query": "hello", "mode": "hybrid", "top_k": 5},
            }
            send_message(client_sock, request)
            response = recv_message(client_sock)

            assert response is not None
            assert response["result"] == results
            client_sock.close()
        finally:
            srv.stop()

    def test_client_gets_index_status(self, short_tmp: Path):
        """A client can retrieve index_status via IPC."""
        sock_path = short_tmp / "daemon.sock"
        status = {"files_indexed": 99, "role": "owner"}
        mock_embecode = _make_embecode_server_mock(index_status=status)
        srv = IPCServer(sock_path, mock_embecode)
        srv.start()

        try:
            client_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client_sock.connect(str(sock_path))

            send_message(client_sock, {"method": "index_status"})
            response = recv_message(client_sock)

            assert response is not None
            assert response["result"] == status
            client_sock.close()
        finally:
            srv.stop()

    def test_multiple_sequential_requests(self, short_tmp: Path):
        """Multiple requests on the same connection all succeed."""
        sock_path = short_tmp / "daemon.sock"
        mock_embecode = _make_embecode_server_mock(
            search_results=[{"file_path": "a.py"}],
            index_status={"files_indexed": 5},
        )
        srv = IPCServer(sock_path, mock_embecode)
        srv.start()

        try:
            client_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client_sock.connect(str(sock_path))

            for _ in range(5):
                send_message(client_sock, {"method": "index_status"})
                resp = recv_message(client_sock)
                assert resp is not None
                assert "result" in resp

            client_sock.close()
        finally:
            srv.stop()

    def test_multiple_concurrent_clients(self, short_tmp: Path):
        """Multiple clients can connect and query concurrently."""
        sock_path = short_tmp / "daemon.sock"
        mock_embecode = _make_embecode_server_mock(
            search_results=[{"file_path": "b.py"}],
        )
        srv = IPCServer(sock_path, mock_embecode)
        srv.start()

        errors: list[str] = []
        barrier = threading.Barrier(3)

        def _client_worker(worker_id: int) -> None:
            try:
                client_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client_sock.connect(str(sock_path))
                barrier.wait(timeout=5)

                send_message(
                    client_sock,
                    {"method": "search_code", "params": {"query": f"q{worker_id}"}},
                )
                resp = recv_message(client_sock)
                if resp is None or "result" not in resp:
                    errors.append(f"Worker {worker_id}: bad response {resp}")
                client_sock.close()
            except Exception as exc:
                errors.append(f"Worker {worker_id}: {exc}")

        try:
            threads = [threading.Thread(target=_client_worker, args=(i,)) for i in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            assert errors == [], f"Client errors: {errors}"
        finally:
            srv.stop()

    def test_stale_socket_file_removed_on_start(self, short_tmp: Path):
        """If a stale daemon.sock exists, start() removes it and binds fresh."""
        sock_path = short_tmp / "daemon.sock"
        sock_path.write_text("stale")

        mock_embecode = _make_embecode_server_mock()
        srv = IPCServer(sock_path, mock_embecode)
        srv.start()

        try:
            # Should be a real socket now, not the stale file
            client_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client_sock.connect(str(sock_path))
            client_sock.close()
        finally:
            srv.stop()


# ---------------------------------------------------------------------------
# IPCClient tests
# ---------------------------------------------------------------------------


class TestIPCClient:
    """Tests for the IPCClient class."""

    def test_connect_and_close(self, short_tmp: Path):
        """Client connects and closes cleanly."""
        sock_path = short_tmp / "daemon.sock"
        mock_embecode = _make_embecode_server_mock()
        srv = IPCServer(sock_path, mock_embecode)
        srv.start()

        try:
            client = IPCClient(sock_path)
            assert not client.is_connected

            client.connect()
            assert client.is_connected

            client.close()
            assert not client.is_connected
        finally:
            srv.stop()

    def test_connect_to_nonexistent_socket(self, tmp_path: Path):
        """Connecting to a nonexistent socket raises ConnectionError."""
        client = IPCClient(tmp_path / "nonexistent.sock")
        with pytest.raises(ConnectionError, match="Cannot connect"):
            client.connect()

    def test_search_code(self, short_tmp: Path):
        """search_code proxies correctly and returns results."""
        sock_path = short_tmp / "daemon.sock"
        results = [{"file_path": "x.py", "score": 0.7}]
        mock_embecode = _make_embecode_server_mock(search_results=results)
        srv = IPCServer(sock_path, mock_embecode)
        srv.start()

        try:
            client = IPCClient(sock_path)
            client.connect()

            got = client.search_code(query="test", mode="keyword", top_k=3, path="src/")
            assert got == results

            mock_embecode.search_code.assert_called_once_with(
                query="test",
                mode="keyword",
                top_k=3,
                path="src/",
            )

            client.close()
        finally:
            srv.stop()

    def test_index_status(self, short_tmp: Path):
        """index_status proxies correctly and returns status."""
        sock_path = short_tmp / "daemon.sock"
        status = {"files_indexed": 77, "role": "owner"}
        mock_embecode = _make_embecode_server_mock(index_status=status)
        srv = IPCServer(sock_path, mock_embecode)
        srv.start()

        try:
            client = IPCClient(sock_path)
            client.connect()

            got = client.index_status()
            assert got == status

            client.close()
        finally:
            srv.stop()

    def test_search_code_owner_error(self, short_tmp: Path):
        """search_code raises RuntimeError when owner returns an error."""
        sock_path = short_tmp / "daemon.sock"
        mock_embecode = _make_embecode_server_mock()
        mock_embecode.search_code.side_effect = RuntimeError("Broken")
        srv = IPCServer(sock_path, mock_embecode)
        srv.start()

        try:
            client = IPCClient(sock_path)
            client.connect()

            with pytest.raises(RuntimeError, match="Broken"):
                client.search_code(query="test")

            client.close()
        finally:
            srv.stop()

    def test_connection_drop_raises(self, short_tmp: Path):
        """When the owner stops, the client raises ConnectionError."""
        sock_path = short_tmp / "daemon.sock"
        mock_embecode = _make_embecode_server_mock()
        srv = IPCServer(sock_path, mock_embecode)
        srv.start()

        client = IPCClient(sock_path)
        client.connect()
        assert client.is_connected

        # Stop the server (closes all connections)
        srv.stop()

        # Give the OS a moment to propagate the close
        time.sleep(0.1)

        with pytest.raises(ConnectionError):
            client.search_code(query="test")

    def test_request_without_connect(self):
        """Calling search_code before connect raises ConnectionError."""
        client = IPCClient(Path("/tmp/nonexistent.sock"))
        with pytest.raises(ConnectionError, match="not connected"):
            client.search_code(query="test")

    def test_double_close_is_safe(self, short_tmp: Path):
        """Calling close() twice does not raise."""
        sock_path = short_tmp / "daemon.sock"
        mock_embecode = _make_embecode_server_mock()
        srv = IPCServer(sock_path, mock_embecode)
        srv.start()

        try:
            client = IPCClient(sock_path)
            client.connect()
            client.close()
            client.close()  # Should not raise
            assert not client.is_connected
        finally:
            srv.stop()
