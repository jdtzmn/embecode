"""IPC server and client for multi-process lock-file daemon protocol.

The owner process runs an :class:`IPCServer` on a Unix domain socket
(``daemon.sock``) so that reader processes can proxy ``search_code`` and
``index_status`` requests without opening their own DuckDB connection.

Message framing
---------------

Each message (request or response) is framed as::

    [4 bytes: payload length, big-endian unsigned int][UTF-8 JSON payload]

Length-prefixed framing avoids delimiter issues with JSON content and allows
efficient reads.
"""

from __future__ import annotations

import json
import logging
import socket
import struct
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from embecode.server import EmbeCodeServer

logger = logging.getLogger(__name__)

# Maximum message size: 64 MiB (generous upper bound for large search results)
_MAX_MESSAGE_SIZE = 64 * 1024 * 1024

# Header format: 4-byte unsigned big-endian integer
_HEADER_FORMAT = "!I"
_HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)


# ---------------------------------------------------------------------------
# Message framing helpers
# ---------------------------------------------------------------------------


def send_message(sock: socket.socket, data: dict[str, Any]) -> None:
    """Send a length-prefixed JSON message over a socket.

    Args:
        sock: Connected socket to send on.
        data: JSON-serialisable dictionary to send.

    Raises:
        ConnectionError: If the socket is closed or broken.
    """
    payload = json.dumps(data).encode("utf-8")
    header = struct.pack(_HEADER_FORMAT, len(payload))
    try:
        sock.sendall(header + payload)
    except (BrokenPipeError, ConnectionResetError, OSError) as exc:
        raise ConnectionError(f"Failed to send IPC message: {exc}") from exc


def recv_message(sock: socket.socket) -> dict[str, Any] | None:
    """Receive a length-prefixed JSON message from a socket.

    Args:
        sock: Connected socket to read from.

    Returns:
        Parsed JSON dictionary, or ``None`` if the peer closed the connection
        cleanly (zero-length read).

    Raises:
        ConnectionError: If the connection is reset unexpectedly.
        ValueError: If the payload exceeds :data:`_MAX_MESSAGE_SIZE` or is not
            valid JSON.
    """
    # Read the 4-byte length header
    header = _recv_exact(sock, _HEADER_SIZE)
    if header is None:
        return None  # clean disconnect

    (length,) = struct.unpack(_HEADER_FORMAT, header)
    if length > _MAX_MESSAGE_SIZE:
        raise ValueError(f"IPC message too large: {length} bytes (max {_MAX_MESSAGE_SIZE})")

    # Read the JSON payload
    payload = _recv_exact(sock, length)
    if payload is None:
        raise ConnectionError("Connection closed while reading IPC payload")

    try:
        return json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid IPC message payload: {exc}") from exc


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    """Read exactly *n* bytes from *sock*.

    Returns:
        The bytes read, or ``None`` if the peer closed the connection before
        any data was read (EOF at the start).

    Raises:
        ConnectionError: If the connection is reset mid-read.
    """
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            raise ConnectionError(f"IPC recv error: {exc}") from exc
        if not chunk:
            if len(buf) == 0:
                return None  # clean EOF before any data
            raise ConnectionError(f"Connection closed mid-read (got {len(buf)}/{n} bytes)")
        buf.extend(chunk)
    return bytes(buf)


# ---------------------------------------------------------------------------
# IPC Server (owner process)
# ---------------------------------------------------------------------------


class IPCServer:
    """Unix domain socket server for the owner process.

    Binds ``daemon.sock``, accepts connections in a listener thread, and
    spawns a handler thread per connection.  Incoming ``search_code`` and
    ``index_status`` requests are dispatched to the :class:`EmbeCodeServer`
    instance.

    Usage::

        server = IPCServer(socket_path, embecode_server)
        server.start()
        # ... later ...
        server.stop()
    """

    def __init__(self, socket_path: Path, embecode_server: EmbeCodeServer) -> None:
        """Initialise the IPC server.

        Args:
            socket_path: Path to the Unix domain socket file.
            embecode_server: The owner's :class:`EmbeCodeServer` instance
                that handles ``search_code`` and ``index_status`` calls.
        """
        self._socket_path = socket_path
        self._server = embecode_server
        self._sock: socket.socket | None = None
        self._listener_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._handler_threads: list[threading.Thread] = []
        self._handler_lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Bind the socket and start the listener thread."""
        # Remove stale socket file if present
        try:
            self._socket_path.unlink()
        except FileNotFoundError:
            pass

        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(str(self._socket_path))
        self._sock.listen(8)
        # Use a short timeout so the listener thread can check _stop_event
        self._sock.settimeout(0.5)

        self._stop_event.clear()
        self._listener_thread = threading.Thread(
            target=self._accept_loop,
            daemon=True,
            name="IPCServer-listener",
        )
        self._listener_thread.start()
        logger.info("IPC server listening on %s", self._socket_path)

    def stop(self) -> None:
        """Stop accepting connections and close existing ones."""
        self._stop_event.set()

        # Close the listening socket to unblock accept()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

        # Wait for the listener thread
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=3.0)
            self._listener_thread = None

        # Wait for handler threads
        with self._handler_lock:
            threads = list(self._handler_threads)
        for t in threads:
            t.join(timeout=2.0)
        with self._handler_lock:
            self._handler_threads.clear()

        # Remove socket file
        try:
            self._socket_path.unlink()
        except FileNotFoundError:
            pass

        logger.info("IPC server stopped")

    # -- internal ------------------------------------------------------------

    def _accept_loop(self) -> None:
        """Accept incoming connections until stopped."""
        while not self._stop_event.is_set():
            try:
                conn, _ = self._sock.accept()  # type: ignore[union-attr]
            except TimeoutError:
                continue
            except OSError:
                # Socket closed by stop()
                break

            t = threading.Thread(
                target=self._handle_connection,
                args=(conn,),
                daemon=True,
                name="IPCServer-handler",
            )
            with self._handler_lock:
                self._handler_threads.append(t)
            t.start()

    def _handle_connection(self, conn: socket.socket) -> None:
        """Handle a single reader connection: read requests, dispatch, reply."""
        try:
            while not self._stop_event.is_set():
                request = recv_message(conn)
                if request is None:
                    # Reader disconnected cleanly
                    break

                response = self._dispatch(request)
                send_message(conn, response)
        except (ConnectionError, ValueError) as exc:
            logger.debug("IPC handler connection ended: %s", exc)
        except Exception:
            logger.exception("Unexpected error in IPC handler")
        finally:
            try:
                conn.close()
            except OSError:
                pass
            # Remove ourselves from the handler list
            current = threading.current_thread()
            with self._handler_lock:
                self._handler_threads = [t for t in self._handler_threads if t is not current]

    def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        """Dispatch an IPC request to the appropriate server method.

        Args:
            request: Parsed JSON request with ``"method"`` and optional
                ``"params"`` keys.

        Returns:
            Response dictionary with either ``"result"`` or ``"error"``.
        """
        method = request.get("method")
        params = request.get("params", {})

        if method == "search_code":
            return self._handle_search_code(params)
        elif method == "index_status":
            return self._handle_index_status()
        else:
            return {"error": f"Unknown IPC method: {method!r}"}

    def _handle_search_code(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle a ``search_code`` request."""
        try:
            results = self._server.search_code(
                query=params.get("query", ""),
                mode=params.get("mode", "hybrid"),
                top_k=params.get("top_k", 10),
                path=params.get("path"),
            )
            return {"result": results}
        except Exception as exc:
            logger.exception("IPC search_code error")
            return {"error": str(exc)}

    def _handle_index_status(self) -> dict[str, Any]:
        """Handle an ``index_status`` request."""
        try:
            status = self._server.get_index_status()
            return {"result": status}
        except Exception as exc:
            logger.exception("IPC index_status error")
            return {"error": str(exc)}


# ---------------------------------------------------------------------------
# IPC Client (reader process)
# ---------------------------------------------------------------------------


class IPCClient:
    """Unix domain socket client for reader processes.

    Maintains a single persistent connection to the owner's IPC server.
    Provides :meth:`search_code` and :meth:`index_status` that mirror the
    local :class:`EmbeCodeServer` API.

    Usage::

        client = IPCClient(socket_path)
        client.connect()
        results = client.search_code(query="hello")
        client.close()
    """

    def __init__(self, socket_path: Path) -> None:
        """Initialise the IPC client.

        Args:
            socket_path: Path to the owner's Unix domain socket.
        """
        self._socket_path = socket_path
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        """Open a persistent connection to the owner's IPC socket.

        Raises:
            ConnectionError: If the socket cannot be reached.
        """
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(str(self._socket_path))
            self._sock = sock
            logger.info("IPC client connected to %s", self._socket_path)
        except (ConnectionRefusedError, FileNotFoundError, OSError) as exc:
            raise ConnectionError(
                f"Cannot connect to owner IPC socket at {self._socket_path}: {exc}"
            ) from exc

    def close(self) -> None:
        """Close the connection to the owner."""
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None

    @property
    def is_connected(self) -> bool:
        """Return True if the client socket is open."""
        return self._sock is not None

    def search_code(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 10,
        path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Proxy a search_code request to the owner via IPC.

        Args:
            query: Search query string.
            mode: Search mode (``"semantic"``, ``"keyword"``, or ``"hybrid"``).
            top_k: Number of results to return.
            path: Optional path prefix filter.

        Returns:
            List of search result dictionaries (same shape as the owner's
            local ``search_code``).

        Raises:
            ConnectionError: If the IPC connection is broken.
            RuntimeError: If the owner returns an error response.
        """
        response = self._request(
            "search_code",
            {
                "query": query,
                "mode": mode,
                "top_k": top_k,
                "path": path,
            },
        )
        if "error" in response:
            raise RuntimeError(f"Owner search_code error: {response['error']}")
        return response["result"]

    def index_status(self) -> dict[str, Any]:
        """Proxy an index_status request to the owner via IPC.

        Returns:
            Index status dictionary (same shape as the owner's local
            ``get_index_status``).

        Raises:
            ConnectionError: If the IPC connection is broken.
            RuntimeError: If the owner returns an error response.
        """
        response = self._request("index_status")
        if "error" in response:
            raise RuntimeError(f"Owner index_status error: {response['error']}")
        return response["result"]

    def _request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a request and read the response.

        Args:
            method: IPC method name.
            params: Optional parameters dictionary.

        Returns:
            Parsed response dictionary.

        Raises:
            ConnectionError: If the socket is disconnected.
        """
        with self._lock:
            if self._sock is None:
                raise ConnectionError("IPC client is not connected")

            request: dict[str, Any] = {"method": method}
            if params is not None:
                request["params"] = params

            send_message(self._sock, request)
            response = recv_message(self._sock)

            if response is None:
                # Owner closed the connection
                self._sock = None
                raise ConnectionError("Owner closed IPC connection")

            return response
