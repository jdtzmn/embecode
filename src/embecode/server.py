"""FastMCP server exposing code search and index status tools."""

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from embecode.cache import CacheManager
from embecode.config import (  # noqa: F401 (EmbeCodeConfig needed for test mocking)
    EmbeCodeConfig,
    load_config,
)
from embecode.db import Database
from embecode.embedder import Embedder
from embecode.indexer import Indexer
from embecode.searcher import IndexNotReadyError, Searcher
from embecode.watcher import Watcher

logger = logging.getLogger(__name__)


def is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running.

    Uses ``os.kill(pid, 0)`` — sending signal 0 does not kill the process but
    raises ``ProcessLookupError`` if the PID does not exist.

    Args:
        pid: Process ID to check.

    Returns:
        True if the process is alive (or we cannot signal it due to permissions),
        False if no such process exists.
    """
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we don't have permission to signal it
        return True


def _cleanup_lock(lock_path: Path, pid: int) -> None:
    """Remove the daemon lock file if we are still the owner.

    This is registered as an ``atexit`` handler on the owner process so that
    readers can promote when the owner exits cleanly.  Safe to call multiple
    times — errors are silently suppressed.

    Args:
        lock_path: Path to the daemon.lock file.
        pid: PID of the owner process (us).
    """
    try:
        with open(lock_path) as f:
            data = json.load(f)
        if data.get("pid") == pid:
            os.unlink(lock_path)
    except (OSError, json.JSONDecodeError):
        pass


class EmbeddingModelChangedError(RuntimeError):
    """Raised when the configured embedding model differs from the one used to build the index."""


# Create FastMCP server instance
mcp = FastMCP(
    name="embecode",
)


class EmbeCodeServer:
    """
    MCP server for embecode.

    Orchestrates full indexing, file watching, and provides search and status tools.

    On startup the server attempts to acquire ``daemon.lock`` atomically.  The
    first process to succeed becomes the **owner** (read-write DB, indexer,
    file watcher).  All other processes become **readers** (read-only DB, lock
    file watcher for promotion).
    """

    def __init__(self, project_path: Path) -> None:
        """
        Initialize embecode server.

        Determines whether this process is the owner or a reader based on
        atomic lock file acquisition.

        Args:
            project_path: Path to the project root directory to index.
        """
        self.project_path = project_path.resolve()
        self.config = load_config(self.project_path)

        # Initialize cache manager and get cache directory
        self.cache_manager = CacheManager()
        self.cache_dir = self.cache_manager.get_cache_dir(self.project_path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Lock file path
        self.lock_path = self.cache_manager.get_lock_path(self.project_path)

        # Initialize database (connection opened after role is determined)
        db_path = self.cache_dir / "index.db"
        self.db = Database(db_path)

        # Initialize embedder and searcher
        self.embedder = Embedder(config=self.config.embeddings)
        self.searcher = Searcher(self.db, self.embedder)

        # Initialize indexer
        self.indexer = Indexer(
            project_path=self.project_path,
            config=self.config,
            db=self.db,
            embedder=self.embedder,
        )

        # Initialize watcher (will be started after catch-up indexing)
        self.watcher: Watcher | None = None

        # Lock file watcher thread (reader only)
        self._lock_watcher_stop = threading.Event()
        self._lock_watcher_thread: threading.Thread | None = None

        # Determine role via atomic lock acquisition
        self._role = self._acquire_lock()

        if self._role == "owner":
            self._setup_owner()
        else:
            self._setup_reader()

        # Update cache access time
        self.cache_manager.update_access_time(self.project_path)

    # ------------------------------------------------------------------
    # Lock acquisition helpers
    # ------------------------------------------------------------------

    def _acquire_lock(self) -> str:
        """Attempt atomic lock file creation to determine owner vs reader.

        Returns:
            ``"owner"`` if this process successfully created the lock file,
            ``"reader"`` if another live process owns it.
        """
        while True:
            try:
                fd = os.open(
                    str(self.lock_path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                # We created the lock file — we are the owner
                with os.fdopen(fd, "w") as f:
                    json.dump({"pid": os.getpid()}, f)
                logger.info("Acquired daemon lock — this process is the OWNER")
                return "owner"
            except FileExistsError:
                # Lock file exists; check if the owner is still alive
                owner_pid = self._read_lock_pid()
                if owner_pid is not None and is_pid_alive(owner_pid):
                    logger.info("Daemon lock held by PID %d — this process is a READER", owner_pid)
                    return "reader"
                else:
                    # Stale lock — remove it and retry
                    logger.info(
                        "Stale daemon lock (PID %s dead) — removing and retrying",
                        owner_pid,
                    )
                    try:
                        os.unlink(str(self.lock_path))
                    except OSError:
                        pass
                    # Loop back to retry atomic creation

    def _read_lock_pid(self) -> int | None:
        """Read the PID stored in the lock file.

        Returns:
            The PID as an integer, or None if the file cannot be read.
        """
        try:
            with open(self.lock_path) as f:
                data = json.load(f)
            return int(data["pid"])
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Owner setup
    # ------------------------------------------------------------------

    def _setup_owner(self) -> None:
        """Set up the owner role: open DB read-write, register cleanup, start indexer."""
        # Register atexit and signal handlers so the lock is removed on exit
        atexit.register(_cleanup_lock, self.lock_path, os.getpid())
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: sys.exit(0))  # triggers atexit

        # Open DB read-write and check embedding model compatibility
        self.db.connect(read_only=False)
        self._check_embedding_model()

        # Spawn background catch-up thread
        logger.info("Starting catch-up index in background (owner)")
        threading.Thread(target=self._catchup_index, daemon=True).start()

    def _check_embedding_model(self) -> None:
        """Validate that the configured embedding model matches the stored one."""
        stored_model = self.db.get_metadata("embedding_model")
        configured_model = self.config.embeddings.model
        if stored_model is None:
            self.db.set_metadata("embedding_model", configured_model)
        elif stored_model != configured_model:
            db_path = self.cache_dir / "index.db"
            raise EmbeddingModelChangedError(
                f"Embedding model changed from '{stored_model}' to '{configured_model}'. "
                f"Existing embeddings are incompatible. Delete the index at {db_path} "
                f"and restart, or revert the model in your config."
            )

    # ------------------------------------------------------------------
    # Reader setup
    # ------------------------------------------------------------------

    def _setup_reader(self) -> None:
        """Set up the reader role: open DB read-only, start lock file watcher."""
        self.db.connect(read_only=True)
        logger.info("Reader: connected to index in read-only mode")

        # Start watching the lock file so we can promote when the owner exits
        self._lock_watcher_stop.clear()
        self._lock_watcher_thread = threading.Thread(
            target=self._watch_lock_file,
            daemon=True,
            name="LockFileWatcher",
        )
        self._lock_watcher_thread.start()

    # ------------------------------------------------------------------
    # Catch-up and file watcher (owner only)
    # ------------------------------------------------------------------

    def _catchup_index(self) -> None:
        """
        Perform catch-up indexing in background thread.

        Detects and indexes missing/modified files, removes stale entries.
        After completion, starts the file watcher if enabled.
        """
        try:
            self.indexer.start_catchup_index(background=False)
        except Exception:
            logger.exception("Catch-up index failed")
        finally:
            if self.config.daemon.auto_watch:
                self._start_watcher()

    def _start_watcher(self) -> None:
        """Start the file watcher if not already running."""
        if self.watcher is None:
            logger.info("Starting file watcher")
            self.watcher = Watcher(
                project_path=self.project_path,
                config=self.config,
                indexer=self.indexer,
            )
            self.watcher.start()

    # ------------------------------------------------------------------
    # Lock file watcher (reader only)
    # ------------------------------------------------------------------

    def _watch_lock_file(self) -> None:
        """Watch the cache directory for ``daemon.lock`` deletion.

        Runs in a background daemon thread (reader only).  When the lock file
        disappears (owner exited cleanly) or is found stale (owner crashed),
        this method attempts to promote this process to owner.
        """
        try:
            from watchfiles import watch as wf_watch
        except ImportError:
            logger.error("watchfiles not available — reader cannot watch lock file")
            return

        logger.info("Reader: watching lock file for owner exit")
        try:
            for _changes in wf_watch(
                str(self.cache_dir),
                stop_event=self._lock_watcher_stop,
                recursive=False,
            ):
                if self._lock_watcher_stop.is_set():
                    break

                # Check whether the lock file is gone or stale
                if not self.lock_path.exists():
                    logger.info("Lock file removed — attempting promotion to owner")
                    self._promote_to_owner()
                    break
                else:
                    owner_pid = self._read_lock_pid()
                    if owner_pid is not None and not is_pid_alive(owner_pid):
                        logger.info(
                            "Lock file has stale PID %d — attempting promotion to owner",
                            owner_pid,
                        )
                        try:
                            os.unlink(str(self.lock_path))
                        except OSError:
                            pass
                        self._promote_to_owner()
                        break
        except Exception:
            logger.exception("Lock file watcher error")

    def _promote_to_owner(self) -> None:
        """Promote this process from reader to owner.

        Steps:
        1. Attempt atomic lock file creation (another reader may win the race).
        2. Stop the lock file watcher.
        3. Close read-only DB connection and reopen read-write.
        4. Run catch-up indexing and start file watcher.
        """
        # Try to atomically claim the lock
        try:
            fd = os.open(
                str(self.lock_path),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
            with os.fdopen(fd, "w") as f:
                json.dump({"pid": os.getpid()}, f)
        except FileExistsError:
            # Another reader won the race — stay as reader, keep watching
            logger.info("Another process claimed ownership — staying as reader")
            self._setup_reader()
            return

        logger.info("Promoted to OWNER")
        self._role = "owner"

        # Stop the lock file watcher (no longer needed)
        self._lock_watcher_stop.set()

        # Register cleanup for our new lock
        atexit.register(_cleanup_lock, self.lock_path, os.getpid())
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: sys.exit(0))

        # Transition DB from read-only to read-write
        self.db.reconnect(read_only=False)
        self._check_embedding_model()

        # Run catch-up indexing then start the file watcher
        threading.Thread(target=self._catchup_index, daemon=True).start()

    # ------------------------------------------------------------------
    # MCP tool implementations
    # ------------------------------------------------------------------

    def search_code(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 10,
        path: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search the codebase using keyword, semantic, or hybrid search.

        Args:
            query: Search query string (natural language or code).
            mode: Search mode - "semantic", "keyword", or "hybrid".
            top_k: Number of results to return.
            path: Optional path prefix filter (e.g., "src/", "apps/ui/").

        Returns:
            List of concise chunk result dictionaries with file_path, language,
            start_line, end_line, definitions, preview, and score.

        Raises:
            IndexNotReadyError: If index is still being built.
        """
        # Mid-reconnect guard (promotion in progress)
        if self.db._conn is None:
            return [
                {
                    "error": "Server is reconnecting to the index. Try again in a moment.",
                    "retry_recommended": True,
                }
            ]

        try:
            response = self.searcher.search(query, mode=mode, top_k=top_k, path=path)
            results = [result.to_dict(query=query) for result in response.results]

            # Add file grouping hint when multiple results share a file
            file_counts = Counter(r["file_path"] for r in results)
            for r in results:
                count = file_counts[r["file_path"]]
                if count > 1:
                    r["file_result_count"] = count

            return results
        except IndexNotReadyError as e:
            # Get progress if indexing is in progress
            status = self.indexer.get_status()
            if status.is_indexing:
                progress_pct = int((status.progress or 0) * 100)
                msg = (
                    f"Index is still being built ({status.files_indexed} files processed, "
                    f"{progress_pct}% complete). Try again in ~30s."
                )
                raise IndexNotReadyError(msg) from e
            else:
                raise

    def get_index_status(self) -> dict[str, Any]:
        """
        Get current index status.

        Returns:
            Dictionary with files_indexed, total_chunks, embedding_model,
            last_updated, is_indexing, current_file, progress, and role.
        """
        status = self.indexer.get_status()
        result = status.to_dict()
        result["role"] = self._role
        return result

    def cleanup(self) -> None:
        """Clean up resources when server shuts down."""
        if self.watcher is not None:
            self.watcher.stop()

        # Stop the lock file watcher (reader)
        if self._lock_watcher_thread is not None and self._lock_watcher_thread.is_alive():
            self._lock_watcher_stop.set()
            self._lock_watcher_thread.join(timeout=3.0)

        # Remove lock file if we are the owner
        if self._role == "owner":
            _cleanup_lock(self.lock_path, os.getpid())

        self.db.close()


# Global server instance
_server: EmbeCodeServer | None = None


def get_server() -> EmbeCodeServer:
    """Get or create the global server instance."""
    global _server
    if _server is None:
        msg = "Server not initialized. Call initialize_server() first."
        raise RuntimeError(msg)
    return _server


def initialize_server(project_path: Path) -> EmbeCodeServer:
    """
    Initialize the global server instance.

    Args:
        project_path: Path to the project root directory.

    Returns:
        The initialized server instance.
    """
    global _server
    if _server is not None:
        logger.warning("Server already initialized, returning existing instance")
        return _server

    _server = EmbeCodeServer(project_path)
    return _server


@mcp.tool()
def search_code(
    query: str,
    mode: str = "hybrid",
    top_k: int = 10,
    path: str | None = None,
) -> list[dict[str, Any]]:
    """
    Search the codebase using keyword, semantic, or hybrid search.

    Args:
        query: Search query string (natural language or code).
        mode: Search mode - "semantic" for vector search, "keyword" for BM25,
              or "hybrid" for RRF fusion of both (default).
        top_k: Number of results to return (default: 10).
        path: Optional path prefix filter (e.g., "src/", "apps/ui/").

    Returns:
        List of concise chunk results with file_path, language, start_line,
        end_line, definitions, preview, and relevance score.
    """
    server = get_server()
    try:
        return server.search_code(query, mode=mode, top_k=top_k, path=path)
    except IndexNotReadyError as e:
        # Return error message in structured format
        return [{"error": str(e), "retry_recommended": True}]


@mcp.tool()
def index_status() -> dict[str, Any]:
    """
    Get current index status.

    Returns:
        Dictionary with:
        - files_indexed: Number of files that have been indexed
        - total_chunks: Total number of chunks in the index
        - embedding_model: Name of the embedding model in use
        - last_updated: ISO timestamp of last index update
        - is_indexing: Whether indexing is currently in progress
        - current_file: Current file being indexed (if is_indexing=True)
        - progress: Progress as a fraction 0-1 (if is_indexing=True)
    """
    server = get_server()
    return server.get_index_status()


def run_server(project_path: Path) -> None:
    """
    Initialize and run the MCP server.

    Args:
        project_path: Path to the project root directory to index.
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    logger.info(f"Starting embecode server for project: {project_path}")

    # Initialize server
    try:
        initialize_server(project_path)
    except Exception:
        logger.exception("Failed to initialize server")
        sys.exit(1)

    # Run FastMCP server
    try:
        logger.info("MCP server ready")
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception:
        logger.exception("Server error")
        sys.exit(1)
    finally:
        # Cleanup
        server = get_server()
        server.cleanup()
        logger.info("Server shutdown complete")
