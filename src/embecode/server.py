"""FastMCP server exposing code search and index status tools."""

from __future__ import annotations

import logging
import sys
import threading
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
    """

    def __init__(self, project_path: Path) -> None:
        """
        Initialize embecode server.

        Args:
            project_path: Path to the project root directory to index.
        """
        self.project_path = project_path.resolve()
        self.config = load_config(self.project_path)

        # Initialize cache manager and get cache directory
        self.cache_manager = CacheManager()
        self.cache_dir = self.cache_manager.get_cache_dir(self.project_path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database
        db_path = self.cache_dir / "index.db"
        self.db = Database(db_path)
        self.db.connect()

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

        # Embedding model change detection
        stored_model = self.db.get_metadata("embedding_model")
        configured_model = self.config.embeddings.model
        if stored_model is None:
            # First run or fresh DB: store the current model
            self.db.set_metadata("embedding_model", configured_model)
        elif stored_model != configured_model:
            db_path = self.cache_dir / "index.db"
            raise EmbeddingModelChangedError(
                f"Embedding model changed from '{stored_model}' to '{configured_model}'. "
                f"Existing embeddings are incompatible. Delete the index at {db_path} "
                f"and restart, or revert the model in your config."
            )

        # Always spawn background catch-up thread
        logger.info("Starting catch-up index in background")
        threading.Thread(target=self._catchup_index, daemon=True).start()

        # Update cache access time
        self.cache_manager.update_access_time(self.project_path)

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

    def search_code(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5,
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
            List of chunk result dictionaries with content, file_path, language,
            start_line, end_line, context, and score.

        Raises:
            IndexNotReadyError: If index is still being built.
        """
        try:
            results = self.searcher.search(query, mode=mode, top_k=top_k, path=path)
            return [result.to_dict() for result in results]
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
            last_updated, is_indexing, current_file, and progress.
        """
        status = self.indexer.get_status()
        return status.to_dict()

    def cleanup(self) -> None:
        """Clean up resources when server shuts down."""
        if self.watcher is not None:
            self.watcher.stop()
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
    top_k: int = 5,
    path: str | None = None,
) -> list[dict[str, Any]]:
    """
    Search the codebase using keyword, semantic, or hybrid search.

    Args:
        query: Search query string (natural language or code).
        mode: Search mode - "semantic" for vector search, "keyword" for BM25,
              or "hybrid" for RRF fusion of both (default).
        top_k: Number of results to return (default: 5).
        path: Optional path prefix filter (e.g., "src/", "apps/ui/").

    Returns:
        List of chunk results with content, file_path, language, start_line,
        end_line, context, and relevance score.
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
