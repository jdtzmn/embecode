"""Index orchestration - full and incremental indexing with background threading."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pathspec

if TYPE_CHECKING:
    from embecode.chunker import Chunk
    from embecode.config import EmbeCodeConfig
    from embecode.db import Database
    from embecode.embedder import Embedder

logger = logging.getLogger(__name__)


class IndexingError(Exception):
    """Base exception for indexing errors."""


class IndexingInProgressError(IndexingError):
    """Raised when attempting to start indexing while already in progress."""


class IndexStatus:
    """Status information about the index."""

    def __init__(
        self,
        files_indexed: int,
        total_chunks: int,
        embedding_model: str,
        last_updated: str | None,
        is_indexing: bool,
        current_file: str | None = None,
        progress: float | None = None,
    ) -> None:
        """
        Initialize index status.

        Args:
            files_indexed: Number of files that have been indexed.
            total_chunks: Total number of chunks in the index.
            embedding_model: Name of the embedding model in use.
            last_updated: ISO timestamp of last index update.
            is_indexing: Whether indexing is currently in progress.
            current_file: Current file being indexed (if is_indexing=True).
            progress: Progress as a fraction 0-1 (if is_indexing=True).
        """
        self.files_indexed = files_indexed
        self.total_chunks = total_chunks
        self.embedding_model = embedding_model
        self.last_updated = last_updated
        self.is_indexing = is_indexing
        self.current_file = current_file
        self.progress = progress

    def to_dict(self) -> dict[str, Any]:
        """Convert status to dictionary for API responses."""
        return {
            "files_indexed": self.files_indexed,
            "total_chunks": self.total_chunks,
            "embedding_model": self.embedding_model,
            "last_updated": self.last_updated,
            "is_indexing": self.is_indexing,
            "current_file": self.current_file,
            "progress": self.progress,
        }


class Indexer:
    """
    Orchestrates full and incremental indexing of a codebase.

    Supports background threading for non-blocking operation.
    Tracks file hashes and chunk hashes for efficient incremental updates.
    """

    def __init__(
        self,
        project_path: Path,
        config: EmbeCodeConfig,
        db: Database,
        embedder: Embedder,
    ) -> None:
        """
        Initialize indexer.

        Args:
            project_path: Path to the project root directory.
            config: Configuration with include/exclude rules and chunk sizes.
            db: Database interface for storing chunks and embeddings.
            embedder: Embedder for generating embeddings.
        """
        self.project_path = project_path
        self.config = config
        self.db = db
        self.embedder = embedder
        self._indexing_thread: threading.Thread | None = None
        self._is_indexing = False
        self._current_file: str | None = None
        self._progress: float | None = None
        self._lock = threading.Lock()
        # Cache for .gitignore PathSpec objects keyed by directory path
        self._gitignore_cache: dict[Path, pathspec.PathSpec | None] = {}

    @property
    def is_indexing(self) -> bool:
        """Check if indexing is currently in progress."""
        with self._lock:
            return self._is_indexing

    def get_status(self) -> IndexStatus:
        """
        Get current index status.

        Returns:
            IndexStatus with current state and statistics.
        """
        with self._lock:
            is_indexing = self._is_indexing
            current_file = self._current_file
            progress = self._progress

        # Get stats from database
        stats = self.db.get_index_stats()

        return IndexStatus(
            files_indexed=stats["files_indexed"],
            total_chunks=stats["total_chunks"],
            embedding_model=self.config.embeddings.model,
            last_updated=stats["last_updated"],
            is_indexing=is_indexing,
            current_file=current_file,
            progress=progress,
        )

    def start_full_index(self, background: bool = True) -> None:
        """
        Start a full index of the codebase.

        Args:
            background: If True, run in background thread. If False, block until complete.

        Raises:
            IndexingInProgressError: If indexing is already in progress.
        """
        with self._lock:
            if self._is_indexing:
                raise IndexingInProgressError("Indexing is already in progress")
            self._is_indexing = True

        if background:
            self._indexing_thread = threading.Thread(
                target=self._run_full_index,
                daemon=True,
            )
            self._indexing_thread.start()
        else:
            try:
                self._run_full_index()
            finally:
                with self._lock:
                    self._is_indexing = False

    def _run_full_index(self) -> None:
        """Run full indexing (called from thread or directly)."""
        try:
            logger.info("Starting full index of %s", self.project_path)

            # Clear existing index
            self.db.clear_index()

            # Walk file tree and collect files to index
            files = self._collect_files()
            total_files = len(files)
            logger.info("Found %d files to index", total_files)

            # Import chunker here to avoid circular imports
            from embecode.chunker import chunk_file

            # Process each file
            indexed_count = 0
            for i, file_path in enumerate(files):
                with self._lock:
                    self._current_file = str(file_path)
                    self._progress = i / total_files if total_files > 0 else 0.0

                try:
                    # Chunk the file
                    chunks = chunk_file(file_path, self.config.index.languages)

                    if not chunks:
                        logger.debug("No chunks for %s, skipping", file_path)
                        continue

                    # Generate embeddings for all chunks
                    # Combine context + content for richer embeddings
                    chunk_texts = [
                        f"{chunk.context}\n{chunk.content}" if chunk.context else chunk.content
                        for chunk in chunks
                    ]
                    embeddings = self.embedder.embed(chunk_texts)

                    # Store chunks and embeddings in database
                    self._store_chunks(file_path, chunks, embeddings)

                    indexed_count += 1
                    if indexed_count % 10 == 0:
                        logger.info("Indexed %d/%d files", indexed_count, total_files)

                except Exception as e:
                    logger.warning("Failed to index %s: %s", file_path, e)
                    continue

            logger.info("Full index complete: %d files indexed", indexed_count)

        except Exception as e:
            logger.error("Full index failed: %s", e)
            raise

        finally:
            with self._lock:
                self._is_indexing = False
                self._current_file = None
                self._progress = None

    def update_file(self, file_path: Path) -> None:
        """
        Incrementally update a single file in the index.

        Compares chunk hashes and only updates changed chunks.

        Args:
            file_path: Path to the file that changed.
        """
        if not self._should_index_file(file_path):
            logger.debug("File %s excluded by config, skipping update", file_path)
            return

        try:
            # Import chunker here to avoid circular imports
            from embecode.chunker import chunk_file

            # Get new chunks
            new_chunks = chunk_file(file_path, self.config.index.languages)
            new_chunk_hashes = {chunk.hash for chunk in new_chunks}

            # Get existing chunk hashes for this file
            existing_hashes = self.db.get_chunk_hashes_for_file(str(file_path))

            # Find chunks to delete (no longer exist in new version)
            hashes_to_delete = existing_hashes - new_chunk_hashes

            # Find chunks to insert (new or changed)
            chunks_to_insert = [chunk for chunk in new_chunks if chunk.hash not in existing_hashes]

            # Delete stale chunks
            if hashes_to_delete:
                self.db.delete_chunks_by_hash(list(hashes_to_delete))
                logger.debug("Deleted %d stale chunks from %s", len(hashes_to_delete), file_path)

            # Insert new chunks
            if chunks_to_insert:
                chunk_texts = [
                    f"{chunk.context}\n{chunk.content}" if chunk.context else chunk.content
                    for chunk in chunks_to_insert
                ]
                embeddings = self.embedder.embed(chunk_texts)
                self._store_chunks(file_path, chunks_to_insert, embeddings)
                logger.debug("Inserted %d new chunks for %s", len(chunks_to_insert), file_path)

            # Update file metadata
            self.db.update_file_metadata(str(file_path), len(new_chunks))

        except Exception as e:
            logger.warning("Failed to update file %s: %s", file_path, e)

    def delete_file(self, file_path: Path) -> None:
        """
        Remove a file from the index.

        Args:
            file_path: Path to the file that was deleted.
        """
        try:
            deleted_count = self.db.delete_file(str(file_path))
            if deleted_count > 0:
                logger.info("Removed %d chunks for deleted file %s", deleted_count, file_path)
        except Exception as e:
            logger.warning("Failed to delete file %s from index: %s", file_path, e)

    def _load_gitignore(self, directory: Path) -> pathspec.PathSpec | None:
        """
        Load and parse a .gitignore file from the given directory.

        Args:
            directory: Directory to check for .gitignore file.

        Returns:
            PathSpec object if .gitignore exists and is parseable, None otherwise.
        """
        if directory in self._gitignore_cache:
            return self._gitignore_cache[directory]

        gitignore_path = directory / ".gitignore"
        if not gitignore_path.exists() or not gitignore_path.is_file():
            self._gitignore_cache[directory] = None
            return None

        try:
            with open(gitignore_path, encoding="utf-8") as f:
                patterns = f.read().splitlines()
            spec = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, patterns)
            self._gitignore_cache[directory] = spec
            return spec
        except Exception as e:
            logger.warning("Failed to parse .gitignore at %s: %s", gitignore_path, e)
            self._gitignore_cache[directory] = None
            return None

    def _collect_files(self) -> list[Path]:
        """
        Walk file tree and collect files matching include/exclude rules.

        Uses depth-first traversal to discover and load .gitignore files lazily.

        Returns:
            List of file paths to index.
        """
        # Clear gitignore cache at start of each collection
        self._gitignore_cache.clear()

        files = []

        # Use os.walk for depth-first traversal with control over directory order
        for root, _dirs, filenames in os.walk(self.project_path):
            root_path = Path(root)

            # Load .gitignore for current directory if present
            self._load_gitignore(root_path)

            # Process files in current directory
            for filename in filenames:
                file_path = root_path / filename
                if self._should_index_file(file_path):
                    files.append(file_path)

        return files

    def _should_index_file(self, file_path: Path) -> bool:
        """
        Check if a file should be indexed based on include/exclude rules.

        Args:
            file_path: Path to check.

        Returns:
            True if file should be indexed.
        """
        project_path = self.project_path
        try:
            relative_path = file_path.relative_to(project_path)
        except ValueError:
            # File is outside project path
            return False

        relative_str = str(relative_path)

        # Check excludes first (higher priority)
        for pattern in self.config.index.exclude:
            if self._matches_pattern(relative_str, pattern):
                return False

        # Check includes (if any specified)
        if self.config.index.include:
            for pattern in self.config.index.include:
                if self._matches_pattern(relative_str, pattern):
                    return True
            return False

        # If no includes specified, include by default (unless excluded above)
        return True

    @staticmethod
    def _matches_pattern(path: str, pattern: str) -> bool:
        """
        Check if a path matches a glob-like pattern.

        Args:
            path: Path to check (relative to project root).
            pattern: Pattern to match (supports * and **).

        Returns:
            True if path matches pattern.
        """
        from fnmatch import fnmatch

        # Handle directory patterns (ending with /)
        if pattern.endswith("/"):
            # Match if path starts with this directory
            return path.startswith(pattern) or fnmatch(path, pattern + "*")

        # Handle ** patterns (match across directories)
        if "**" in pattern:
            # Convert ** to match any number of directory levels
            pattern = pattern.replace("**/", "")
            parts = pattern.split("/")
            if len(parts) == 1:
                # Just a filename pattern with **
                return fnmatch(Path(path).name, pattern.replace("**", "*"))
            else:
                # Match anywhere in path
                return fnmatch(path, f"*{pattern}")

        # Standard glob match
        return fnmatch(path, pattern)

    def _store_chunks(
        self, file_path: Path, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        """
        Store chunks and their embeddings in the database.

        Args:
            file_path: Path to the source file.
            chunks: List of chunks to store.
            embeddings: List of embedding vectors (one per chunk).
        """
        if len(chunks) != len(embeddings):
            raise IndexingError(
                f"Chunk count ({len(chunks)}) does not match embedding count ({len(embeddings)})"
            )

        # Prepare chunk records
        chunk_records = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            chunk_records.append(
                {
                    "file_path": str(file_path),
                    "language": chunk.language,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "content": chunk.content,
                    "context": chunk.context,
                    "hash": chunk.hash,
                    "embedding": embedding,
                }
            )

        # Insert all chunks in batch
        self.db.insert_chunks(chunk_records)

    def wait_for_completion(self, timeout: float | None = None) -> bool:
        """
        Wait for indexing to complete.

        Args:
            timeout: Maximum time to wait in seconds. None = wait forever.

        Returns:
            True if indexing completed, False if timeout occurred.
        """
        if self._indexing_thread is None:
            return True

        self._indexing_thread.join(timeout=timeout)
        return not self._indexing_thread.is_alive()
