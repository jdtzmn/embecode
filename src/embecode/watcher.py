"""File watcher with debounce logic for incremental indexing."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from watchfiles import Change, watch

if TYPE_CHECKING:
    from embecode.config import EmbeCodeConfig
    from embecode.indexer import Indexer

logger = logging.getLogger(__name__)


class WatcherError(Exception):
    """Base exception for watcher errors."""


class Watcher:
    """
    File watcher that monitors a project directory for changes.

    Uses watchfiles for file system monitoring and implements debounce
    logic to batch rapid changes before triggering incremental indexing.
    """

    def __init__(
        self,
        project_path: Path,
        config: EmbeCodeConfig,
        indexer: Indexer,
    ) -> None:
        """
        Initialize the file watcher.

        Args:
            project_path: Path to the project root directory to watch.
            config: Configuration with debounce settings and include/exclude rules.
            indexer: Indexer instance to call for incremental updates.
        """
        self.project_path = project_path
        self.config = config
        self.indexer = indexer
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending_changes: dict[Path, Change] = {}
        self._pending_lock = threading.Lock()

    def start(self) -> None:
        """
        Start the file watcher in a background daemon thread.

        The thread will continue watching until stop() is called or the
        main process exits.
        """
        if self._thread and self._thread.is_alive():
            logger.warning("Watcher already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="FileWatcher")
        self._thread.start()
        logger.info("File watcher started for %s", self.project_path)

    def stop(self) -> None:
        """
        Stop the file watcher and wait for the thread to exit.

        This is a blocking call that waits for the watcher thread to finish.
        """
        if not self._thread or not self._thread.is_alive():
            logger.warning("Watcher not running")
            return

        logger.info("Stopping file watcher...")
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            logger.warning("Watcher thread did not exit cleanly")
        else:
            logger.info("File watcher stopped")

    def _run(self) -> None:
        """
        Main watcher loop - runs in background thread.

        Monitors file system changes and maintains a pending changes queue
        with debounce logic.
        """
        # Start a separate thread for processing debounced changes
        processor_thread = threading.Thread(
            target=self._process_pending_changes,
            daemon=True,
            name="ChangeProcessor",
        )
        processor_thread.start()

        try:
            # Use watchfiles to monitor the project directory
            # We'll handle filtering ourselves via _should_process_file
            for changes in watch(
                self.project_path,
                stop_event=self._stop_event,
                recursive=True,
                # Don't need to set ignore_permission_denied as watchfiles handles it
            ):
                if self._stop_event.is_set():
                    break

                # Record changes in pending queue
                with self._pending_lock:
                    for change_type, path_str in changes:
                        file_path = Path(path_str)

                        # Check if this is a .gitignore file change
                        if file_path.name == ".gitignore" and change_type in (
                            Change.added,
                            Change.modified,
                        ):
                            logger.info(
                                "Detected .gitignore change: %s - triggering full reindex",
                                file_path.relative_to(self.project_path),
                            )
                            # Trigger full reindex immediately (gitignore rules have changed)
                            self.indexer.start_full_index(background=False)
                            # Skip adding to pending changes since we're doing a full reindex
                            continue

                        # Skip files that don't match our include/exclude rules
                        if not self._should_process_file(file_path):
                            continue

                        # Store the most recent change type for each file
                        self._pending_changes[file_path] = change_type
                        logger.debug(
                            "Recorded change: %s %s",
                            change_type.name,
                            file_path.relative_to(self.project_path),
                        )

        except Exception as e:
            logger.error("Watcher error: %s", e)
            raise WatcherError(f"File watcher failed: {e}") from e

        finally:
            # Wait for processor thread to finish pending changes
            processor_thread.join(timeout=2.0)

    def _process_pending_changes(self) -> None:
        """
        Process pending changes with debounce logic.

        Runs in a separate daemon thread. Periodically checks for pending
        changes and processes them after the debounce interval has elapsed.
        """
        debounce_seconds = self.config.daemon.debounce_ms / 1000.0

        while not self._stop_event.is_set():
            # Sleep for the debounce interval
            time.sleep(debounce_seconds)

            if self._stop_event.is_set():
                break

            # Get pending changes and clear the queue
            with self._pending_lock:
                if not self._pending_changes:
                    continue

                changes_to_process = self._pending_changes.copy()
                self._pending_changes.clear()

            # Process each change
            for file_path, change_type in changes_to_process.items():
                try:
                    relative_path = file_path.relative_to(self.project_path)

                    if change_type == Change.deleted:
                        logger.info("Processing deletion: %s", relative_path)
                        self.indexer.delete_file(file_path)

                    elif change_type in (Change.added, Change.modified):
                        action = "addition" if change_type == Change.added else "modification"
                        logger.info("Processing %s: %s", action, relative_path)
                        self.indexer.update_file(file_path)

                except Exception as e:
                    logger.warning(
                        "Failed to process change for %s: %s",
                        file_path.relative_to(self.project_path),
                        e,
                    )
                    continue

    def _should_process_file(self, file_path: Path) -> bool:
        """
        Check if a file should be processed based on include/exclude rules.

        Args:
            file_path: Absolute path to the file.

        Returns:
            True if the file should be processed, False otherwise.
        """
        try:
            relative_path = file_path.relative_to(self.project_path)
        except ValueError:
            # File is not under project_path
            return False

        # Convert to string for pattern matching
        path_str = str(relative_path)

        # Check exclude patterns first
        for pattern in self.config.index.exclude:
            if self._matches_pattern(path_str, pattern):
                return False

        # Check include patterns
        # If no include patterns specified, include everything (that's not excluded)
        if not self.config.index.include:
            return True

        for pattern in self.config.index.include:
            if self._matches_pattern(path_str, pattern):
                return True

        # If include patterns are specified but none matched, exclude
        return False

    def _matches_pattern(self, path_str: str, pattern: str) -> bool:
        """
        Check if a path matches a glob-style pattern.

        Supports:
        - Simple wildcards: *.py
        - Directory wildcards: **/__pycache__/
        - Prefix matching: src/ matches src/foo/bar.py

        Args:
            path_str: Path string (relative to project root).
            pattern: Pattern string (e.g., "src/", "*.py", "**/__pycache__/").

        Returns:
            True if the path matches the pattern.
        """
        from pathlib import Path as PathlibPath

        # If pattern contains wildcards, use glob matching
        if "*" in pattern or "?" in pattern:
            path_obj = PathlibPath(path_str)
            try:
                # Use match() for glob patterns
                # For patterns ending with /, also try matching the directory and its contents
                if pattern.endswith("/"):
                    # Match both "pattern" and "pattern/*" (directory and its contents)
                    return path_obj.match(pattern.rstrip("/")) or path_obj.match(pattern + "**")
                return path_obj.match(pattern)
            except ValueError:
                # Invalid pattern, no match
                return False

        # If pattern ends with / (no wildcards), it's a directory prefix match
        if pattern.endswith("/"):
            return path_str.startswith(pattern) or path_str.startswith(pattern[:-1] + "/")

        # Otherwise, exact match or prefix match
        return path_str == pattern or path_str.startswith(pattern + "/")
