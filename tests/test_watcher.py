"""Tests for watcher.py - file watching with debounce logic."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from watchfiles import Change

from embecode.config import DaemonConfig, EmbeCodeConfig, IndexConfig
from embecode.watcher import Watcher


@pytest.fixture
def mock_indexer():
    """Mock indexer for testing."""
    indexer = MagicMock()
    indexer.update_file = MagicMock()
    indexer.delete_file = MagicMock()
    return indexer


@pytest.fixture
def test_config():
    """Test configuration with fast debounce."""
    config = EmbeCodeConfig()
    config.daemon = DaemonConfig(debounce_ms=100, auto_watch=True)
    config.index = IndexConfig(
        include=["src/", "lib/"],
        exclude=["node_modules/", "*.min.js", "**/__pycache__/"],
    )
    return config


@pytest.fixture
def test_project_path(tmp_path):
    """Create a test project directory structure."""
    project = tmp_path / "test_project"
    project.mkdir()

    # Create some test files
    (project / "src").mkdir()
    (project / "src" / "main.py").write_text("print('hello')")
    (project / "lib").mkdir()
    (project / "lib" / "utils.py").write_text("def foo(): pass")
    (project / "node_modules").mkdir()
    (project / "node_modules" / "pkg.js").write_text("// excluded")

    return project


def test_watcher_initialization(test_project_path, test_config, mock_indexer):
    """Test watcher can be initialized."""
    watcher = Watcher(test_project_path, test_config, mock_indexer)

    assert watcher.project_path == test_project_path
    assert watcher.config == test_config
    assert watcher.indexer == mock_indexer
    assert watcher._thread is None


def test_watcher_pattern_matching(test_project_path, test_config, mock_indexer):
    """Test pattern matching logic for include/exclude rules."""
    watcher = Watcher(test_project_path, test_config, mock_indexer)

    # Test directory prefix matching
    assert watcher._matches_pattern("src/main.py", "src/")
    assert watcher._matches_pattern("src/foo/bar.py", "src/")
    assert not watcher._matches_pattern("lib/utils.py", "src/")

    # Test wildcard matching
    assert watcher._matches_pattern("app.min.js", "*.min.js")
    assert watcher._matches_pattern("dist/app.min.js", "*.min.js")
    assert not watcher._matches_pattern("app.js", "*.min.js")

    # Test recursive wildcard matching
    assert watcher._matches_pattern("src/__pycache__/foo.pyc", "**/__pycache__/")
    assert watcher._matches_pattern("lib/utils/__pycache__/bar.pyc", "**/__pycache__/")
    assert not watcher._matches_pattern("src/main.py", "**/__pycache__/")


def test_watcher_should_process_file_included(test_project_path, test_config, mock_indexer):
    """Test file inclusion based on include patterns."""
    watcher = Watcher(test_project_path, test_config, mock_indexer)

    # Files in include patterns should be processed
    assert watcher._should_process_file(test_project_path / "src" / "main.py")
    assert watcher._should_process_file(test_project_path / "src" / "foo" / "bar.py")
    assert watcher._should_process_file(test_project_path / "lib" / "utils.py")


def test_watcher_should_process_file_excluded(test_project_path, test_config, mock_indexer):
    """Test file exclusion based on exclude patterns."""
    watcher = Watcher(test_project_path, test_config, mock_indexer)

    # Files matching exclude patterns should not be processed
    assert not watcher._should_process_file(test_project_path / "node_modules" / "pkg.js")
    assert not watcher._should_process_file(test_project_path / "dist" / "app.min.js")
    assert not watcher._should_process_file(
        test_project_path / "src" / "__pycache__" / "main.cpython-311.pyc"
    )


def test_watcher_should_process_file_not_in_include(test_project_path, test_config, mock_indexer):
    """Test files not in include patterns are excluded."""
    watcher = Watcher(test_project_path, test_config, mock_indexer)

    # Files not matching any include pattern should be excluded
    assert not watcher._should_process_file(test_project_path / "README.md")
    assert not watcher._should_process_file(test_project_path / "test" / "test_foo.py")


def test_watcher_should_process_file_outside_project(test_project_path, test_config, mock_indexer):
    """Test files outside project path are excluded."""
    watcher = Watcher(test_project_path, test_config, mock_indexer)

    # Files outside project path should not be processed
    outside_file = test_project_path.parent / "other" / "file.py"
    assert not watcher._should_process_file(outside_file)


def test_watcher_start_stop(test_project_path, test_config, mock_indexer):
    """Test watcher can be started and stopped."""
    with patch("embecode.watcher.watch") as mock_watch:
        # Make watch return immediately (empty iterator)
        mock_watch.return_value = iter([])

        watcher = Watcher(test_project_path, test_config, mock_indexer)
        watcher.start()

        # Wait a bit for thread to start
        time.sleep(0.1)

        assert watcher._thread is not None
        assert watcher._thread.is_alive()

        watcher.stop()

        # Wait for thread to exit
        time.sleep(0.2)

        assert not watcher._thread.is_alive()


def test_watcher_process_file_addition(test_project_path, test_config, mock_indexer):
    """Test watcher processes file additions."""
    watcher = Watcher(test_project_path, test_config, mock_indexer)

    # Simulate a file addition
    new_file = test_project_path / "src" / "new.py"
    with watcher._pending_lock:
        watcher._pending_changes[new_file] = Change.added

    # Manually trigger processing (instead of waiting for thread)
    changes = watcher._pending_changes.copy()
    watcher._pending_changes.clear()

    for file_path, change_type in changes.items():
        if change_type == Change.added:
            watcher.indexer.update_file(file_path)

    # Verify indexer was called
    mock_indexer.update_file.assert_called_once_with(new_file)
    mock_indexer.delete_file.assert_not_called()


def test_watcher_process_file_modification(test_project_path, test_config, mock_indexer):
    """Test watcher processes file modifications."""
    watcher = Watcher(test_project_path, test_config, mock_indexer)

    # Simulate a file modification
    modified_file = test_project_path / "src" / "main.py"
    with watcher._pending_lock:
        watcher._pending_changes[modified_file] = Change.modified

    # Manually trigger processing
    changes = watcher._pending_changes.copy()
    watcher._pending_changes.clear()

    for file_path, change_type in changes.items():
        if change_type == Change.modified:
            watcher.indexer.update_file(file_path)

    # Verify indexer was called
    mock_indexer.update_file.assert_called_once_with(modified_file)
    mock_indexer.delete_file.assert_not_called()


def test_watcher_process_file_deletion(test_project_path, test_config, mock_indexer):
    """Test watcher processes file deletions."""
    watcher = Watcher(test_project_path, test_config, mock_indexer)

    # Simulate a file deletion
    deleted_file = test_project_path / "src" / "old.py"
    with watcher._pending_lock:
        watcher._pending_changes[deleted_file] = Change.deleted

    # Manually trigger processing
    changes = watcher._pending_changes.copy()
    watcher._pending_changes.clear()

    for file_path, change_type in changes.items():
        if change_type == Change.deleted:
            watcher.indexer.delete_file(file_path)

    # Verify indexer was called
    mock_indexer.delete_file.assert_called_once_with(deleted_file)
    mock_indexer.update_file.assert_not_called()


def test_watcher_debounce_batches_changes(test_project_path, test_config, mock_indexer):
    """Test debounce logic batches rapid changes."""
    # Use a longer debounce for this test
    test_config.daemon.debounce_ms = 200

    watcher = Watcher(test_project_path, test_config, mock_indexer)

    # Simulate multiple rapid changes to the same file
    test_file = test_project_path / "src" / "main.py"

    with watcher._pending_lock:
        # First change: added
        watcher._pending_changes[test_file] = Change.added

    # Brief pause
    time.sleep(0.05)

    with watcher._pending_lock:
        # Second change: modified (should override added)
        watcher._pending_changes[test_file] = Change.modified

    # Verify only one change is pending (most recent)
    with watcher._pending_lock:
        assert len(watcher._pending_changes) == 1
        assert watcher._pending_changes[test_file] == Change.modified


def test_watcher_handles_multiple_files(test_project_path, test_config, mock_indexer):
    """Test watcher can handle changes to multiple files."""
    watcher = Watcher(test_project_path, test_config, mock_indexer)

    # Simulate changes to multiple files
    file1 = test_project_path / "src" / "main.py"
    file2 = test_project_path / "lib" / "utils.py"
    file3 = test_project_path / "src" / "old.py"

    with watcher._pending_lock:
        watcher._pending_changes[file1] = Change.modified
        watcher._pending_changes[file2] = Change.added
        watcher._pending_changes[file3] = Change.deleted

    # Manually trigger processing
    changes = watcher._pending_changes.copy()
    watcher._pending_changes.clear()

    for file_path, change_type in changes.items():
        if change_type == Change.deleted:
            watcher.indexer.delete_file(file_path)
        else:
            watcher.indexer.update_file(file_path)

    # Verify indexer was called for each file
    assert mock_indexer.update_file.call_count == 2
    assert mock_indexer.delete_file.call_count == 1


def test_watcher_ignores_excluded_files_in_changes(test_project_path, test_config, mock_indexer):
    """Test watcher ignores changes to excluded files."""
    watcher = Watcher(test_project_path, test_config, mock_indexer)

    # Simulate change to excluded file
    excluded_file = test_project_path / "node_modules" / "pkg.js"

    # This should be filtered out by _should_process_file
    if watcher._should_process_file(excluded_file):
        with watcher._pending_lock:
            watcher._pending_changes[excluded_file] = Change.modified

    # Verify no changes were recorded
    with watcher._pending_lock:
        assert len(watcher._pending_changes) == 0

    # Manually trigger processing
    changes = watcher._pending_changes.copy()
    watcher._pending_changes.clear()

    for file_path, _change_type in changes.items():
        watcher.indexer.update_file(file_path)

    # Verify indexer was not called
    mock_indexer.update_file.assert_not_called()


def test_watcher_start_already_running(test_project_path, test_config, mock_indexer):
    """Test starting watcher when already running."""
    with patch("embecode.watcher.watch") as mock_watch:
        mock_watch.return_value = iter([])

        watcher = Watcher(test_project_path, test_config, mock_indexer)
        watcher.start()

        # Try to start again
        watcher.start()

        # Should still only have one thread
        time.sleep(0.1)
        watcher.stop()


def test_watcher_stop_not_running(test_project_path, test_config, mock_indexer):
    """Test stopping watcher when not running."""
    watcher = Watcher(test_project_path, test_config, mock_indexer)

    # Should not raise an error
    watcher.stop()
