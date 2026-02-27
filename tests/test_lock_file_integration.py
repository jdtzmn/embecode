"""Integration tests for the lock-file daemon protocol.

These tests launch real subprocesses (not mocks) to verify:
- Two-process scenario: owner + reader both start, reader promotes after owner exits.
- Race condition: multiple readers simultaneously after owner exits, exactly one promotes.
- Crash recovery: lock file with dead PID is cleaned up and new process becomes owner.

Each test uses real lock file coordination in a temporary directory.  The
subprocesses use mocked DB/indexer/embedder to avoid needing a real embedding
model, but the lock-file state machine and ``_promote_to_owner`` paths run
exactly as they would in production.

IPC note: the subprocesses communicate results via JSON files rather than
multiprocessing.Queue because the ``signal.signal`` calls in ``_setup_owner``
replace SIGTERM/SIGINT handlers in a way that can terminate the Queue's
background pipe-writer thread unexpectedly.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Worker functions - run in forked subprocesses
# ---------------------------------------------------------------------------
# All worker functions write their result to a JSON file and signal readiness
# via a multiprocessing.Event.  They accept a ``stop_event`` to exit cleanly.
# The ``result_file`` path is passed as a string so it survives the fork.


def _worker_owner(
    cache_dir: str,
    project_dir: str,
    result_file: str,
    ready_event: multiprocessing.Event,
    stop_event: multiprocessing.Event,
) -> None:
    """
    Start an EmbeCodeServer (expects to become owner) and wait until stop_event.

    Writes {"role": ..., "pid": ...} to result_file then sets ready_event.
    """
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch

    cache_dir_path = Path(cache_dir)
    project_dir_path = Path(project_dir)

    def _cm_mock():
        cm = Mock()
        cm.get_cache_dir.return_value = cache_dir_path
        cm.get_lock_path.return_value = cache_dir_path / "daemon.lock"
        cm.update_access_time.return_value = None
        return cm

    def _db_mock():
        db = Mock()
        db.get_metadata.return_value = None
        db.set_metadata.return_value = None
        db.connect.return_value = None
        db.close.return_value = None
        db._conn = object()
        return db

    def _cfg_mock():
        cfg = Mock()
        cfg.embeddings.model = "test-model"
        cfg.daemon.auto_watch = False
        cfg.daemon.debounce_ms = 500
        return cfg

    with (
        patch("embecode.server.threading.Thread"),
        patch("embecode.server.Indexer"),
        patch("embecode.server.Searcher"),
        patch("embecode.server.Embedder"),
        patch("embecode.server.Database") as mock_db_cls,
        patch("embecode.server.CacheManager") as mock_cm_cls,
        patch("embecode.server.load_config") as mock_cfg,
    ):
        mock_cfg.return_value = _cfg_mock()
        mock_cm_cls.return_value = _cm_mock()
        mock_db_cls.return_value = _db_mock()

        from embecode.server import EmbeCodeServer

        try:
            server = EmbeCodeServer(project_dir_path)
            Path(result_file).write_text(
                json.dumps({"role": server._role, "pid": os.getpid(), "error": None})
            )
        except Exception as exc:
            Path(result_file).write_text(
                json.dumps({"role": None, "pid": os.getpid(), "error": str(exc)})
            )
        finally:
            ready_event.set()

        # Keep running until told to stop or timeout
        stop_event.wait(timeout=30)
        try:
            server.cleanup()
        except Exception:
            pass


def _worker_reader_then_promote(
    cache_dir: str,
    project_dir: str,
    init_result_file: str,
    final_result_file: str,
    ready_event: multiprocessing.Event,
    stop_event: multiprocessing.Event,
) -> None:
    """
    Start as a reader, then poll for owner exit and call ``_promote_to_owner``.

    Writes initial role to init_result_file and sets ready_event.
    After promotion attempt, writes final role to final_result_file.
    """
    import json
    import os
    import time
    from pathlib import Path
    from unittest.mock import Mock, patch

    cache_dir_path = Path(cache_dir)
    project_dir_path = Path(project_dir)
    lock_path = cache_dir_path / "daemon.lock"

    def _cm_mock():
        cm = Mock()
        cm.get_cache_dir.return_value = cache_dir_path
        cm.get_lock_path.return_value = lock_path
        cm.update_access_time.return_value = None
        return cm

    def _db_mock():
        db = Mock()
        db.get_metadata.return_value = None
        db.set_metadata.return_value = None
        db.connect.return_value = None
        db.close.return_value = None
        db._conn = object()
        return db

    def _cfg_mock():
        cfg = Mock()
        cfg.embeddings.model = "test-model"
        cfg.daemon.auto_watch = False
        cfg.daemon.debounce_ms = 500
        return cfg

    with (
        patch("embecode.server.threading.Thread"),
        patch("embecode.server.Indexer"),
        patch("embecode.server.Searcher"),
        patch("embecode.server.Embedder"),
        patch("embecode.server.Database") as mock_db_cls,
        patch("embecode.server.CacheManager") as mock_cm_cls,
        patch("embecode.server.load_config") as mock_cfg,
    ):
        mock_cfg.return_value = _cfg_mock()
        mock_cm_cls.return_value = _cm_mock()
        mock_db_cls.return_value = _db_mock()

        from embecode.server import EmbeCodeServer, is_pid_alive

        try:
            server = EmbeCodeServer(project_dir_path)
        except Exception as exc:
            Path(init_result_file).write_text(
                json.dumps({"initial_role": None, "pid": os.getpid(), "error": str(exc)})
            )
            ready_event.set()
            return

        Path(init_result_file).write_text(
            json.dumps({"initial_role": server._role, "pid": os.getpid(), "error": None})
        )
        ready_event.set()

        # Poll: wait for lock to disappear or become stale, then promote
        deadline = time.time() + 25
        while time.time() < deadline and not stop_event.is_set():
            if not lock_path.exists():
                try:
                    server._promote_to_owner()
                except Exception:
                    pass
                break
            else:
                try:
                    data = json.loads(lock_path.read_text())
                    owner_pid = data.get("pid")
                    if owner_pid and not is_pid_alive(owner_pid):
                        try:
                            lock_path.unlink()
                        except OSError:
                            pass
                        try:
                            server._promote_to_owner()
                        except Exception:
                            pass
                        break
                except (OSError, json.JSONDecodeError):
                    pass
            time.sleep(0.05)

        Path(final_result_file).write_text(
            json.dumps({"final_role": server._role, "pid": os.getpid()})
        )

        # Keep running until told to stop so that if we became the new owner,
        # the lock file stays alive long enough for the test to verify roles.
        # This prevents a promoted-owner from removing its lock before other
        # readers have had a chance to finish their own promotion attempts.
        stop_event.wait(timeout=30)

        try:
            server.cleanup()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helper: wait for a result file to appear and read it
# ---------------------------------------------------------------------------


def _wait_for_result(result_file: Path, timeout: float = 15.0) -> dict:
    """Poll until result_file exists and is valid JSON, then return its contents."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if result_file.exists():
            try:
                return json.loads(result_file.read_text())
            except json.JSONDecodeError:
                pass
        time.sleep(0.05)
    raise TimeoutError(f"Result file not written within {timeout}s: {result_file}")


def _start_owner(tmp_path: Path, cache_dir: Path, project_dir: Path):
    """Convenience: start an owner worker process."""
    ctx = multiprocessing.get_context("fork")
    result_file = tmp_path / f"owner_init_{os.getpid()}.json"
    ready = ctx.Event()
    stop = ctx.Event()
    proc = ctx.Process(
        target=_worker_owner,
        args=(str(cache_dir), str(project_dir), str(result_file), ready, stop),
        daemon=True,
    )
    proc.start()
    return proc, result_file, ready, stop


def _start_reader_watcher(
    tmp_path: Path,
    cache_dir: Path,
    project_dir: Path,
    idx: int = 0,
):
    """Convenience: start a reader-then-promote worker process."""
    ctx = multiprocessing.get_context("fork")
    init_file = tmp_path / f"reader_init_{idx}_{os.getpid()}.json"
    final_file = tmp_path / f"reader_final_{idx}_{os.getpid()}.json"
    ready = ctx.Event()
    stop = ctx.Event()
    proc = ctx.Process(
        target=_worker_reader_then_promote,
        args=(
            str(cache_dir),
            str(project_dir),
            str(init_file),
            str(final_file),
            ready,
            stop,
        ),
        daemon=True,
    )
    proc.start()
    return proc, init_file, final_file, ready, stop


# ---------------------------------------------------------------------------
# TestTwoProcessScenario
# ---------------------------------------------------------------------------


class TestTwoProcessScenario:
    """Owner and reader coexist; reader promotes when owner exits."""

    def test_owner_and_reader_both_start_successfully(self, tmp_path: Path) -> None:
        """
        Start an owner process, then start a reader process.
        Both must report the correct role without errors.
        """
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        owner_proc, owner_result_file, owner_ready, owner_stop = _start_owner(
            tmp_path, cache_dir, project_dir
        )

        # Wait for owner to be ready
        assert owner_ready.wait(timeout=15), "Owner did not start in time"
        owner_result = _wait_for_result(owner_result_file)
        assert owner_result["error"] is None, f"Owner error: {owner_result['error']}"
        assert owner_result["role"] == "owner", f"Expected owner role, got: {owner_result}"

        # Now start a reader (the lock file exists with the owner's live PID)
        reader_proc, reader_init_file, _reader_final_file, reader_ready, reader_stop = (
            _start_reader_watcher(tmp_path, cache_dir, project_dir, idx=0)
        )
        assert reader_ready.wait(timeout=15), "Reader did not start in time"
        reader_init = _wait_for_result(reader_init_file)
        assert reader_init["error"] is None, f"Reader error: {reader_init['error']}"
        assert reader_init["initial_role"] == "reader", f"Expected reader role, got: {reader_init}"

        # Cleanup
        owner_stop.set()
        reader_stop.set()
        owner_proc.join(timeout=10)
        reader_proc.join(timeout=10)

    def test_reader_promotes_to_owner_after_owner_exits(self, tmp_path: Path) -> None:
        """
        Start an owner, then a reader/watcher.  Stop the owner cleanly.
        The reader must detect the lock removal and promote to owner.
        """
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Start owner
        owner_proc, owner_result_file, owner_ready, owner_stop = _start_owner(
            tmp_path, cache_dir, project_dir
        )
        assert owner_ready.wait(timeout=15), "Owner did not start in time"
        owner_result = _wait_for_result(owner_result_file)
        assert owner_result["role"] == "owner", f"Expected owner role: {owner_result}"

        # Start reader + watcher
        reader_proc, reader_init_file, reader_final_file, reader_ready, reader_stop = (
            _start_reader_watcher(tmp_path, cache_dir, project_dir, idx=0)
        )
        assert reader_ready.wait(timeout=15), "Reader did not start in time"
        reader_init = _wait_for_result(reader_init_file)
        assert reader_init["initial_role"] == "reader", (
            f"Reader should start as reader: {reader_init}"
        )

        # Stop owner cleanly — triggers atexit → removes lock file
        owner_stop.set()
        owner_proc.join(timeout=10)

        # Verify lock file was removed
        lock_path = cache_dir / "daemon.lock"
        deadline = time.time() + 5
        while lock_path.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert not lock_path.exists(), "Lock file should be removed after owner cleanup"

        # Wait for reader to detect the removal and promote
        final_result = _wait_for_result(reader_final_file, timeout=20)
        assert final_result["final_role"] == "owner", (
            f"Reader should have promoted to owner: {final_result}"
        )

        reader_stop.set()
        reader_proc.join(timeout=10)

    def test_lock_file_absent_after_owner_exits(self, tmp_path: Path) -> None:
        """After the owner process exits cleanly, ``daemon.lock`` must be removed."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        lock_path = cache_dir / "daemon.lock"

        owner_proc, owner_result_file, owner_ready, owner_stop = _start_owner(
            tmp_path, cache_dir, project_dir
        )
        assert owner_ready.wait(timeout=15), "Owner did not start in time"
        _wait_for_result(owner_result_file)

        # Lock must exist while owner is alive
        assert lock_path.exists(), "daemon.lock must exist while owner is running"

        # Stop owner cleanly
        owner_stop.set()
        owner_proc.join(timeout=10)

        # Lock must be gone
        deadline = time.time() + 5
        while lock_path.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert not lock_path.exists(), "daemon.lock must be removed after owner cleanup"


# ---------------------------------------------------------------------------
# TestRaceCondition
# ---------------------------------------------------------------------------


class TestRaceCondition:
    """When multiple readers see the owner exit simultaneously, exactly one promotes."""

    def test_exactly_one_reader_becomes_owner_in_race(self, tmp_path: Path) -> None:
        """
        Start an owner and N readers.  Stop the owner.  Verify that among the
        readers, exactly one becomes the new owner (the rest stay as readers).
        """
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        lock_path = cache_dir / "daemon.lock"

        N_READERS = 3

        # Start owner
        owner_proc, owner_result_file, owner_ready, owner_stop = _start_owner(
            tmp_path, cache_dir, project_dir
        )
        assert owner_ready.wait(timeout=15), "Owner did not start in time"
        owner_result = _wait_for_result(owner_result_file)
        assert owner_result["role"] == "owner"
        original_owner_pid = owner_result["pid"]

        # Start N readers
        readers = []
        for i in range(N_READERS):
            proc, init_file, final_file, ready, stop = _start_reader_watcher(
                tmp_path, cache_dir, project_dir, idx=i
            )
            readers.append((proc, init_file, final_file, ready, stop))

        # Wait for all readers to be ready
        for _proc, init_file, _final_file, ready, _stop in readers:
            assert ready.wait(timeout=15), "A reader did not start in time"
            init_result = _wait_for_result(init_file)
            assert init_result["initial_role"] == "reader", f"Expected reader, got: {init_result}"

        # Stop owner cleanly
        owner_stop.set()
        owner_proc.join(timeout=10)

        # Verify lock file was removed by owner
        deadline = time.time() + 5
        while lock_path.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert not lock_path.exists(), "Owner must remove lock on exit"

        # Give readers a brief window to detect lock removal and race to promote.
        # The poll interval is 50ms, so 2s is ample for all readers to detect
        # the lock removal and complete their promotion attempts.
        time.sleep(2.0)

        # Signal all readers to exit their polling loops so they write final files.
        # Workers write final_result_file BEFORE entering stop_event.wait(), so
        # the files should appear very quickly once stop is set.
        for _proc, _init_file, _final_file, _ready, stop in readers:
            stop.set()

        # Collect final roles
        final_roles = []
        final_pids = []
        for proc, _init_file, final_file, _ready, _stop in readers:
            try:
                final_result = _wait_for_result(final_file, timeout=15)
                final_roles.append(final_result.get("final_role"))
                final_pids.append(final_result.get("pid"))
            except TimeoutError:
                final_roles.append(None)
                final_pids.append(None)
            finally:
                proc.join(timeout=5)

        owner_count = final_roles.count("owner")
        reader_count = final_roles.count("reader")

        assert owner_count == 1, (
            f"Expected exactly 1 new owner, got {owner_count}. Roles: {final_roles}"
        )
        assert reader_count == N_READERS - 1, (
            f"Expected {N_READERS - 1} readers to stay, got {reader_count}. Roles: {final_roles}"
        )

        # The new owner's PID must differ from the old owner's PID
        new_owner_idx = final_roles.index("owner")
        new_owner_pid = final_pids[new_owner_idx]
        assert new_owner_pid is not None, "New owner PID should be readable from final file"
        assert new_owner_pid != original_owner_pid, "New owner PID should differ from old owner PID"


# ---------------------------------------------------------------------------
# TestCrashRecovery
# ---------------------------------------------------------------------------


class TestCrashRecovery:
    """Lock file with a dead PID is cleaned up on next startup."""

    def test_dead_pid_in_lock_file_cleaned_up_on_startup(self, tmp_path: Path) -> None:
        """
        Write a lock file with a definitely-dead PID.
        The next process to start must remove the stale lock and become owner.
        """
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        lock_path = cache_dir / "daemon.lock"

        # Simulate crashed owner: dead PID
        dead_pid = 999_999_999
        lock_path.write_text(json.dumps({"pid": dead_pid}))
        assert lock_path.exists()

        proc, result_file, ready, stop = _start_owner(tmp_path, cache_dir, project_dir)
        assert ready.wait(timeout=15), "Process did not start in time"
        result = _wait_for_result(result_file)

        try:
            assert result["error"] is None, f"Unexpected error: {result['error']}"
            assert result["role"] == "owner", (
                f"Expected owner after stale lock cleanup, got: {result['role']}"
            )
            # Lock file must now contain our process's PID, not the dead one
            assert lock_path.exists()
            data = json.loads(lock_path.read_text())
            assert data["pid"] == result["pid"], "Lock file must contain new owner's PID"
            assert data["pid"] != dead_pid, "Stale PID must be replaced"
        finally:
            stop.set()
            proc.join(timeout=5)

    def test_multiple_processes_start_after_crash_exactly_one_owner(self, tmp_path: Path) -> None:
        """
        Write a stale lock file, then start N processes simultaneously.
        Exactly one must become owner; the rest become readers.
        """
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        lock_path = cache_dir / "daemon.lock"

        # Stale lock
        dead_pid = 999_999_998
        lock_path.write_text(json.dumps({"pid": dead_pid}))

        N = 3
        workers = []
        for i in range(N):
            # Use unique result files per worker
            result_file = tmp_path / f"crash_result_{i}.json"
            ctx = multiprocessing.get_context("fork")
            ready = ctx.Event()
            stop = ctx.Event()
            proc = ctx.Process(
                target=_worker_owner,
                args=(str(cache_dir), str(project_dir), str(result_file), ready, stop),
                daemon=True,
            )
            proc.start()
            workers.append((proc, result_file, ready, stop))

        roles = []
        for _proc, result_file, ready, _stop in workers:
            assert ready.wait(timeout=15), "A process did not start in time"
            result = _wait_for_result(result_file)
            roles.append(result.get("role"))

        for proc, _result_file, _ready, stop in workers:
            stop.set()
            proc.join(timeout=5)

        owner_count = roles.count("owner")
        reader_count = roles.count("reader")

        assert owner_count == 1, f"Expected exactly 1 owner, got {owner_count}. Roles: {roles}"
        assert reader_count == N - 1, (
            f"Expected {N - 1} readers, got {reader_count}. Roles: {roles}"
        )

    def test_new_process_starts_fresh_after_corrupt_lock(self, tmp_path: Path) -> None:
        """
        If the lock file contains corrupt JSON (e.g., partial write at crash),
        the next process must clean it up and become owner.
        """
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        lock_path = cache_dir / "daemon.lock"

        # Corrupt lock file (partial write / crash during write)
        lock_path.write_text("{corrupt json!!!")

        proc, result_file, ready, stop = _start_owner(tmp_path, cache_dir, project_dir)
        assert ready.wait(timeout=15), "Process did not start in time"
        result = _wait_for_result(result_file)

        try:
            assert result["error"] is None, f"Unexpected error: {result['error']}"
            assert result["role"] == "owner", (
                f"Expected owner after corrupt lock cleanup, got: {result['role']}"
            )
        finally:
            stop.set()
            proc.join(timeout=5)
