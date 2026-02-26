# Spec: Lock-File Daemon (v2 Multi-Process Architecture)

## Goal

Allow multiple MCP server processes to share the same DuckDB index file
without lock errors. The first process to start becomes the "owner" (read-write,
indexing, watching). Subsequent processes connect as "readers" (read-only, search
only). If the owner exits, a reader automatically promotes itself to owner and
resumes indexing/watching.

---

## Problem

The current architecture (v1) opens a single read-write DuckDB connection per
MCP server process (`db.py:45`). DuckDB does not support concurrent read-write
access from multiple processes to the same file. This means:

1. **User opens two AI tools** (e.g., Claude Code and Cursor) pointed at the
   same project. Both resolve to the same cache directory
   (`~/.cache/embecode/<hash>/`). The second process fails with a DuckDB lock
   error because the first already holds the read-write connection.

2. **Thread safety within a single process.** The background catch-up indexing
   thread (`server.py:95`) and the watcher thread (`watcher.py:66`) both write
   to the database while the main thread serves search queries. The single
   `duckdb.DuckDBPyConnection` object is shared across threads without
   synchronization.

---

## Scope

- **Lock-file protocol**: atomic `daemon.lock` creation to elect a single owner
  process per project.
- **Owner/reader roles**: owner opens DB read-write with indexing and file
  watching; readers open DB read-only with search only.
- **Automatic promotion**: when the owner exits, a reader detects the lock file
  removal via filesystem events and promotes itself to owner.
- **Thread safety**: add a threading lock to `Database` to serialize all DB
  operations within a single process.
- **Graceful cleanup**: owner removes `daemon.lock` on shutdown via signal
  handlers and `atexit`.
- **Status visibility**: `index_status` tool reports the process role.
- **Out of scope**: standalone daemon process (contradicts the "no daemons to
  manage" design goal), HNSW index, symbol index.

---

## Design

### Lock File Protocol

The lock file lives at `~/.cache/embecode/<hash>/daemon.lock` (already reserved
in the cache directory structure per PLAN.md).

#### On startup

```
1. Resolve cache dir: ~/.cache/embecode/<hash>/
2. Attempt atomic lock file creation:
     fd = os.open(lock_path, O_CREAT | O_EXCL | O_WRONLY)
3. If success → OWNER
     - Write PID to lock file, close fd
     - Open DuckDB read-write
     - Start catch-up indexer in background thread
     - Start file watcher after catch-up completes
4. If FileExistsError → check existing lock
     - Read PID from lock file
     - If PID is alive → READER
         - Open DuckDB read-only
         - Start lock file watcher (for promotion detection)
         - Do NOT start indexer or file watcher
     - If PID is dead → stale lock
         - Remove stale lock file
         - Retry from step 2
```

#### Atomic creation

`os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)` guarantees that exactly
one process succeeds when multiple processes race to create the same file. This
is atomic on all POSIX systems and on Windows (via `CreateFile` with
`CREATE_NEW`).

#### Lock file contents

```json
{"pid": 12345}
```

Simple JSON with the owner's PID. Only the PID is needed — no heartbeat
timestamp required because promotion is driven by filesystem events, not polling.

### Owner Role

The owner behaves identically to the current v1 server:

1. Opens DuckDB read-write
2. Runs catch-up indexing in a background thread
3. Starts the file watcher after catch-up completes
4. Serves both `search_code` and `index_status` tool calls

### Reader Role

A reader is a lightweight search-only server:

1. Opens DuckDB read-only (`duckdb.connect(path, read_only=True)`)
2. Does NOT start the indexer or file watcher for the project
3. Starts a **lock file watcher** — a minimal `watchfiles` watch on the cache
   directory, filtered to `daemon.lock` only
4. Serves `search_code` and `index_status` tool calls (reads work fine against
   a read-only connection while the owner writes)
5. `index_status` includes `"role": "reader"` to indicate this process is not
   indexing

### Promotion: Reader → Owner

When the owner exits (cleanly or crashes), it removes `daemon.lock` (or the file
is left behind with a dead PID). The reader detects this and promotes:

```
1. Lock file watcher fires: daemon.lock deleted (or detected stale on check)
2. Attempt atomic lock file creation (O_CREAT | O_EXCL)
3. If success → promote to OWNER
     a. Stop lock file watcher
     b. Close read-only DuckDB connection
     c. Open DuckDB read-write
     d. Write PID to lock file
     e. Run catch-up indexing (to pick up any changes missed between
        the old owner's death and now)
     f. Start file watcher after catch-up completes
     g. Update role to "owner"
4. If FileExistsError → another reader won the race
     a. Stay as reader
     b. Continue watching lock file
```

**Why catch-up indexing on promotion?** Between the old owner's death and the
new owner's promotion, file changes on disk are not being tracked. The catch-up
indexer already handles this — it diffs the current disk state against the DB
and indexes only the gaps. No new logic needed.

**Why readers don't watch project files:** A reader cannot write to the DB, so
file change events would be useless. Buffering them adds complexity for minimal
gain since catch-up indexing already covers the gap on promotion. Keeping readers
minimal also reduces resource usage.

### Brief search interruption during promotion

When a reader promotes, it must close the read-only DuckDB connection and reopen
as read-write. During this brief window (typically <100ms), search queries will
fail. The `search_code` tool should handle this gracefully:

- If the DB connection is `None` (mid-reconnect), return a retriable error:
  ```json
  [{"error": "Server is reconnecting to the index. Try again in a moment.",
    "retry_recommended": true}]
  ```

### Thread Safety

Add a `threading.Lock` to the `Database` class that serializes all operations:

```python
class Database:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn = None
        self._is_initialized = False
        self._db_lock = threading.Lock()  # NEW

    def connect(self, read_only: bool = False) -> None:
        with self._db_lock:
            if self._conn is not None:
                return
            self._conn = duckdb.connect(str(self.db_path), read_only=read_only)
            ...

    def vector_search(self, ...) -> ...:
        with self._db_lock:
            ...
```

Every public method on `Database` acquires `_db_lock` before touching
`self._conn`. This prevents the catch-up thread, watcher thread, and MCP
request handler from corrupting DuckDB state within a single process.

### Graceful Cleanup

The owner must remove `daemon.lock` on exit to allow readers to promote. This
must handle both clean and unclean shutdowns:

```python
import atexit
import signal

def _cleanup_lock(lock_path, pid):
    """Remove lock file if we are still the owner."""
    try:
        with open(lock_path) as f:
            data = json.load(f)
        if data.get("pid") == pid:
            os.unlink(lock_path)
    except (OSError, json.JSONDecodeError):
        pass

# Register cleanup in owner setup:
atexit.register(_cleanup_lock, lock_path, os.getpid())
for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, lambda *_: sys.exit(0))  # triggers atexit
```

**Crash recovery:** If the owner is killed with `SIGKILL` (or the machine
reboots), the lock file is left behind with a dead PID. The next process to
start will detect the stale PID on startup (step 4 in the startup sequence)
and clean it up. Existing readers will detect the stale lock when they try to
promote — they read the PID, check if it's alive, and if not, remove the
stale lock and retry atomic creation.

### Stale PID Detection

```python
import os

def is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        os.kill(pid, 0)  # signal 0 = existence check, no signal sent
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists but we can't signal it
```

---

## File Changes

### `src/embecode/db.py`

- Add `threading.Lock` (`_db_lock`) to `__init__`.
- Add `read_only` parameter to `connect()`.
- Wrap all public methods with `with self._db_lock:`.
- Add `reconnect(read_only: bool)` method for promotion (close + reopen).
- Add `is_read_only` property.

### `src/embecode/server.py`

- New: lock file acquisition logic in `EmbeCodeServer.__init__`.
- New: `_role` attribute (`"owner"` or `"reader"`).
- Owner path: same as current (catch-up index → watcher).
- Reader path: skip indexer/watcher, start lock file watcher instead.
- New: `_watch_lock_file()` method — watches cache dir for `daemon.lock`
  deletion.
- New: `_promote_to_owner()` method — atomic lock creation, DB reconnect,
  catch-up index, start watcher.
- Update `cleanup()` to remove lock file if owner, stop lock watcher if reader.
- Update `get_index_status()` to include `"role"` field.
- Register `atexit` and signal handlers for lock cleanup.

### `src/embecode/cache.py`

- Add `get_lock_path(project_path) -> Path` helper method.

### `src/embecode/indexer.py`

- No structural changes. The indexer is simply not started in reader mode.

### `src/embecode/watcher.py`

- No structural changes. The watcher is simply not started in reader mode.

### `src/embecode/cli.py`

- No changes expected.

---

## Testing

### Unit tests

- **Lock file creation**: verify atomic creation succeeds for first process and
  raises `FileExistsError` for second.
- **Stale lock detection**: create lock with dead PID, verify new process
  removes it and becomes owner.
- **Reader role**: verify reader opens DB read-only and does not start
  indexer/watcher.
- **Thread safety**: verify concurrent DB operations from multiple threads do
  not raise errors.
- **DB reconnect**: verify `reconnect(read_only=False)` correctly transitions
  from read-only to read-write.
- **Cleanup**: verify `daemon.lock` is removed on clean shutdown.

### Integration tests

- **Two-process scenario**: start owner, start reader, verify both can serve
  search queries. Stop owner, verify reader promotes and starts indexing.
- **Race condition**: start multiple readers simultaneously after owner exits,
  verify exactly one promotes.
- **Crash recovery**: create lock with dead PID, start new process, verify it
  cleans up and becomes owner.

---

## Edge Cases

- **Lock file on network filesystem (NFS):** `O_EXCL` is not atomic on NFSv2.
  This is acceptable — the cache directory (`~/.cache/embecode/`) should always
  be on local disk. Document this assumption.
- **PID reuse:** In theory, a PID could be reused by a different process. In
  practice, PID reuse with the same lock file path is extremely unlikely in the
  short window between owner death and reader promotion. If it becomes a
  concern, the lock file could include a process start time.
- **Read-only DuckDB + concurrent writes:** DuckDB supports read-only
  connections reading while a separate read-write connection writes. The reader
  sees a consistent snapshot. The reader may need to periodically "refresh" by
  closing and reopening the connection to see new data written by the owner.
  This should be tested empirically. If DuckDB auto-refreshes on each query,
  no extra work is needed.
- **Lock file watcher on cache directory:** The watcher monitors the cache
  directory (not the project directory). This is a very low-traffic directory
  so the watcher overhead is negligible.
- **Multiple projects:** Each project has its own `<hash>/daemon.lock`. The
  protocol is fully independent per project — no global coordination needed.

---

## Migration

No migration needed. The v1 server does not create `daemon.lock`, so existing
deployments will simply start using the new protocol on upgrade. The first
process to start after upgrade becomes the owner.
