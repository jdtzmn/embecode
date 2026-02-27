# Spec: Lock-File Daemon (v2 Multi-Process Architecture)

## Goal

Allow multiple MCP server processes to share the same DuckDB index file
without lock errors. The first process to start becomes the "owner" (read-write,
indexing, watching). Subsequent processes connect as "readers" (search only, via
IPC to the owner). If the owner exits, a reader automatically promotes itself to
owner and resumes indexing/watching.

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

3. **DuckDB does not support mixed-mode cross-process access.** Empirical
   testing (see `tests/test_concurrent.py`) confirmed that
   `duckdb.connect(path, read_only=True)` raises `duckdb.IOException` when
   another process holds a read-write connection to the same file. DuckDB's
   [concurrency docs](https://duckdb.org/docs/stable/connect/concurrency)
   describe only two cross-process modes: single-process read-write, or
   multi-process all-read-only. There is no mixed mode. This rules out the
   original design where readers open their own read-only DuckDB connection
   while the owner holds a read-write connection. Readers must instead route
   queries to the owner via IPC.

---

## Scope

- **Lock-file protocol**: atomic `daemon.lock` creation to elect a single owner
  process per project.
- **Owner/reader roles**: owner opens DB read-write with indexing, file
  watching, and an IPC server; readers route queries to the owner via a Unix
  domain socket (readers do not open DuckDB).
- **Automatic promotion**: when the owner exits, a reader detects the lock file
  removal via filesystem events and promotes itself to owner.
- **Thread safety**: add a threading lock to `Database` to serialize all DB
  operations within a single process.
- **IPC protocol**: lightweight JSON-over-Unix-socket protocol for readers to
  call `search_code` and `index_status` on the owner.
- **Graceful cleanup**: owner removes `daemon.lock` and `daemon.sock` on
  shutdown via signal handlers and `atexit`.
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
     - Start IPC server on daemon.sock
     - Start catch-up indexer in background thread
     - Start file watcher after catch-up completes
4. If FileExistsError → check existing lock
     - Read PID from lock file
     - If PID is alive → READER
         - Connect to owner's IPC socket (daemon.sock)
         - Start lock file watcher (for promotion detection)
         - Do NOT open DuckDB, start indexer, or file watcher
     - If PID is dead → stale lock
         - Remove stale lock file (and stale daemon.sock if present)
         - Retry from step 2
```

#### Atomic creation

`os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)` guarantees that exactly
one process succeeds when multiple processes race to create the same file. This
is atomic on all POSIX systems and on Windows (via `CreateFile` with
`CREATE_NEW`).

#### Lock file contents

```json
{"pid": 12345, "socket": "/Users/.../.cache/embecode/<hash>/daemon.sock"}
```

JSON with the owner's PID and the absolute path to the IPC socket. The socket
path is deterministic (always `daemon.sock` in the same cache directory), but
including it in the lock file makes the protocol self-describing and simplifies
debugging. No heartbeat timestamp is required because promotion is driven by
filesystem events, not polling.

### Owner Role

The owner extends the current v1 server with an IPC server:

1. Opens DuckDB read-write
2. Starts an IPC server on `~/.cache/embecode/<hash>/daemon.sock` (Unix domain
   socket) that accepts connections from reader processes
3. Runs catch-up indexing in a background thread
4. Starts the file watcher after catch-up completes
5. Serves `search_code` and `index_status` tool calls both locally (for its own
   MCP client) and remotely (for reader processes via IPC)

### Reader Role

A reader is a lightweight proxy server that does **not** open DuckDB:

1. Connects to the owner's Unix domain socket (`daemon.sock`) as an IPC client
2. Does NOT open DuckDB, start the indexer, or start the file watcher
3. Starts a **lock file watcher** — a minimal `watchfiles` watch on the cache
   directory, filtered to `daemon.lock` only
4. Proxies `search_code` and `index_status` tool calls to the owner via IPC and
   returns the owner's responses to its own MCP client
5. `index_status` responses include `"role": "reader"` (appended by the reader
   after receiving the owner's response) to indicate this process is not the
   owner

### IPC Protocol

The owner and readers communicate over a Unix domain socket using a minimal
synchronous JSON protocol. The socket file lives at
`~/.cache/embecode/<hash>/daemon.sock`.

#### Transport

- **Unix domain socket** (`AF_UNIX`, `SOCK_STREAM`).
- The owner binds and listens; each reader connects as a client.
- Each reader connection is handled in a dedicated thread spawned by the owner's
  IPC server.

#### Message framing

Each message (request or response) is framed as:

```
[4 bytes: payload length, big-endian unsigned int][UTF-8 JSON payload]
```

Length-prefixed framing avoids delimiter issues with JSON content and allows
efficient reads.

#### Requests (reader → owner)

```json
{"method": "search_code", "params": {"query": "...", "mode": "hybrid", "top_k": 10, "path": null}}
```

```json
{"method": "index_status"}
```

Only these two methods are supported. Unknown methods receive an error response.

#### Responses (owner → reader)

Success:

```json
{"result": [...]}
```

Error:

```json
{"error": "message describing what went wrong"}
```

#### Connection lifecycle

- A reader opens a single persistent connection to the socket on startup.
- If the connection drops (owner exit, crash), the reader detects this via
  `ConnectionResetError` / `BrokenPipeError` and enters the promotion flow.
- The owner's IPC server stops accepting new connections during shutdown and
  closes existing connections before removing the socket file.

### Promotion: Reader → Owner

When the owner exits (cleanly or crashes), it removes `daemon.lock` and
`daemon.sock` (or they are left behind with a dead PID). The reader detects
this via either the lock file watcher or a dropped IPC connection and promotes:

```
1. Detection: lock file watcher fires (daemon.lock deleted) or IPC connection
   drops (ConnectionResetError / BrokenPipeError)
2. Disconnect IPC client (connection is already gone if owner crashed)
3. Attempt atomic lock file creation (O_CREAT | O_EXCL)
4. If success → promote to OWNER
     a. Stop lock file watcher
     b. Open DuckDB read-write (this is the first DB connection for this
        process — readers never open DuckDB)
     c. Start IPC server on daemon.sock
     d. Write PID and socket path to lock file
     e. Run catch-up indexing (to pick up any changes missed between
        the old owner's death and now)
     f. Start file watcher after catch-up completes
     g. Update role to "owner"
5. If FileExistsError → another reader won the race
     a. Stay as reader
     b. Connect to the new owner's IPC socket
     c. Continue watching lock file
```

**Why catch-up indexing on promotion?** Between the old owner's death and the
new owner's promotion, file changes on disk are not being tracked. The catch-up
indexer already handles this — it diffs the current disk state against the DB
and indexes only the gaps. No new logic needed.

**Why readers don't watch project files:** A reader has no DuckDB connection and
cannot write to the DB, so file change events would be useless. Buffering them
adds complexity for minimal gain since catch-up indexing already covers the gap
on promotion. Keeping readers minimal also reduces resource usage.

### Brief search interruption during promotion

When a reader promotes, it must open a new DuckDB connection and start the IPC
server. During this brief window (typically <100ms), the promoting process
cannot serve search queries. The `search_code` tool should handle this:

- **Promoting reader (becoming owner):** If the DB connection is not yet
  established (mid-promotion), return a retriable error:
  ```json
  [{"error": "Server is reconnecting to the index. Try again in a moment.",
    "retry_recommended": true}]
  ```
- **Other readers (still connected to old owner):** The IPC connection drops
  when the old owner exits. The reader should catch `ConnectionResetError` /
  `BrokenPipeError` and return the same retriable error to its MCP client while
  it either promotes itself or reconnects to the new owner's IPC socket.

### Thread Safety

Add a `threading.Lock` to the `Database` class that serializes all operations:

```python
class Database:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn = None
        self._is_initialized = False
        self._db_lock = threading.Lock()  # NEW

    def connect(self) -> None:
        with self._db_lock:
            if self._conn is not None:
                return
            self._conn = duckdb.connect(str(self.db_path))
            ...

    def vector_search(self, ...) -> ...:
        with self._db_lock:
            ...
```

Every public method on `Database` acquires `_db_lock` before touching
`self._conn`. This prevents the catch-up thread, watcher thread, IPC handler
threads (serving reader requests), and the MCP request handler from corrupting
DuckDB state within a single process. IPC handler threads acquire the same
`_db_lock` when executing queries on behalf of readers, so reader requests are
serialized alongside local owner operations.

### Graceful Cleanup

The owner must remove `daemon.lock` and `daemon.sock` on exit to allow readers
to promote. This must handle both clean and unclean shutdowns:

```python
import atexit
import signal

def _cleanup_lock(lock_path, socket_path, pid):
    """Remove lock and socket files if we are still the owner."""
    try:
        with open(lock_path) as f:
            data = json.load(f)
        if data.get("pid") == pid:
            os.unlink(lock_path)
    except (OSError, json.JSONDecodeError):
        pass
    # Always attempt to remove the socket file
    try:
        os.unlink(socket_path)
    except OSError:
        pass

# Register cleanup in owner setup:
atexit.register(_cleanup_lock, lock_path, socket_path, os.getpid())
for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, lambda *_: sys.exit(0))  # triggers atexit
```

**Crash recovery:** If the owner is killed with `SIGKILL` (or the machine
reboots), the lock file and socket file are left behind with a dead PID. The
next process to start will detect the stale PID on startup (step 4 in the
startup sequence) and clean up both files. Existing readers will detect the
stale lock when they try to promote — they read the PID, check if it's alive,
and if not, remove the stale lock and socket and retry atomic creation.

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

### New: `src/embecode/ipc.py`

IPC server and client classes:

- **`IPCServer`**: binds a Unix domain socket, accepts connections in a listener
  thread, spawns a handler thread per connection. Dispatches `search_code` and
  `index_status` requests to the `EmbeCodeServer` instance. Provides `start()`
  and `stop()` methods.
- **`IPCClient`**: connects to a Unix domain socket, sends length-prefixed JSON
  requests, reads length-prefixed JSON responses. Provides `search_code()` and
  `index_status()` methods that mirror the local API. Raises
  `ConnectionError` on socket disconnect.
- **Message framing helpers**: `send_message(sock, data)` and
  `recv_message(sock)` handle 4-byte length prefix + UTF-8 JSON encoding.

### `src/embecode/db.py`

- Add `threading.Lock` (`_db_lock`) to `__init__`.
- ~~Add `read_only` parameter to `connect()`.~~ Remove `read_only` parameter —
  readers no longer open DuckDB. The `connect()` method always opens read-write.
- Wrap all public methods with `with self._db_lock:`.
- Remove `reconnect(read_only: bool)` — no longer needed since readers don't
  open DuckDB and promotion opens a fresh connection (not a reconnect).
- Remove `is_read_only` property.
- Correct the comment at line 54 that incorrectly claims read-only connections
  can coexist with read-write connections across processes.

### `src/embecode/server.py`

- New: lock file acquisition logic in `EmbeCodeServer.__init__`.
- New: `_role` attribute (`"owner"` or `"reader"`).
- Owner path: open DB read-write, start IPC server, catch-up index, watcher.
- Reader path: connect IPC client to owner, start lock file watcher. Skip
  indexer, watcher, and DB connection entirely.
- New: `_start_ipc_server()` method — creates `IPCServer`, binds socket, starts
  listener thread.
- New: `_connect_ipc_client()` method — creates `IPCClient`, connects to
  owner's socket.
- Update `search_code()` — if reader, proxy via IPC client; if owner, execute
  locally.
- Update `get_index_status()` — if reader, proxy via IPC client and append
  `"role": "reader"`; if owner, execute locally.
- New: `_watch_lock_file()` method — watches cache dir for `daemon.lock`
  deletion.
- New: `_promote_to_owner()` method — disconnect IPC client, atomic lock
  creation, open DB read-write, start IPC server, catch-up index, start watcher.
- Update `cleanup()` to stop IPC server and remove socket if owner, disconnect
  IPC client and stop lock watcher if reader.
- Register `atexit` and signal handlers for lock + socket cleanup.

### `src/embecode/cache.py`

- Add `get_lock_path(project_path) -> Path` helper method.
- Add `get_socket_path(project_path) -> Path` helper method.

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
- **Reader role**: verify reader connects via IPC client and does not open
  DuckDB, start indexer, or file watcher.
- **Thread safety**: verify concurrent DB operations from multiple threads do
  not raise errors.
- **Cleanup**: verify `daemon.lock` and `daemon.sock` are both removed on clean
  shutdown.
- **IPC message framing**: verify `send_message` / `recv_message` correctly
  round-trip JSON payloads of various sizes (empty, small, large).
- **IPC server request dispatch**: verify the IPC server correctly dispatches
  `search_code` and `index_status` requests and returns results.
- **IPC unknown method**: verify the IPC server returns an error response for
  unrecognized method names.
- **IPC client connection drop**: verify the IPC client raises
  `ConnectionError` when the socket is closed by the server.

### Integration tests

- **Two-process scenario**: start owner, start reader, verify the reader can
  serve search queries via IPC while the owner holds its DuckDB connection. Stop
  owner, verify reader promotes and starts indexing. (This replaces the current
  `xfail` test in `test_concurrent.py` with a passing test.)
- **Race condition**: start multiple readers simultaneously after owner exits,
  verify exactly one promotes and the others reconnect to the new owner.
- **Crash recovery**: create lock with dead PID, start new process, verify it
  cleans up stale lock + socket and becomes owner.
- **IPC under load**: owner is indexing (background thread), reader sends
  concurrent search requests via IPC, verify all requests complete without
  errors (may be slower due to `_db_lock` contention).

---

## Edge Cases

- **Lock file on network filesystem (NFS):** `O_EXCL` is not atomic on NFSv2.
  This is acceptable — the cache directory (`~/.cache/embecode/`) should always
  be on local disk. Document this assumption.
- **PID reuse:** In theory, a PID could be reused by a different process. In
  practice, PID reuse with the same lock file path is extremely unlikely in the
  short window between owner death and reader promotion. If it becomes a
  concern, the lock file could include a process start time.
- **DuckDB cross-process limitation:** DuckDB does **not** support mixed-mode
  cross-process access (one read-write + one read-only). This was confirmed
  empirically — `duckdb.connect(path, read_only=True)` raises `IOException`
  when another process holds a read-write connection. This is why readers use
  IPC rather than opening their own DuckDB connection. See
  `tests/test_concurrent.py` for the reproducer.
- **Socket file left behind after crash:** If the owner is killed with `SIGKILL`
  (or the machine reboots), `daemon.sock` is left on disk alongside
  `daemon.lock`. The next process detects the dead PID via the lock file,
  removes both `daemon.lock` and `daemon.sock`, and becomes the new owner.
  A stale socket file is harmless — `connect()` on a stale socket raises
  `ConnectionRefusedError`, which triggers the same stale-lock recovery path.
- **Unix socket path length:** Unix domain socket paths are limited to 104 bytes
  on macOS and 108 bytes on Linux. The path
  `~/.cache/embecode/<32-char-hash>/daemon.sock` is well within this limit for
  typical home directory paths. If the expanded path exceeds the limit, the
  `socket.bind()` call will raise `OSError` and the server should log a clear
  error message.
- **IPC latency:** Unix domain sockets add ~0.1ms per request round-trip,
  negligible compared to embedding time (~50-200ms) and DuckDB query time
  (~1-10ms). Readers should not observe meaningful latency degradation.
- **Owner backpressure during indexing:** When the owner is actively indexing,
  IPC requests from readers queue behind `_db_lock`. Search latency for readers
  may increase during heavy indexing. This is acceptable since search and
  indexing are already serialized within the owner process; the IPC layer does
  not make this worse.
- **Lock file watcher on cache directory:** The watcher monitors the cache
  directory (not the project directory). This is a very low-traffic directory
  so the watcher overhead is negligible.
- **Multiple projects:** Each project has its own `<hash>/daemon.lock` and
  `<hash>/daemon.sock`. The protocol is fully independent per project — no
  global coordination needed.

---

## Migration

No migration needed. The v1 server does not create `daemon.lock` or
`daemon.sock`, so existing deployments will simply start using the new protocol
on upgrade. The first process to start after upgrade becomes the owner and
starts the IPC server.

The `read_only` parameter on `Database.connect()` and the `reconnect()` method
are no longer needed and should be removed. The `is_read_only` property is also
removed. These were part of the original design that assumed readers could open
their own read-only DuckDB connection, which was disproven.
