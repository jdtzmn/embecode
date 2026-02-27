# Findings: DuckDB Cross-Process Concurrency

## Summary

DuckDB does **not** support mixed read-write + read-only connections across
separate OS processes. A `read_only=True` connection will raise
`duckdb.IOException` if another process holds a read-write lock on the same
database file. This is a fundamental limitation of DuckDB's file-level locking
model, not a bug in embecode's connection code.

## Background

The lock-file daemon design (`specs/lock-file-daemon.md`) assumed two embecode
processes could share the same DuckDB index simultaneously: an OWNER process
with a read-write connection (for indexing) and a READER process with a
read-only connection (for search). The comment at `src/embecode/db.py:54`
stated:

> A read-only connection cannot write to the database but can coexist with a
> separate read-write connection from another process (DuckDB supports this).

This claim is **incorrect**.

## Investigation

Three fix options were tested empirically (iteration #4):

### Option A: Config dict connect signature

Tried `duckdb.connect(database=path, config={"access_mode": "read_only"})`
instead of the positional `read_only=True` argument.

**Result: FAILED.** Same `IOException` cross-process, `ConnectionException`
in-process. The call signature is not the issue.

### Option B: Explicit CHECKPOINT before reader connects

Tried issuing `CHECKPOINT` on the owner connection before the reader attempts
to connect, to ensure WAL pages are flushed to the main database file.

**Result: FAILED.** `CHECKPOINT` flushes WAL pages to disk but does **not**
release the process-level file lock. The reader still cannot open the database.

### Option C: Fundamental DuckDB limitation

DuckDB's documentation
([duckdb.org/docs/stable/connect/concurrency](https://duckdb.org/docs/stable/connect/concurrency))
explicitly describes only two supported cross-process modes:

1. **Single process, read-write** -- one process opens the database in RW mode;
   no other process can open it at all.
2. **Multiple processes, all read-only** -- any number of processes can open the
   database in RO mode simultaneously, but none can write.

There is **no mixed mode** where one process holds a read-write connection while
another opens a read-only connection. This was confirmed empirically: after the
RW connection is closed, multiple RO connections from different processes work
perfectly.

**Result: CONFIRMED.** This is a DuckDB architectural constraint, not a
configuration or API issue.

## Evidence

### Reproducing the error

The test `test_reader_can_search_while_owner_holds_connection` in
`tests/test_concurrent.py` reliably reproduces the failure:

```
duckdb.IOException: IO Error: Could not set lock on file ".../index.db":
Conflicting lock is held in python3.12 (PID XXXXX)
```

The error occurs at the `duckdb.connect()` call in `db.py:65`. The test is
marked `xfail(strict=True, raises=duckdb.IOException)` to document this as a
known limitation -- if a future DuckDB release lifts the restriction, the strict
xfail will break, alerting us.

### Multi-reader model works

The test `test_reader_sees_owner_written_data` confirms the viable alternative:
an owner process indexes data, closes its connection, and then a reader process
opens the database read-only and successfully finds the indexed content via both
BM25 search and `get_index_stats()`.

## Implications for embecode

1. **The db.py comment at line 54 is wrong** and should be corrected to reflect
   this finding.

2. **The two-process model (simultaneous OWNER + READER) is not viable** with
   DuckDB's current locking semantics.

3. **The sequential model works**: owner indexes, closes the database, then
   readers open read-only. This is validated by the passing integration test.

## Open Questions / Follow-up

If simultaneous OWNER + READER access is required, potential alternatives
include:

- **Snapshot copy**: reader copies the `.db` file at startup and queries the
  snapshot. Simple but stale.
- **IPC routing**: reader sends search requests to the owner process via IPC
  (Unix socket, HTTP, etc.) instead of opening its own DuckDB connection. More
  complex but provides live data.
- **Queue behind owner**: eliminate the reader role entirely; all processes queue
  behind the single owner for both indexing and search operations.

These options are out of scope for this investigation and should be addressed in
a follow-up spec if simultaneous access is needed.

## Test Commands

```bash
# Run the concurrent tests
pytest tests/test_concurrent.py -v --no-cov

# Expected output:
#   test_reader_can_search_while_owner_holds_connection — XFAIL
#   test_reader_sees_owner_written_data — PASSED
```
