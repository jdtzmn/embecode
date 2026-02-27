# Spec: Concurrent Process Integration Test

## Goal

Verify that two embecode processes pointed at the same project can run
simultaneously without errors — one as OWNER (read-write, indexing) and one as
READER (read-only, search-only) — and that the READER can serve search queries
while the OWNER holds its connection.

---

## Problem

The lock-file daemon spec (`lock-file-daemon.md`) designed this two-process
model and it is fully implemented in `server.py` and `db.py`. However there is
a confirmed runtime failure:

```
duckdb.IOException: IO Error: Could not set lock on file "…/index.db":
Conflicting lock is held in … (PID 75066)
```

The reader process calls `duckdb.connect(path, read_only=True)` while the owner
holds its read-write connection, and DuckDB refuses the shared lock. The comment
in `db.py:54` claims this works (*"a read-only connection can coexist with a
separate read-write connection from another process"*) but empirically it does
not. No integration test currently validates the two-process scenario
end-to-end.

---

## Scope

- **New integration test** in `tests/test_concurrent.py` that starts an owner
  process and a reader process against the same real DuckDB index and confirms
  the reader can execute search queries without error.
- **Root cause investigation** of the DuckDB lock conflict as part of making the
  test pass — the fix emerges from the test.
- **Document findings** in `FINDINGS.md` regardless of outcome.
- **Out of scope**: promotion (reader → owner), race conditions between multiple
  readers, NFS edge cases — separate specs if needed.

---

## Design

### Test structure

The test spawns two separate OS processes using `subprocess`, because the lock
behaviour is cross-process and cannot be reproduced in-process with threads.

```
1. Build a real DuckDB index in tmp_path (Indexer + real Database + FixedVectorEmbedder)
2. Spawn OWNER process: opens DB read-write, prints "ready", blocks on stdin
3. Wait for "ready" from owner stdout
4. Open DB read_only=True in the test process (acting as reader)
5. Run a vector_search and a bm25_search directly against the read-only connection
6. Assert results are returned without raising
7. Terminate owner process
```

### Helper script

`tests/helpers/owner_process.py` — a minimal script that accepts a DB path as
`argv[1]`, opens a real `Database` read-write, prints `"ready\n"` to stdout,
then blocks on `sys.stdin.read()` until stdin closes. The parent test controls
its lifetime via `process.communicate()`.

### Investigating the fix

Running the test will confirm whether the issue is:

**A) `duckdb.connect()` call signature** — newer DuckDB may require the config
dict form rather than the positional `read_only` arg:
```python
duckdb.connect(config={"access_mode": "read_only", "path": str(db_path)})
```

**B) WAL / checkpoint state** — the owner must have committed and checkpointed
before the reader can connect. An explicit `CHECKPOINT` after owner setup may
unblock the reader.

**C) DuckDB cross-process read-only is not supported** — if neither A nor B
works, document the finding in `FINDINGS.md` and stop. A follow-up spec will
address the architectural change required.

---

## File Changes

### New: `specs/concurrent-process-test.md`

This document.

### New: `tests/helpers/__init__.py`

Empty, makes `helpers` a package so the owner script can be imported in tests
if needed.

### New: `tests/helpers/owner_process.py`

Minimal owner process helper script.

### New: `tests/test_concurrent.py`

Marked `@pytest.mark.integration` and `@pytest.mark.slow`. Contains:

- `test_reader_can_search_while_owner_holds_connection` — core scenario.
- `test_reader_sees_owner_written_data` — owner indexes a file, reader searches
  for content from it (validates reader sees committed writes).

Run with:

```bash
pytest tests/test_concurrent.py -v --no-cov
```

### Possibly: `src/embecode/db.py`

`connect()` may need a small change to how `duckdb.connect()` is called in
read-only mode, depending on which fix works.

### New: `FINDINGS.md`

Documents the outcome of the investigation — either confirming the fix or
recording that DuckDB cross-process read-only is not supported and why.

---

## Open Questions

1. If DuckDB cross-process read-only is fundamentally unsupported, what is the
   fallback? Options include: reader copies the DB file at startup (snapshot),
   reader queries route through owner via IPC, or the reader role is eliminated
   and all processes queue behind the owner. This is out of scope here — document
   and open a follow-up spec.
