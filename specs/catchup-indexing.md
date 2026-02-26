# Spec: Catch-Up Indexing on Startup

## Goal

When embecode starts with a partially indexed database (e.g., due to a previous
interrupted indexing run, or files added/modified/deleted while the server was
down), it should automatically detect the gaps and index only the missing or
changed files. It should also detect embedding model changes and refuse to start
with incompatible embeddings.

---

## Problem

The current startup logic in `server.py:75` uses a single check:

```python
if self.db.get_index_stats()["total_chunks"] == 0:
```

This creates a silent failure mode:

1. Server starts, `total_chunks == 0`, begins full indexing.
2. `_run_full_index` calls `clear_index()` (deleting all data), then processes
   files one by one.
3. If the process is killed after N files, the DB has chunks for N files but
   nothing for the rest.
4. On restart, `total_chunks > 0`, so the server skips indexing entirely.
5. The file watcher only reacts to future changes — it never scans for files
   that were never indexed.
6. No warning, no detection, no recovery.

Additionally, files modified or deleted while the server is down are never
reconciled on the next startup.

---

## Scope

- **Catch-up indexing on startup**: detect and index files that are missing,
  modified, or stale in the existing index.
- **Unified startup path**: replace the bifurcated `total_chunks == 0` check
  with a single catch-up flow that handles empty, partial, and complete indexes.
- **Embedding model change detection**: refuse to start if the configured
  embedding model differs from the one used to build the existing index.
- **Index status enhancements**: add `indexing_type` and `files_to_process`
  fields so the `index_status` tool can distinguish catch-up from full indexing
  and show how many files remain.
- **Out of scope**: explicit re-index CLI command (future feature),
  `clear_index()` removal from `start_full_index` (preserved for explicit
  re-index scenarios).

---

## Design

### Unified Startup Flow

On every startup, the server spawns a single background catch-up thread. There
is no longer a separate "full index" vs "existing index" startup path:

- **Empty DB**: catch-up discovers all files as missing, indexes them all.
  Equivalent to a full index, but without the destructive `clear_index()`.
- **Partial DB**: catch-up fills the gaps (missing files, modified files,
  stale entries).
- **Complete DB**: catch-up compares, finds nothing to do, exits quickly.

`start_full_index()` (which calls `clear_index()`) is preserved as a method
on `Indexer` for future explicit re-index scenarios. It is no longer called
from the startup path.

### Catch-Up Algorithm

The background catch-up thread runs the following steps:

1. **Collect files on disk**: call `_collect_files()` to get all files that
   should be indexed per current config and gitignore rules.
2. **Get indexed files from DB**: query the `files` table for all paths and
   their `last_indexed` timestamps.
3. **Classify files into three categories**:
   - **Missing**: on disk but not in DB. Never indexed.
   - **Modified**: on disk AND in DB, but the file's `mtime` is newer than
     `last_indexed`. Modified while the server was down.
   - **Stale**: in DB but not on disk. Deleted while the server was down.
4. **If no work needed**: log a message and exit. Do not set `_is_indexing`.
5. **If work exists**:
   a. Set `_is_indexing = True` (blocks search during catch-up).
   b. Set `_indexing_type = "catchup"` and
      `_files_to_process = len(missing) + len(modified)`.
   c. Log a summary: `"Catch-up indexing: X missing, Y modified, Z stale"`.
   d. Delete stale files from DB using `delete_file()`.
   e. Index missing files using existing `update_file()` logic.
   f. Re-index modified files using existing `update_file()` logic.
   g. Track progress via `_current_file` and `_progress`.
   h. Set `_is_indexing = False` in a `finally` block. Clear `_indexing_type`,
      `_files_to_process`, `_current_file`, and `_progress`.
6. **Start watcher** (if `auto_watch` is enabled) in a `finally` block, so
   the watcher starts regardless of whether catch-up succeeded, failed, or
   had nothing to do.

### Search During Catch-Up

Search is blocked while catch-up is actively indexing (`_is_indexing = True`).
The existing `IndexNotReadyError` mechanism handles this — callers see progress
info ("N files processed, X% complete"). If catch-up finds nothing to do,
`_is_indexing` is never set and search is available immediately.

### File Modification Detection

For files that exist both on disk and in the DB, we compare the file's
`mtime` (via `os.path.getmtime()`) against the `last_indexed` timestamp in
the `files` table. If `mtime > last_indexed`, the file is considered modified
and will be re-indexed.

Both values are compared in UTC. `last_indexed` is already stored as a UTC
timestamp. `os.path.getmtime()` returns seconds since epoch, which is
converted to a UTC `datetime` for comparison.

---

## Embedding Model Change Detection

### Problem

If a user changes the `embeddings.model` config between runs, existing
embeddings become incompatible with new query embeddings (vectors are in
different spaces). Search silently returns degraded or meaningless results.

### Solution

A new `metadata` table in the DB stores key-value pairs. On first run, the
configured embedding model name is stored as `embedding_model`. On subsequent
startups, the stored model is compared against the configured model.

**Startup check (in `EmbeCodeServer.__init__`, before spawning catch-up
thread):**

1. Read `embedding_model` from DB metadata.
2. If `None` (first run or fresh DB): store the current model immediately
   and proceed. Storing early ensures that even if catch-up is interrupted,
   the next run will detect a model mismatch correctly.
3. If stored model equals config model: proceed normally.
4. If stored model differs from config model: raise
   `EmbeddingModelChangedError` with a message telling the user to delete
   the index file and restart.

**Error message format:**

```
Embedding model changed from '<stored>' to '<configured>'.
Existing embeddings are incompatible. Delete the index at <db_path>
and restart, or revert the model in your config.
```

The server refuses to start. The existing `run_server()` error handling
catches the exception, logs it, and exits with code 1.

---

## Index Status Tool Enhancements

### Problem

The `index_status` MCP tool returns `is_indexing`, `current_file`, and
`progress`, but cannot distinguish between a full index and a catch-up, nor
does it show how many files remain to be processed.

### Solution

Two new fields are added to `IndexStatus`:

- **`indexing_type: str | None`** — `None` when not indexing, `"full"` during
  `_run_full_index()`, `"catchup"` during `_run_catchup_index()`.
- **`files_to_process: int | None`** — `None` when not indexing. During a full
  index, the total number of files to index. During catch-up, the count of
  missing + modified files (stale deletions are excluded since they're fast).

These fields are tracked alongside the existing `_is_indexing`, `_current_file`,
and `_progress` using the same lock. They're set when work begins and cleared
in the `finally` block.

**Changes to `Indexer`:**

- Add `_indexing_type: str | None = None` and
  `_files_to_process: int | None = None` instance variables.
- `get_status()` reads them under the lock and passes them to `IndexStatus`.
- `_run_full_index()` sets `_indexing_type = "full"` and
  `_files_to_process = total_files` at the start of the processing loop.
- `_run_catchup_index()` sets `_indexing_type = "catchup"` and
  `_files_to_process = len(missing) + len(modified)` when work is detected.
- Both methods clear both fields in their `finally` block.

**Changes to `IndexStatus`:**

- Constructor gains `indexing_type: str | None = None` and
  `files_to_process: int | None = None` parameters.
- `to_dict()` includes both new fields.

**MCP tool output** — the `index_status` tool automatically returns the new
fields since it calls `status.to_dict()`. Example response during catch-up:

```json
{
  "files_indexed": 42,
  "total_chunks": 210,
  "embedding_model": "all-MiniLM-L6-v2",
  "last_updated": "2026-02-26T10:00:00",
  "is_indexing": true,
  "indexing_type": "catchup",
  "current_file": "src/utils.py",
  "progress": 0.3,
  "files_to_process": 15
}
```

---

## Database Changes

### New table: `metadata`

```sql
CREATE TABLE IF NOT EXISTS metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
)
```

Created in `_initialize_schema()` alongside the existing tables.

`clear_index()` does NOT clear the `metadata` table. Metadata persists
across index clears (so that model tracking survives explicit re-indexes).

### New methods on `Database`

- **`get_indexed_file_paths() -> set[str]`**
  Returns all file paths from the `files` table. Used by catch-up to
  determine which files are already indexed.

- **`get_indexed_files_with_timestamps() -> dict[str, datetime]`**
  Returns a mapping of `{path: last_indexed}` from the `files` table.
  Used by catch-up for mtime comparison.

- **`get_metadata(key: str) -> str | None`**
  Read a value from the `metadata` table. Returns `None` if the key
  does not exist.

- **`set_metadata(key: str, value: str) -> None`**
  Write (upsert) a value into the `metadata` table.

---

## Indexer Changes

### New method: `start_catchup_index(background: bool = True)`

Public entry point, structured identically to `start_full_index()`:

- Checks `_is_indexing` (raises `IndexingInProgressError` if already running).
- If `background=True`: spawns a daemon thread running `_run_catchup_index`.
- If `background=False`: calls `_run_catchup_index` directly.

### New method: `_run_catchup_index()`

Implements the catch-up algorithm described above. Reuses `update_file()` for
indexing individual missing/modified files and `delete_file()` for removing
stale entries. Sets/clears `_is_indexing`, `_indexing_type`,
`_files_to_process`, `_current_file`, and `_progress` with the same locking
pattern as `_run_full_index`.

Progress is tracked as `(files_processed) / (total_missing + total_modified)`
(stale deletions are fast and not counted in the denominator).

If the total work (missing + modified + stale) is zero, the method returns
without setting `_is_indexing`.

### `start_full_index()` updates

No structural changes. `_run_full_index()` is updated to set
`_indexing_type = "full"` and `_files_to_process = total_files`, and clear
them in the `finally` block. It is no longer called from the server startup
path.

---

## Server Changes

### `EmbeCodeServer.__init__`

Replace the current startup logic (lines 74-84) with:

1. **Embedding model check**: read stored model from DB, compare with config.
   Raise `EmbeddingModelChangedError` if mismatched. Store model if first run.
2. **Always spawn background catch-up thread**: call `self._catchup_index()`
   in a daemon thread. No bifurcation on `total_chunks`.

### New method: `_catchup_index()`

Replaces `_initial_index()`. Calls
`self.indexer.start_catchup_index(background=False)`, then starts the watcher
in a `finally` block:

```python
def _catchup_index(self) -> None:
    try:
        self.indexer.start_catchup_index(background=False)
    except Exception:
        logger.exception("Catch-up index failed")
    finally:
        if self.config.daemon.auto_watch:
            self._start_watcher()
```

### New exception: `EmbeddingModelChangedError`

Defined in `server.py`. Subclass of `RuntimeError`.

---

## Edge Cases

### Server killed mid-catch-up

On restart, catch-up runs again and discovers the remaining gaps. Since
catch-up never calls `clear_index()`, it's idempotent — files already
indexed are compared via `update_file()` hash logic and skipped if unchanged.

### Config changes between runs

If `include`/`exclude` rules changed, `_collect_files()` returns the new
file set. Files matching the OLD config but not the new one appear as stale
(in DB but not in `_collect_files()` result) and are removed. Files matching
the NEW config but not the old one appear as missing and are indexed.

### Race between catch-up and watcher

The watcher starts AFTER catch-up completes (in `finally`). No concurrent
indexing of the same files.

### Empty project (no indexable files)

`_collect_files()` returns an empty list. Catch-up sees all DB entries as
stale (if any) and removes them. No files are indexed. Watcher starts
normally.

### `.gitignore` changed while server was down

Catch-up uses the current `.gitignore` rules via `_collect_files()`. Files
newly ignored will appear as stale and be removed. Files newly un-ignored
will appear as missing and be indexed.

---

## Testing Requirements

### File: `tests/test_db.py` (additions)

- **`test_metadata_table_created`**
  After `connect()`, `metadata` table exists in schema.

- **`test_get_metadata_missing_key`**
  `get_metadata("nonexistent")` returns `None`.

- **`test_set_and_get_metadata`**
  `set_metadata("key", "value")` then `get_metadata("key")` returns `"value"`.

- **`test_set_metadata_upsert`**
  `set_metadata("key", "a")` then `set_metadata("key", "b")`. Second call
  overwrites. `get_metadata("key")` returns `"b"`.

- **`test_clear_index_preserves_metadata`**
  `set_metadata("embedding_model", "test")`, then `clear_index()`.
  `get_metadata("embedding_model")` still returns `"test"`.

- **`test_get_indexed_file_paths`**
  Insert file metadata for 3 files. `get_indexed_file_paths()` returns a set
  of those 3 paths.

- **`test_get_indexed_file_paths_empty`**
  No files in DB. `get_indexed_file_paths()` returns empty set.

- **`test_get_indexed_files_with_timestamps`**
  Insert file metadata for 2 files. `get_indexed_files_with_timestamps()`
  returns a dict mapping paths to `datetime` objects.

### File: `tests/test_indexer.py` (additions)

#### Class: `TestIndexStatus` (updates)

- **`test_to_dict_includes_new_fields`**
  `to_dict()` includes `indexing_type` and `files_to_process` keys, defaulting
  to `None` when not provided.

#### Class: `TestCatchUpIndex`

- **`test_catchup_empty_db_indexes_all_files`**
  DB is empty. Catch-up discovers all files as missing, indexes them all.
  `clear_index` is NOT called. `update_file` is called for each file.

- **`test_catchup_partial_db_indexes_missing_files`**
  DB has 2 of 5 files indexed. Catch-up indexes the 3 missing files.
  Already-indexed files are not re-processed.

- **`test_catchup_removes_stale_files`**
  DB has a file that no longer exists on disk. Catch-up calls `delete_file`
  for that path.

- **`test_catchup_reindexes_modified_files`**
  DB has a file with `last_indexed` older than the file's mtime.
  Catch-up calls `update_file` for that file.

- **`test_catchup_skips_unmodified_files`**
  DB has a file with `last_indexed` newer than the file's mtime.
  That file is not re-processed.

- **`test_catchup_complete_index_noop`**
  All files on disk are in DB and unmodified. Catch-up logs and returns.
  `_is_indexing` is never set to `True`.

- **`test_catchup_sets_is_indexing_when_work_exists`**
  DB is missing files. During catch-up, `_is_indexing` is `True`.
  After completion, `_is_indexing` is `False`.

- **`test_catchup_does_not_set_is_indexing_when_no_work`**
  Index is complete. `_is_indexing` remains `False` throughout.

- **`test_catchup_progress_tracking`**
  With 3 missing files, `_progress` updates from 0 to 1 as files are
  processed.

- **`test_catchup_already_in_progress_raises`**
  `_is_indexing` is already `True`. `start_catchup_index()` raises
  `IndexingInProgressError`.

- **`test_catchup_background_thread`**
  `start_catchup_index(background=True)` runs in a background thread.
  `wait_for_completion()` returns `True` after it finishes.

- **`test_catchup_individual_file_failure_continues`**
  One file fails to index (e.g., parse error). Catch-up logs a warning
  and continues with remaining files.

- **`test_catchup_status_shows_indexing_type`**
  During catch-up with work, `get_status()` returns
  `indexing_type="catchup"` and `files_to_process` equal to the count of
  missing + modified files.

- **`test_catchup_noop_status_shows_no_indexing_type`**
  When catch-up finds nothing to do, `get_status()` returns
  `indexing_type=None` and `files_to_process=None`.

- **`test_full_index_status_shows_indexing_type`**
  During `start_full_index()`, `get_status()` returns
  `indexing_type="full"` and `files_to_process` equal to total file count.

- **`test_status_cleared_after_indexing`**
  After both full and catch-up indexing complete, `indexing_type` and
  `files_to_process` are both `None`.

### File: `tests/test_server.py` (additions)

- **`test_startup_always_spawns_catchup_thread`**
  Both empty and non-empty DBs result in a background thread being spawned.
  No bifurcation on `total_chunks`.

- **`test_startup_catchup_starts_watcher_after_completion`**
  Catch-up thread completes. Watcher is started afterward (if
  `auto_watch` is enabled).

- **`test_startup_catchup_failure_still_starts_watcher`**
  Catch-up thread raises an exception. Watcher is still started.

- **`test_startup_embedding_model_match_proceeds`**
  Stored model matches config model. Server initializes normally.

- **`test_startup_embedding_model_mismatch_refuses_to_start`**
  Stored model is `"model-a"`, config model is `"model-b"`.
  `EmbeCodeServer.__init__` raises `EmbeddingModelChangedError` with a
  message containing both model names and the DB path.

- **`test_startup_no_stored_model_stores_current`**
  First run (no metadata). Server stores current model in DB metadata
  and proceeds normally.

- **`test_startup_stores_model_before_catchup`**
  Model is stored during `__init__` (main thread), not after catch-up
  (background thread). This ensures interrupted runs still have the model
  recorded.
