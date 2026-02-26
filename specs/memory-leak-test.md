# Spec: Memory Leak Detection Test

## Goal

Embecode consumes unbounded memory during full indexing — observed growing to
30 GB on a large codebase. Before fixing the root cause, a live test must be
written that reliably catches the regression. The test must fail on the current
codebase and pass only after the leak is fixed.

---

## Scope

- One new test file: `tests/test_memory.py`
- One new pytest marker: `memory`
- One new dev dependency: `psutil`
- No changes to production code in this phase

---

## Test File: `tests/test_memory.py`

### Markers and skipping

The test is marked `@pytest.mark.slow` and `@pytest.mark.memory`. It can be
excluded from the default run with `-m "not slow"` or `-m "not memory"`.

The test requires `psutil`. If `psutil` is not importable, the test is skipped
with a clear message rather than erroring.

### Synthetic codebase

The test generates approximately 2 000 source files procedurally into a
`tmp_path` temporary directory. The generation must not itself be a significant
memory consumer — files are written to disk and not held in memory as a
collection.

File distribution:

| Language | Count | File size |
|---|---|---|
| Python (`.py`) | 1 200 | ~150–300 lines each |
| TypeScript (`.ts`) | 400 | ~100–200 lines each |
| JavaScript (`.js`) | 400 | ~80–150 lines each |

Directory structure uses at least four levels of nesting (e.g.
`src/core/auth/models/`) to exercise the directory-walking and `.gitignore`
cache paths under realistic conditions.

Each generated file contains valid syntax for its language: class definitions,
function definitions, imports, and inline comments. The content does not need to
be semantically meaningful — it just must not be rejected by the tree-sitter
parser (i.e. no intentional parse errors).

File content is varied enough that the chunker does not produce identical hashes
across all files (avoiding artificial deduplication that would mask memory
behaviour).

### Components under test

| Component | Configuration |
|---|---|
| `Indexer` | Real implementation from `embecode.indexer` |
| `Database` | Real `embecode.db.Database` writing to a `tmp_path` `.duckdb` file |
| `Embedder` | **Mock**: returns a list of `[0.0] * 768` per text, never loads a model |
| Config | Loaded via `load_config()` with `include = []` and `exclude` defaults |

Using the real `Database` is essential because DuckDB's C-level memory and
connection lifecycle are the most plausible leak site at the scale observed.

### Memory measurement

Memory is sampled using `psutil.Process(os.getpid()).memory_info().rss`, which
returns the Resident Set Size in bytes. This captures C-extension memory
(DuckDB, tree-sitter) that Python's `tracemalloc` would miss.

Sampling strategy:

1. **Baseline** — record RSS immediately before `indexer.start_full_index()` is
   called. This is after all imports and setup are complete.
2. **Peak** — poll RSS every 2 seconds in a background thread for the duration
   of indexing, recording the maximum observed value.
3. **Final** — record RSS once `start_full_index` returns (foreground mode).

The test asserts that both peak RSS and final RSS remain below **2.0 GB**.

### Pass/fail criterion

The test fails if either of the following is true:

- Peak RSS during indexing exceeds 2.0 GB
- Final RSS after indexing completes exceeds 2.0 GB

The test passes when both values stay below 2.0 GB for the full 2 000-file run.

### Printed diagnostics

Regardless of pass or fail, the test prints a human-readable summary to stdout
(visible with `pytest -s`) including:

- Total files generated
- Total files indexed (from `indexer.get_status()`)
- Total chunks stored (from `db.get_index_stats()`)
- Baseline RSS in MB
- Peak RSS in MB (and which file index approximately triggered it, if
  determinable)
- Final RSS in MB
- Wall-clock duration of indexing

This output is the primary tool for diagnosing the leak once the test is
catching it.

### Test must catch the current bug

The test must be validated (before any fix is applied) to confirm it actually
fails. The spec is complete only when the test fails on the unfixed codebase.
Do not merge the test alongside a fix — the test comes first, confirmed red,
then the fix brings it green.

---

## Dependency Change

`psutil` is added to `[dependency-groups] dev` in `pyproject.toml`. It must not
appear in `[project] dependencies`.

---

## Pytest Marker Registration

The `memory` marker is registered in `[tool.pytest.ini_options] markers` in
`pyproject.toml` with the description:
`"memory: marks tests that measure process memory (deselect with '-m \"not memory\"')"`.

---

## What the test does NOT cover

- The real `sentence-transformers` embedder (model download not required)
- Incremental / watcher-triggered indexing (full index only)
- Search query memory behaviour
- Multi-process or multi-threaded indexing scenarios
- Fixing the leak (that is a separate phase)
