# Spec: Granular Search Timing

## Goal

Add timing instrumentation to the `search_code` pipeline so that every query
logs the duration of each phase to stderr. This lets developers identify
bottlenecks (embedding generation vs. vector search vs. BM25 vs. RRF fusion)
without attaching a profiler or modifying tool output.

---

## Problem

The search pipeline flows through four layers — `server.py` tool handler,
`Searcher` methods, `Embedder.embed()`, and `Database` queries — but none of
them log execution time. When a query feels slow there is no way to know
whether the bottleneck is embedding generation, DuckDB vector search, DuckDB
FTS, or fusion logic without manually adding `time.perf_counter()` calls and
re-deploying.

---

## Scope

- **Add a `SearchTimings` dataclass** to `searcher.py` to capture per-phase
  durations.
- **Add a `SearchResponse` dataclass** to `searcher.py` to bundle results
  with timings.
- **Instrument `Searcher.search()`** and its internal methods to measure time
  for each phase.
- **Log timings** at `INFO` level to stderr after each search completes.
- **No changes to MCP tool response shape.** The `search_code` tool continues
  to return `list[dict]`. Timings are internal-only.
- **Out of scope**: persisting timing data to DB, aggregating histograms,
  returning timings to the client, or adding a dedicated timing MCP tool.

---

## Design

### `SearchTimings` Dataclass

New dataclass in `searcher.py`:

```python
@dataclass
class SearchTimings:
    """Per-phase timing breakdown for a search query."""

    embedding_ms: float = 0.0
    vector_search_ms: float = 0.0
    bm25_search_ms: float = 0.0
    fusion_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "embedding_ms": round(self.embedding_ms, 2),
            "vector_search_ms": round(self.vector_search_ms, 2),
            "bm25_search_ms": round(self.bm25_search_ms, 2),
            "fusion_ms": round(self.fusion_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }
```

Not all fields are populated for every mode:

- **`semantic`**: `embedding_ms`, `vector_search_ms`, `total_ms`
- **`keyword`**: `bm25_search_ms`, `total_ms`
- **`hybrid`**: all fields

### `SearchResponse` Dataclass

New dataclass in `searcher.py`:

```python
@dataclass
class SearchResponse:
    """Search results with timing breakdown (internal use only)."""

    results: list[ChunkResult]
    timings: SearchTimings
```

`Searcher.search()` returns `SearchResponse` instead of `list[ChunkResult]`.

### Instrumentation Points

All timing uses `time.perf_counter()` for monotonic, high-resolution
measurement.

**`_search_semantic()`** (`searcher.py:121`):

```python
def _search_semantic(self, query, top_k, path) -> SearchResponse:
    timings = SearchTimings()
    t0 = time.perf_counter()

    # Phase: embedding
    t = time.perf_counter()
    query_embedding = self.embedder.embed([query])[0]
    timings.embedding_ms = (time.perf_counter() - t) * 1000

    # Phase: vector search
    t = time.perf_counter()
    results = self.db.vector_search(...)
    timings.vector_search_ms = (time.perf_counter() - t) * 1000

    timings.total_ms = (time.perf_counter() - t0) * 1000
    return SearchResponse(results=[...], timings=timings)
```

**`_search_keyword()`** (`searcher.py:161`):

```python
def _search_keyword(self, query, top_k, path) -> SearchResponse:
    timings = SearchTimings()
    t0 = time.perf_counter()

    # Phase: BM25 search
    t = time.perf_counter()
    results = self.db.bm25_search(...)
    timings.bm25_search_ms = (time.perf_counter() - t) * 1000

    timings.total_ms = (time.perf_counter() - t0) * 1000
    return SearchResponse(results=[...], timings=timings)
```

**`_search_hybrid()`** (`searcher.py:198`):

```python
def _search_hybrid(self, query, top_k, path) -> SearchResponse:
    timings = SearchTimings()
    t0 = time.perf_counter()

    # Reuse timings from sub-searches
    sem_response = self._search_semantic(query, fetch_k, path)
    timings.embedding_ms = sem_response.timings.embedding_ms
    timings.vector_search_ms = sem_response.timings.vector_search_ms

    kw_response = self._search_keyword(query, fetch_k, path)
    timings.bm25_search_ms = kw_response.timings.bm25_search_ms

    # Phase: RRF fusion
    t = time.perf_counter()
    # ... existing fusion logic using sem_response.results, kw_response.results ...
    timings.fusion_ms = (time.perf_counter() - t) * 1000

    timings.total_ms = (time.perf_counter() - t0) * 1000
    return SearchResponse(results=fused_results, timings=timings)
```

**`search()`** (`searcher.py:75`):

The top-level method delegates to the mode-specific method, logs timings,
and returns the `SearchResponse`:

```python
def search(self, query, mode, top_k, path) -> SearchResponse:
    # ... validation, index-ready check ...
    response = self._search_{mode}(query, top_k, path)
    logger.info(
        "search query=%r mode=%s %s",
        query, mode, response.timings.to_dict(),
    )
    return response
```

### Server Layer — No Response Shape Change

**`EmbeCodeServer.search_code()`** (`server.py:126`):

Unwraps `SearchResponse` and returns only the results list — same return
type as today:

```python
def search_code(self, query, mode, top_k, path) -> list[dict[str, Any]]:
    response = self.searcher.search(query, mode=mode, top_k=top_k, path=path)
    return [result.to_dict() for result in response.results]
```

**MCP `search_code` tool** (`server.py:216`):

No changes. It calls `server.search_code()` and returns `list[dict]` as
before. Timings never appear in the MCP response.

---

## Logging Format

Timings are logged at `INFO` level on stderr (MCP convention: stdout is
reserved for the protocol). Example log line:

```
2026-02-26 10:15:32 - embecode.searcher - INFO - search query='database connection' mode=hybrid {'embedding_ms': 45.23, 'vector_search_ms': 12.87, 'bm25_search_ms': 8.41, 'fusion_ms': 0.15, 'total_ms': 67.12}
```

Single log line per query. Fields with `0.0` (unused phases) are still
included for consistent structure.

---

## Edge Cases

### First query cold start

The first semantic/hybrid query may be slower because the embedding model is
loaded lazily (`Embedder._model` is `None` until first `embed()` call, per
`embedder.py:40`). The `embedding_ms` field captures this naturally — it will
be higher on the first call. No special handling needed.

### Index not ready

When `IndexNotReadyError` is raised, no timings are generated. The error is
raised before any timed work begins. No logging occurs.

### Zero-result queries

Timings are still populated and logged even if the query returns no results.
All phases execute normally.

### Keyword mode skips embedding

When `mode="keyword"`, `embedding_ms` and `vector_search_ms` remain `0.0`.
This is correct — those phases don't run.

---

## Files Changed

| File | Change |
|---|---|
| `src/embecode/searcher.py` | Add `SearchTimings`, `SearchResponse` dataclasses. Add `import time, logging`. Instrument `search()`, `_search_semantic()`, `_search_keyword()`, `_search_hybrid()`. Change return types from `list[ChunkResult]` to `SearchResponse`. Log timings in `search()`. |
| `src/embecode/server.py` | Update `EmbeCodeServer.search_code()` to unwrap `SearchResponse.results`. No change to MCP tool or its return type. |
| `tests/test_searcher.py` | Update ~15 existing test call sites to access `.results` on returned `SearchResponse`. Add new timing tests. |
| `tests/test_integration.py` | Update ~14 call sites to access `.results` on returned `SearchResponse`. |
| `tests/test_performance.py` | Update ~3 call sites to access `.results` on returned `SearchResponse`. |
| `tests/test_edge_cases.py` | Update ~5 call sites to access `.results` on returned `SearchResponse`. |
| `tests/test_server.py` | No changes expected (server return shape unchanged). |

---

## Testing Requirements

### File: `tests/test_searcher.py` (additions)

- **`test_search_returns_search_response`**
  `Searcher.search()` returns a `SearchResponse` with `.results`
  (`list[ChunkResult]`) and `.timings` (`SearchTimings`) attributes.

- **`test_timings_hybrid_has_all_phases`**
  Hybrid search populates `embedding_ms`, `vector_search_ms`,
  `bm25_search_ms`, `fusion_ms`, and `total_ms` — all > 0.

- **`test_timings_semantic_has_embedding_and_vector`**
  Semantic search populates `embedding_ms` and `vector_search_ms` > 0.
  `bm25_search_ms` and `fusion_ms` are 0.

- **`test_timings_keyword_has_bm25_only`**
  Keyword search populates `bm25_search_ms` > 0. `embedding_ms`,
  `vector_search_ms`, and `fusion_ms` are 0.

- **`test_timings_total_gte_sum_of_parts`**
  `total_ms >= embedding_ms + vector_search_ms + bm25_search_ms + fusion_ms`
  (total includes minor overhead like result construction).

- **`test_timings_to_dict_rounds_to_two_decimals`**
  `SearchTimings.to_dict()` values are rounded to 2 decimal places.

- **`test_timings_logged_at_info_level`**
  After calling `search()`, verify that a log record at `INFO` level was
  emitted containing the query text, mode, and timing dict. Use `caplog` or
  a mock logger.

### File: `tests/test_searcher.py` (updates to existing tests)

All existing tests that call `Searcher.search()` and assert on the result
list are updated to access `.results` on the returned `SearchResponse`. For
example:

```python
# Before
results = searcher.search("query", mode="hybrid")
assert len(results) == 5

# After
response = searcher.search("query", mode="hybrid")
assert len(response.results) == 5
```
