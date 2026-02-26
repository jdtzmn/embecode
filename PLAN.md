# embecode — MCP Code Search Server: Project Plan

## Overview

A local-first MCP server that indexes a codebase and exposes semantic + keyword hybrid search to AI assistants (Claude, Cursor, Cline, etc.). Zero external services, no API keys required, single-command setup via `uvx`.

---

## Goals

- Dead-simple setup: `uvx embecode` — defaults to current working directory, `--path` optional override
- No daemons to manage manually — watcher runs inside the MCP server process
- No footprint in the project repo (cache stored in `~/.cache/embecode/`)
- Local embeddings — no API key required (optional upgrade path available)
- Project-level config via `.embecode.toml` committed to the repo
- PyPI-published from day one so `uvx` works out of the box

---

## Tech Stack

| Layer | Library | Notes |
|---|---|---|
| MCP server | `fastmcp` | Higher-level abstraction over official MCP SDK |
| Parsing | `tree-sitter` + `tree-sitter-languages` | Universal AST parsing, all major languages |
| Chunking | cAST algorithm | Custom implementation (see Algorithms section) |
| Embeddings | `sentence-transformers` | Runs locally, no API key |
| Default model | `nomic-embed-text-v1.5` | 8192-token context, best quality/speed balance |
| Fallback model | `BAAI/bge-base-en-v1.5` | Faster, smaller |
| Vector store | `duckdb` + VSS extension | Single-file, no daemon |
| BM25 search | `duckdb` FTS extension | Native SQL, no extra deps |
| Fusion | Reciprocal Rank Fusion (RRF) | No tuning required |
| File watching | `watchfiles` | Rust-backed, fast, simple API |
| Config parsing | `tomllib` (stdlib 3.11+) / `tomli` fallback | Zero extra deps on 3.11+ |

---

## Algorithms

### Chunking: cAST

Parse each file into an AST via tree-sitter, then apply recursive split-then-merge:

1. Walk the AST nodes
2. Greedily merge sibling nodes into a chunk until the size budget is exceeded (measured in non-whitespace characters, not lines)
3. If a single node exceeds the budget, recurse into its children
4. Chunk boundaries always align with complete syntactic units (no splitting a function in half)
5. All chunks concatenated must reconstruct the original file verbatim

Per-language chunk size defaults (non-whitespace chars):
- Python: 1500
- TypeScript/JavaScript: 1200
- All others: 1000

**Chunk enrichment:** Prepend contextual metadata to each chunk before embedding:
- File path
- Class/module scope
- What the chunk defines (function name, class name, etc.)
- Key imports visible to this chunk

This significantly improves embedding quality for code.

### Search: Hybrid BM25 + Dense Vector with RRF

Two parallel search legs:
1. **BM25 (lexical):** DuckDB FTS extension. Best for exact identifier matches, function names, variable names.
2. **Dense vector (semantic):** DuckDB VSS extension with cosine similarity. Best for natural language queries.

Results from both legs are fused using **Reciprocal Rank Fusion (RRF)**:
```
RRF_score(d) = Σ 1 / (k + rank(d))   where k=60
```

No normalization or hyperparameter tuning needed.

### Vector Index

- **Default:** Flat brute-force index (exact, 100% recall). Sufficient for all typical codebases (<100k chunks).
- **Future:** HNSW via DuckDB VSS for monorepos exceeding ~500k chunks.

### Incremental Indexing

- Hash each chunk's content (SHA1)
- On file change: re-parse the file, diff chunks by hash, delete stale embeddings, insert only new ones
- On branch switch: re-index only files that changed between branches

---

## MCP Tools Exposed

### `search_code`
```python
search_code(
    query: str,
    mode: str = "hybrid",     # "semantic" | "keyword" | "hybrid"
    top_k: int = 5,
    path: str | None = None   # prefix filter e.g. "apps/ui"
) -> list[ChunkResult]
```

Returns full chunk text inline (not snippets) with file path, language, line range, and relevance score. Balances context window usage vs. round-trip cost — default `top_k=5` keeps context lean.

### `index_status`
```python
index_status() -> IndexStatus
```

Returns: files indexed, total chunks, embedding model in use, last updated timestamp, whether indexing is currently in progress.

---

## First-Run Behavior

On first activation with an empty cache:

1. MCP server starts and becomes available immediately (non-blocking)
2. Full index begins in a background thread:
   - Walk file tree respecting include/exclude rules
   - Parse + chunk each file via cAST
   - Load embedding model on first use (lazy load)
   - Write chunks + embeddings to DuckDB
3. File watcher starts after full index completes
4. Any tool call during indexing returns a "not ready" response with progress:
   ```
   Index is still being built (247/1,840 files processed). Try again in ~30s.
   ```
   Partial results are intentionally not returned — incomplete index could produce misleading "not found" answers.

---

## Cache Architecture

Index stored at `~/.cache/embecode/<hash>/` where `<hash>` is the first 8 chars of SHA1 of the absolute project path.

```
~/.cache/embecode/
  registry.json          # metadata for all cached projects
  a3f9b2c1/              # hash of /Users/john/projects/myapp
    index.db             # DuckDB file (chunks, embeddings, FTS index)
    daemon.lock          # PID lock file (reserved for v2 daemon)
  7e4d1a8f/
    index.db
    daemon.lock
```

### `registry.json` schema
```json
{
  "a3f9b2c1": {
    "project_path": "/Users/john/projects/myapp",
    "last_accessed": "2025-02-24T10:30:00",
    "size_bytes": 45000000
  }
}
```

`last_accessed` is updated every time the MCP server starts for that project.

### Cache Eviction

- **Size cap:** Default 2GB total across all projects (configurable in `~/.config/embecode/config.toml`)
- **Strategy:** LRU — evict least recently accessed projects first until under the cap
- **Stale detection:** On any server startup, scan registry for project paths that no longer exist on disk and delete them immediately
- Eviction runs automatically on startup, after stale detection

### CLI Cache Commands
```bash
uvx embecode cache status          # show all projects, sizes, last accessed
uvx embecode cache clean           # evict LRU entries down to size cap
uvx embecode cache purge           # delete all caches
uvx embecode cache purge .         # delete cache for current project only
```

---

## Configuration

### Project-level: `.embecode.toml` (committed to repo)
```toml
[index]
include = ["src/", "lib/", "tests/"]
exclude = ["node_modules/", "dist/", ".git/", "*.min.js", "**/__pycache__/"]

[index.languages]
python = 1500
typescript = 1200
default = 1000

[embeddings]
model = "local"   # uses nomic-embed-text-v1.5
# model = "BAAI/bge-base-en-v1.5"
# api_key_env = "VOYAGE_API_KEY"   # optional upgrade path

[search]
default_mode = "hybrid"
top_k = 10

[daemon]
debounce_ms = 500
auto_watch = true
```

### Config resolution order (highest to lowest priority)
1. CLI args (`--path`, `--exclude`)
2. `.embecode.toml` in project root
3. `~/.config/embecode/config.toml` (user-global defaults)
4. Built-in defaults

---

## Daemon / Lifecycle Architecture

### v1: Single Process (MCP server owns the watcher)

```
uvx embecode
  → read config
  → resolve cache dir (~/.cache/embecode/<hash>/)
  → open DuckDB
  → if DB empty: start full index in background thread
  → start MCP server (immediately ready)
  → after full index: start watchfiles watcher in daemon thread
  → serve tool calls

On disconnect / SIGTERM:
  → watcher thread exits (daemon=True, no cleanup needed)
  → DuckDB closes cleanly
```

The watcher thread is a Python daemon thread (`thread.daemon = True`) so the process exits cleanly without waiting for it.

### v2: Lock-File Daemon (future)

The first MCP client to connect becomes the watcher owner (writes `daemon.lock`). Subsequent clients connect read-only to the same DuckDB file. Last client to exit removes the lock. Keeps the index warm between AI sessions.

---

## Project Structure

```
embecode/
  pyproject.toml
  README.md
  .embecode.toml.example
  src/
    embecode/
      __init__.py
      cli.py              # entry point, arg parsing, uvx target
      server.py           # fastmcp server, tool definitions
      indexer.py          # orchestrates full + incremental indexing
      chunker.py          # cAST algorithm implementation
      embedder.py         # sentence-transformers wrapper, lazy load
      searcher.py         # hybrid search, RRF fusion
      watcher.py          # watchfiles wrapper, debounce logic
      cache.py            # cache dir resolution, registry, eviction
      config.py           # config loading, resolution order
      db.py               # DuckDB setup, schema, migrations
```

---

## Distribution

- Published to **PyPI** as `embecode` (or chosen name)
- Canonical usage: `uvx embecode` (from project root) or `uvx embecode --path /path/to/repo` (optional override)
- MCP config entry:
```json
{
  "mcpServers": {
    "embecode": {
      "command": "uvx",
      "args": ["embecode"]
      // optionally: "args": ["embecode", "--path", "/absolute/path/to/repo"]
    }
  }
}
```

---

## Embedding Model Upgrade Path

| Tier | Model | Requires |
|---|---|---|
| Default (local) | `nomic-embed-text-v1.5` | Nothing |
| Fast/offline | `all-MiniLM-L6-v2` | Nothing |
| Best quality | `voyage-code-3` | `VOYAGE_API_KEY` env var |
| Alternative API | `text-embedding-3-large` | `OPENAI_API_KEY` env var |

---

## Future Roadmap

- **v2:** Lock-file daemon so index stays warm between AI sessions
- **v2:** HNSW index for monorepos >500k chunks
- **v3:** Symbol index (function/class definitions + call graph) via tree-sitter queries
- **v3:** Git-aware indexing (only reindex files changed since last commit)
