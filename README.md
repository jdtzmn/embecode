# embecode

> Local-first MCP server for semantic + keyword hybrid code search. Zero external services. No API keys required.

[![CI](https://github.com/jdtzmn/embecode/actions/workflows/ci.yml/badge.svg)](https://github.com/jdtzmn/embecode/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/embecode)](https://pypi.org/project/embecode/)
[![Python](https://img.shields.io/pypi/pyversions/embecode)](https://pypi.org/project/embecode/)

## Usage

```bash
# From your project root
uvx embecode

# Or with an explicit path
uvx embecode --path /path/to/repo
```

Add to your MCP client config (Claude Desktop, Cursor, Cline, etc.):

```json
{
  "mcpServers": {
    "embecode": {
      "command": "uvx",
      "args": ["embecode"]
    }
  }
}
```

## Tools

| Tool | Description |
|---|---|
| `search_code` | Hybrid semantic + keyword search over your codebase |
| `index_status` | Check indexing progress, file count, and last updated time |

## How it works

- Parses files into AST chunks via **tree-sitter** (cAST algorithm)
- Embeds chunks locally with **sentence-transformers** (`nomic-embed-text-v1.5`)
- Stores vectors + FTS index in a single **DuckDB** file at `~/.cache/embecode/`
- Fuses BM25 and dense vector results with **Reciprocal Rank Fusion**
- Watches for file changes via **watchfiles** and re-indexes incrementally

## Development

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

### Benchmarks

Two benchmark classes live in `tests/test_performance.py` and use [pytest-benchmark](https://pytest-benchmark.readthedocs.io/):

| Class | DB | What it measures |
|---|---|---|
| `TestSearchBenchmark` | Mock (in-memory dict) | `Searcher` + RRF code path only — no real DB or model |
| `TestSearchBenchmarkReal` | Real DuckDB (VSS + FTS) | Actual query latency: cosine-similarity scan, BM25, and fusion |

**Run the real benchmarks:**

```bash
pytest tests/test_performance.py::TestSearchBenchmarkReal -v --benchmark-only --no-cov -s
```

The first run builds a 200-file synthetic index into `.bench_db/` (~20s). Subsequent runs reuse it and start immediately. Delete `.bench_db/` to force a rebuild.

**Run the mock benchmarks** (no setup cost, useful for isolating Searcher logic overhead):

```bash
pytest tests/test_performance.py::TestSearchBenchmark -v --benchmark-only --no-cov -s
```

**Reading the output:**

Each test prints a per-phase timing breakdown from `SearchTimings` on the last benchmark round:

```
phase breakdown (last run): {'embedding_ms': 0.0, 'vector_search_ms': 78.5, 'bm25_search_ms': 6.5, 'fusion_ms': 0.01, 'total_ms': 85.0}
```

pytest-benchmark then prints a summary table with min, max, mean, median, and stddev across all rounds.

Requires Python 3.12.
