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

Requires Python 3.12.
