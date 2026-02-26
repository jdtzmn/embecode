# Spec: Concise Search Output

## Goal

The `search_code` MCP tool should return concise, navigable results by default
instead of dumping full chunk source code. Results should help the caller
identify the right file and line range without consuming excessive context
window.

---

## Problem

The current `search_code` tool returns full chunk content in every result.
With `top_k=10` and chunks up to ~1500 non-whitespace characters each (more
with whitespace), a single query can return 30-80KB+ of raw source. This
causes three concrete problems:

1. **Context window waste**: 84KB of output for a navigational query that
   really just needed file paths and line ranges. The caller never reads
   the full source inline — they use `Read` or `Glob` to fetch the file.
2. **Truncation**: MCP clients truncate large responses, so the caller
   can't even see the results. The truncation then triggers workaround
   suggestions (e.g., delegate to an agent), adding unnecessary friction.
3. **Ineffective tool**: The search becomes overhead rather than help. The
   caller ends up using glob/grep to get the same information more
   efficiently.

---

## Scope

- **Change `search_code` output format**: return concise summaries instead
  of full chunk content.
- **Enrich chunk context at index time**: extract definition names (function,
  class, etc.) from AST nodes during chunking and store them as metadata.
- **Wire `SearchConfig.top_k` from config**: the config value is currently
  dead code; connect it to the MCP tool's default.
- **Update PLAN.md**: the plan says "Returns full chunk text inline (not
  snippets)" — update to reflect the new default.
- **Out of scope**: adding a `detail` parameter or separate `read_chunk` tool.
  Full content is removed from the search response entirely.

---

## Design

### New Search Output Format

Each result in the `search_code` response contains:

```json
{
  "file_path": "src/embecode/searcher.py",
  "language": "python",
  "start_line": 21,
  "end_line": 43,
  "definitions": "class ChunkResult, method to_dict",
  "preview": "class ChunkResult:\n    \"\"\"A search result containing a code chunk with metadata and score.\"\"\"",
  "score": 0.032
}
```

Field descriptions:

- **`file_path`**: Unchanged.
- **`language`**: Unchanged.
- **`start_line`** / **`end_line`**: Unchanged.
- **`definitions`**: Comma-separated list of named definitions in the chunk
  (e.g., `"function search, class Searcher"`). Extracted from AST nodes at
  index time. May be empty string for chunks with no named definitions
  (e.g., top-level imports, config blocks).
- **`preview`**: The first 2 non-empty lines of the chunk content. Gives the
  caller enough signal to identify the chunk without reading the full source.
  Capped at 200 characters total.
- **`score`**: Unchanged.

### Fields Removed

- **`content`**: The full chunk source code is no longer returned. This was
  the primary source of bloated responses.
- **`context`**: This field only contained `"File: path\nLanguage: lang"`,
  duplicating `file_path` and `language`. It is replaced by `definitions`.

### Response Size

With the new format, each result is approximately 200-400 bytes. A `top_k=10`
query produces ~2-4KB total — a 20-40x reduction from the current output.

---

## Definition Extraction

### Approach

At chunk creation time, inspect the AST nodes being merged and extract
the names of any definitions they contain. This uses tree-sitter's
`node.type` and `node.child_by_field_name("name")` API.

### Definition Node Types

The following node types are recognized as named definitions:

| Language | Node type | Label |
|---|---|---|
| Python | `function_definition` | `function` |
| Python | `class_definition` | `class` |
| Python | `decorated_definition` | (unwrap to inner def/class) |
| TypeScript/TSX | `function_declaration` | `function` |
| TypeScript/TSX | `class_declaration` | `class` |
| TypeScript/TSX | `interface_declaration` | `interface` |
| TypeScript/TSX | `type_alias_declaration` | `type` |
| TypeScript/TSX | `enum_declaration` | `enum` |
| TypeScript/TSX | `method_definition` | `method` |
| JavaScript | `function_declaration` | `function` |
| JavaScript | `class_declaration` | `class` |
| JavaScript | `method_definition` | `method` |
| Go | `function_declaration` | `function` |
| Go | `method_declaration` | `method` |
| Go | `type_declaration` | `type` |
| Rust | `function_item` | `function` |
| Rust | `struct_item` | `struct` |
| Rust | `enum_item` | `enum` |
| Rust | `impl_item` | `impl` |
| Rust | `trait_item` | `trait` |
| Java | `class_declaration` | `class` |
| Java | `interface_declaration` | `interface` |
| Java | `method_declaration` | `method` |
| Java | `enum_declaration` | `enum` |

For languages not in this table, `definitions` is empty string. New
languages can be added later by extending the lookup table.

### Extraction Logic

A new function `_extract_definition_names(nodes, language)` walks the list
of AST nodes being merged into a chunk:

1. For each node, check if `node.type` is in the language's definition
   table.
2. If yes, get `node.child_by_field_name("name")` and decode the name.
3. Format as `"{label} {name}"` (e.g., `"function search"`).
4. Special case: Python `decorated_definition` — inspect the inner child
   node (the actual function/class definition) for the name.
5. Special case: Go `type_declaration` — the name is on the `type_spec`
   child node, not directly on `type_declaration`.
6. Special case: Rust `impl_item` — the name is in the `type` field, not
   `name`.
7. Return the list of definition strings.

When recursion splits a large node into its children (line 208 of
`chunker.py`), the children may include method definitions inside a class.
These are captured naturally since we inspect every node being merged,
not just top-level definitions.

### Storage

The extracted definitions are stored in the existing `context` column of
the `chunks` table. The format changes from:

```
File: src/foo.py
Language: python
```

to:

```
File: src/foo.py
Language: python
Defines: function search, class Searcher
```

The `Defines:` line is only present when definitions is non-empty. This
preserves backward compatibility with the context format used for embedding
enrichment — the definitions line adds semantic signal that improves
retrieval quality.

The `Chunk` dataclass gains a new field:

```python
@dataclass
class Chunk:
    content: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    context: str
    hash: str
    definitions: str  # NEW: comma-separated definition names
```

`definitions` is computed by `_extract_definition_names()` and passed into
`Chunk.create()`. It is also appended to `context` as the `Defines:` line.

### Database Schema Change

The `chunks` table gains a new column:

```sql
ALTER TABLE chunks ADD COLUMN definitions TEXT NOT NULL DEFAULT ''
```

This is handled as a schema migration in `_initialize_schema()`. If the
column already exists (on an existing database), the `ALTER TABLE` is
skipped. New chunks store definitions; existing chunks have empty string
until re-indexed.

The `insert_chunks` method is updated to write the `definitions` field.
Search query methods (`vector_search`, `bm25_search`,
`_fallback_keyword_search`) are updated to read the `definitions` column.

---

## Preview Generation

The `preview` field is computed at search time (not stored in DB) from the
chunk's `content`:

1. Split `content` into lines.
2. Filter out empty/whitespace-only lines.
3. Take the first 2 non-empty lines.
4. Join with `\n`.
5. Truncate to 200 characters total. If truncated, append `...`.

This is implemented as a method on `ChunkResult`:

```python
def preview(self) -> str:
    """Generate a 2-line preview from chunk content."""
    lines = [l for l in self.content.splitlines() if l.strip()]
    preview = "\n".join(lines[:2])
    if len(preview) > 200:
        return preview[:197] + "..."
    return preview
```

---

## ChunkResult Changes

The `ChunkResult` dataclass is updated:

```python
@dataclass
class ChunkResult:
    content: str        # Still needed internally for preview generation
    file_path: str
    language: str
    start_line: int
    end_line: int
    definitions: str    # CHANGED from context
    score: float

    def to_dict(self) -> dict:
        """Convert to concise API response (no full content)."""
        lines = [l for l in self.content.splitlines() if l.strip()]
        preview = "\n".join(lines[:2])
        if len(preview) > 200:
            preview = preview[:197] + "..."

        return {
            "file_path": self.file_path,
            "language": self.language,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "definitions": self.definitions,
            "preview": preview,
            "score": self.score,
        }
```

`content` remains on the dataclass for internal use (preview generation,
potential future use) but is never serialized to the MCP response.

---

## Config Wiring

### Problem

`SearchConfig.top_k = 10` is defined in config.py but never used. The MCP
tool function hardcodes `top_k: int = 5`.

### Solution

The MCP tool signature changes from:

```python
def search_code(query, mode="hybrid", top_k=5, path=None)
```

to:

```python
def search_code(query, mode="hybrid", top_k=10, path=None)
```

The default is changed from 5 to 10. With concise output, 10 results cost
~2-4KB — less than what 5 results cost before.

---

## Database Query Changes

The `vector_search`, `bm25_search`, and `_fallback_keyword_search` methods
in `db.py` currently `SELECT c.content` in their queries. They are updated
to also `SELECT c.definitions`.

The returned dicts gain a `definitions` key alongside the existing fields.

---

## Embedding Enrichment

The `context` field prepended to content before embedding (in
`indexer.py`) now includes definition names:

```
File: src/embecode/searcher.py
Language: python
Defines: class ChunkResult, method to_dict, class Searcher
```

This improves semantic search quality — a query for "ChunkResult" will
score higher on chunks that define it, because the definition name appears
in the enrichment text that was embedded.

---

## Re-indexing Requirement

Existing indexes do not have the `definitions` column populated. Two
behaviors:

1. **New databases**: definition extraction runs automatically during
   indexing. No action needed.
2. **Existing databases**: the `definitions` column is added with default
   `''`. Search works correctly — `definitions` is empty string and
   `preview` is generated from the stored `content`. To get full benefit
   of definition-enriched search, the user must re-index (delete the cache
   and restart).

No forced re-index is triggered. This is a graceful degradation — the
feature works partially on old indexes and fully on new ones.

---

## PLAN.md Update

Line 101 currently reads:

> Returns full chunk text inline (not snippets) with file path, language,
> line range, and relevance score. Balances context window usage vs.
> round-trip cost — default `top_k=5` keeps context lean.

Update to:

> Returns concise results with file path, language, line range, definition
> names, a 2-line preview, and relevance score. Default `top_k=10`.

---

## Edge Cases

### Chunks with no definitions

Some chunks contain only imports, constants, or top-level statements with
no named definitions. `definitions` is empty string. `preview` still shows
the first 2 lines, which typically indicate what the chunk contains (e.g.,
`from typing import ...` or `LANGUAGE_MAP = {`).

### Very short chunks

A chunk with a single line produces a single-line preview. No padding or
special handling needed.

### Binary/non-parseable files

Files that fall back to single-chunk mode (no tree-sitter parser available)
have empty `definitions`. Preview is generated normally from content.

### `decorated_definition` in Python

A decorated function like:

```python
@app.route("/foo")
def handler():
    ...
```

Produces a `decorated_definition` node. The extraction logic unwraps it
to find the inner `function_definition` and extracts `"function handler"`.

### Content still stored in DB

Full chunk content remains stored in the `chunks` table — it's needed for
embedding generation, FTS indexing, and preview computation. Only the MCP
tool response omits it.

---

## Testing Requirements

### File: `tests/test_chunker.py` (additions)

#### Class: `TestDefinitionExtraction`

- **`test_extract_python_function_definition`**
  Python source with `def foo():`. Chunk `definitions` contains
  `"function foo"`.

- **`test_extract_python_class_definition`**
  Python source with `class Bar:`. Chunk `definitions` contains
  `"class Bar"`.

- **`test_extract_python_decorated_function`**
  Python source with `@decorator\ndef baz():`. Chunk `definitions`
  contains `"function baz"`.

- **`test_extract_python_decorated_class`**
  Python source with `@decorator\nclass Qux:`. Chunk `definitions`
  contains `"class Qux"`.

- **`test_extract_multiple_definitions`**
  Python source with a class containing two methods. Chunk `definitions`
  lists all named definitions found.

- **`test_extract_typescript_function`**
  TypeScript source with `function search() {}`. Chunk `definitions`
  contains `"function search"`.

- **`test_extract_typescript_class`**
  TypeScript source with `class Foo {}`. Chunk `definitions` contains
  `"class Foo"`.

- **`test_extract_typescript_interface`**
  TypeScript source with `interface Bar {}`. Chunk `definitions` contains
  `"interface Bar"`.

- **`test_extract_javascript_function`**
  JavaScript source with `function init() {}`. Chunk `definitions`
  contains `"function init"`.

- **`test_extract_no_definitions`**
  Python source with only imports and constants. Chunk `definitions`
  is empty string.

- **`test_extract_unsupported_language`**
  TOML or JSON file. Chunk `definitions` is empty string.

- **`test_definitions_in_context`**
  Chunk `context` field includes `"Defines: function foo"` line when
  definitions are present.

- **`test_no_defines_line_when_empty`**
  Chunk `context` field does NOT include a `"Defines:"` line when
  definitions is empty.

### File: `tests/test_searcher.py` (updates)

- **`test_result_to_dict_concise_format`**
  `ChunkResult.to_dict()` returns keys: `file_path`, `language`,
  `start_line`, `end_line`, `definitions`, `preview`, `score`.
  Does NOT contain `content` or `context`.

- **`test_result_preview_two_lines`**
  Content with 10 lines. Preview contains only the first 2 non-empty
  lines.

- **`test_result_preview_skips_empty_lines`**
  Content starts with blank lines. Preview skips them and shows the
  first 2 non-empty lines.

- **`test_result_preview_truncated_at_200_chars`**
  Content with very long first line (>200 chars). Preview is truncated
  to 200 chars with `...` appended.

- **`test_result_preview_single_line_chunk`**
  Content with only one non-empty line. Preview is that single line.

### File: `tests/test_server.py` (updates)

- **`test_search_code_returns_concise_results`**
  Call `search_code` via the MCP tool. Each result has `file_path`,
  `language`, `start_line`, `end_line`, `definitions`, `preview`,
  `score`. No `content` field.

- **`test_search_code_default_top_k_is_10`**
  Call `search_code` without explicit `top_k`. Verify up to 10 results
  are returned.

### File: `tests/test_db.py` (additions)

- **`test_chunks_table_has_definitions_column`**
  After `connect()`, the `definitions` column exists in the `chunks`
  table schema.

- **`test_insert_chunk_with_definitions`**
  Insert a chunk with `definitions="function foo, class Bar"`. Retrieve
  it and verify the field is stored correctly.

- **`test_vector_search_returns_definitions`**
  Perform a vector search. Results include the `definitions` key.

- **`test_bm25_search_returns_definitions`**
  Perform a BM25 search. Results include the `definitions` key.

- **`test_existing_db_migration_adds_definitions`**
  Open an existing DB without the `definitions` column. Re-run
  `_initialize_schema()`. Column is added with default `''`.

### File: `tests/test_config.py` (updates)

- **`test_search_top_k_default_is_10`**
  Default `SearchConfig` has `top_k = 10`.
