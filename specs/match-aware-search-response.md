# Spec: Match-Aware Search Response

## Goal

Improve the `search_code` MCP tool response so that results highlight what
actually matched the query, rather than always showing the first 2 lines of
each chunk. Three complementary improvements:

1. **Match-aware preview**: Show lines containing query terms instead of the
   chunk's opening lines.
2. **Match line numbers**: Report which lines within a chunk matched, so the
   caller can `Read` with a precise offset.
3. **File grouping hint**: Signal when multiple results come from the same
   file.

---

## Problem

The current `preview` field always shows the first 2 non-empty lines of a
chunk. When searching for `UserAvatar`, a chunk that defines
`export function UserAvatar` on line 82 might preview as:

```
import React from 'react'
import { cn } from '@/lib/utils'
```

The caller gets no signal about *why* this chunk matched. Compare with a
grep-style tool that shows the exact matching line:

```
Line 82: export function UserAvatar({
```

Additionally, when 3 of 10 results come from the same file, there is no
signal to the caller that those results are related — the flat list treats
each result independently.

---

## Scope

- **Match-aware preview**: Update `ChunkResult.preview()` to prefer lines
  containing query terms when a query is provided.
- **`match_lines` field**: Add an optional field to the response listing
  absolute line numbers that matched query terms.
- **`file_result_count` field**: Add an optional field when multiple results
  share the same `file_path`.
- **Shared tokenizer**: Extract query tokens once and reuse for both preview
  selection and match line computation.
- **Out of scope**: Restructuring the response into a grouped-by-file format.
  The response remains a flat list of results.

---

## Design

### Query Tokenization

A new module-level function `_tokenize_query(query: str) -> list[str]`:

1. Extract all `[A-Za-z0-9_]+` tokens from the query string.
2. Lowercase each token.
3. Deduplicate (preserve order).
4. Drop single-character tokens (too noisy — `a`, `b`, etc.).
5. Return the list.

Example: `"UserAvatar component"` → `["useravatar", "component"]`

This handles both natural language queries ("authentication logic") and
symbol queries ("UserAvatar") uniformly.

### Match Line Detection

A new module-level function in `searcher.py`:

```python
def _find_match_lines(
    content: str,
    query: str,
    start_line: int,
) -> list[int]:
```

1. Tokenize the query using `_tokenize_query()`.
2. If no tokens remain (empty query, all single-char), return `[]`.
3. For each line in `content.splitlines()`:
   - Check if any query token appears as a case-insensitive substring.
   - If yes, record the absolute line number (`start_line + line_index`).
4. Return the list of matching absolute line numbers, sorted ascending.
5. Cap at 8 entries maximum to prevent bloat on large chunks.

### Match-Aware Preview

A new module-level function in `searcher.py`:

```python
def _pick_preview_lines(
    content: str,
    query: str | None = None,
) -> str:
```

**When `query` is provided and has lexical matches:**

1. Split content into lines. Build list of `(line_index, line_text,
   match_count)` for non-empty lines where `match_count > 0`.
2. Pick the **best matching line**: highest `match_count`, earliest index
   as tiebreak.
3. Pick a **second line**:
   - If another matching line exists, pick the nearest one to the best
     match.
   - Otherwise, pick the nearest non-empty line (above or below) as
     context.
4. Order the two lines by their original position in the chunk.
5. Join with `\n`. Apply 200-character truncation with `...`.

**When `query` is `None` or has no lexical matches (fallback):**

Use the existing behavior: first 2 non-empty lines of content, 200-char
cap.

### Updated `ChunkResult` Methods

```python
def preview(self, query: str | None = None) -> str:
    """Generate a 2-line preview, preferring lines matching query terms."""
    return _pick_preview_lines(self.content, query)

def to_dict(self, query: str | None = None) -> dict:
    """Convert result to concise dictionary for API responses."""
    match_lines = _find_match_lines(self.content, query, self.start_line) if query else []
    result = {
        "file_path": self.file_path,
        "language": self.language,
        "start_line": self.start_line,
        "end_line": self.end_line,
        "definitions": self.definitions,
        "preview": self.preview(query),
        "score": self.score,
    }
    if match_lines:
        result["match_lines"] = match_lines
    return result
```

Key decisions:
- `query` defaults to `None` — calling `to_dict()` without arguments
  preserves identical behavior to today.
- `match_lines` is **omitted** from the dict when empty (not set to `[]`).
  This keeps the response lean for semantic queries with no lexical overlap.

### File Grouping Hint

Computed in `EmbeCodeServer.search_code()` after building the results list:

```python
from collections import Counter

results = [result.to_dict(query=query) for result in response.results]

# Add file grouping hint when multiple results share a file
file_counts = Counter(r["file_path"] for r in results)
for r in results:
    count = file_counts[r["file_path"]]
    if count > 1:
        r["file_result_count"] = count

return results
```

`file_result_count` is **only present when > 1**. Results from unique files
have no extra field. This is more robust than a `same_file_as_previous`
flag because it works even when same-file chunks are non-adjacent in the
ranked list.

---

## Updated Response Format

### Result with lexical matches (keyword/hybrid query)

```json
{
  "file_path": "src/components/UserAvatar.tsx",
  "language": "tsx",
  "start_line": 27,
  "end_line": 95,
  "definitions": "interface UserAvatarProps, function UserAvatar",
  "preview": "interface UserAvatarProps extends PropsWithClassName {\nexport function UserAvatar({",
  "score": 0.032,
  "match_lines": [27, 82, 91],
  "file_result_count": 2
}
```

### Result with no lexical matches (semantic-only)

```json
{
  "file_path": "src/auth/verify.py",
  "language": "python",
  "start_line": 15,
  "end_line": 42,
  "definitions": "function verify_credentials",
  "preview": "def verify_credentials(user, password):\n    \"\"\"Verify user credentials against the database.\"\"\"",
  "score": 0.028
}
```

Note: no `match_lines` or `file_result_count` keys — they are omitted
when not applicable.

### Response Size Impact

- `match_lines`: ~20-40 bytes per result when present. Zero when omitted.
- `file_result_count`: ~25 bytes per result when present. Zero when omitted.
- `preview`: Same size as before (2 lines, 200-char cap). Content changes
  but length is comparable.
- **Net increase**: negligible. Typical query adds ~100-200 bytes total
  across all results.

---

## Files Changed

| File | Change |
|---|---|
| `src/embecode/searcher.py` | Add `_tokenize_query()`, `_find_match_lines()`, `_pick_preview_lines()`. Update `preview()` and `to_dict()` signatures to accept optional `query`. |
| `src/embecode/server.py` | Pass `query` to `to_dict()`. Add `file_result_count` post-processing with `Counter`. |
| `tests/test_searcher.py` | New tests for match-aware behavior, fallback, match_lines. |
| `tests/test_server.py` | Tests for file_result_count and query passthrough. |

---

## Implementation Phases

### Phase 1: Shared matching logic + match-aware preview

Add `_tokenize_query()`, `_find_match_lines()`, and `_pick_preview_lines()`
to `searcher.py`. Update `ChunkResult.preview()` to accept optional
`query`. Write tests for all new functions.

### Phase 2: Wire query through `to_dict()` and add `match_lines`

Update `ChunkResult.to_dict()` to accept optional `query`, compute
`match_lines`, and include it in the response. Update `server.py` line 412
to pass `query` to `to_dict()`. Update server and searcher tests.

### Phase 3: File grouping hint

Add `file_result_count` post-processing in `EmbeCodeServer.search_code()`.
Write server tests.

---

## Edge Cases

### No query tokens after filtering

Query is `"a b c"` (all single-char tokens dropped). `_tokenize_query()`
returns `[]`. `_find_match_lines()` returns `[]`. Preview falls back to
first-2-lines. `match_lines` is omitted.

### Query matches every line

Large chunk where every line contains `self`. `match_lines` is capped at 8
entries. Preview picks the first matching line + nearest context.

### Chunk with only 1 non-empty line

Preview is that single line regardless of match behavior. If it matches the
query, `match_lines` contains 1 entry.

### Multiple tokens, partial matches

Query `"UserAvatar component"`. A line containing only `UserAvatar` (not
`component`) still matches — any token match counts. Lines with both tokens
score higher for preview selection (higher `match_count`).

### CamelCase and snake_case

Query `"UserAvatar"` tokenizes to `["useravatar"]`. This matches lines
containing `UserAvatar` (case-insensitive substring). It does NOT match
`user_avatar` — the tokens are different strings. This is acceptable
because the primary use case is searching for exact symbol names.

### Semantic query with no lexical overlap

Query `"authentication logic"`, chunk contains `verify_credentials`. Tokens
`["authentication", "logic"]` match zero lines. Preview falls back to
first-2-lines. `match_lines` is omitted from response.

### Single result per file (common case)

`file_result_count` is not present. Zero overhead.

---

## Backward Compatibility

- `preview` field semantics change (may show different lines), but this is
  an improvement in quality, not a format break. The field type (`str`) and
  truncation rules are identical.
- `match_lines` is a new optional field. Existing consumers that iterate
  over known keys are unaffected. Consumers that do strict key equality
  checks (e.g., `assert set(keys) == expected`) need test updates.
- `file_result_count` is a new optional field. Same compatibility story as
  `match_lines`.
- `to_dict(query=None)` defaults to `None` — calling without arguments
  produces identical output to today.

---

## Testing Requirements

### File: `tests/test_searcher.py` (additions)

#### Tokenization

- **`test_tokenize_query_basic`**
  `"UserAvatar component"` → `["useravatar", "component"]`.

- **`test_tokenize_query_drops_single_chars`**
  `"a b foo"` → `["foo"]`.

- **`test_tokenize_query_deduplicates`**
  `"foo bar foo"` → `["foo", "bar"]`.

- **`test_tokenize_query_empty_string`**
  `""` → `[]`.

- **`test_tokenize_query_special_characters`**
  `"user-avatar.tsx"` → `["user", "avatar", "tsx"]`.

#### Match-aware preview

- **`test_preview_shows_matching_lines_when_query_provided`**
  Chunk has `import React` on line 1 and `export function UserAvatar` on
  line 5. Query `"UserAvatar"`. Preview shows line 5 (not line 1).

- **`test_preview_falls_back_when_no_lexical_match`**
  Query `"authentication logic"`, content has `verify_credentials`. Preview
  shows first 2 non-empty lines.

- **`test_preview_falls_back_when_no_query`**
  `query=None`. Same as current first-2-lines behavior.

- **`test_preview_case_insensitive_match`**
  Query `"useravatar"` matches line with `"UserAvatar"`.

- **`test_preview_picks_best_match_and_nearest`**
  Query `"foo"`, chunk has foo on lines 3, 8, and 15. Preview shows line 3
  (first best match) + line 8 (nearest other match).

- **`test_preview_single_match_with_context`**
  Query matches only line 5 of a 10-line chunk. Preview shows line 5 +
  line 6 (nearest non-empty neighbor).

- **`test_preview_match_preserves_200_char_limit`**
  Matching line is 250 chars. Preview is truncated to 200 with `...`.

#### Match lines

- **`test_match_lines_absolute_line_numbers`**
  Chunk starts at line 10. Query matches relative lines 0 and 2.
  `match_lines` is `[10, 12]`.

- **`test_match_lines_omitted_when_empty`**
  Semantic query, no text match. `"match_lines"` key absent from
  `to_dict()` result.

- **`test_match_lines_capped_at_8`**
  Content with 20 matching lines. `match_lines` has exactly 8 entries.

- **`test_to_dict_with_query_has_match_lines`**
  `to_dict(query="foo")` returns dict with `match_lines` key.

- **`test_to_dict_without_query_no_match_lines`**
  `to_dict()` (no query) returns dict without `match_lines` key.

### File: `tests/test_server.py` (additions)

- **`test_search_code_passes_query_to_to_dict`**
  Verify the query string flows through from the MCP tool to `to_dict()`.

- **`test_search_code_file_result_count_for_duplicates`**
  Two results from `src/auth.py`, one from `src/main.py`. Both auth
  results have `file_result_count: 2`. The main result has no
  `file_result_count` key.

- **`test_search_code_file_result_count_absent_for_unique_files`**
  All results from different files. No result has `file_result_count`.

- **`test_search_code_concise_results_allows_optional_fields`**
  Update existing `test_search_code_returns_concise_results` to verify
  that the required keys are present and `content` is absent, without
  asserting exact key equality (allowing optional `match_lines` and
  `file_result_count`).
