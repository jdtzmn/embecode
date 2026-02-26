# Spec: .gitignore Support

## Goal

Embecode should respect `.gitignore` files exactly as git does, so that files
excluded by a project's ignore rules are never indexed.

---

## Scope

- **Project `.gitignore` files only.** Embecode reads every `.gitignore` file
  found inside the project tree being indexed.
- **Global git ignore is out of scope.** `~/.config/git/ignore` and
  `$GIT_DIR/info/exclude` are not read.
- **Non-git repos are supported.** Embecode respects `.gitignore` files even
  when no `.git` directory exists at or above the project root.

---

## Dependency

Use the [`pathspec`](https://github.com/cpburnz/python-pathspec) library with
`GitWildMatchPattern` for all pattern matching. This delegates the full git
pattern grammar to a well-tested, widely-used implementation and avoids
re-implementing edge cases by hand.

`pathspec` must be added to `[project.dependencies]` in `pyproject.toml`.

---

## Pattern Semantics

Pattern interpretation must match the
[gitignore specification](https://git-scm.com/docs/gitignore) exactly.
The following behaviors are all delegated to `pathspec.GitWildMatchPattern`
and are listed here for traceability:

| Rule | Example | Behavior |
|---|---|---|
| Blank lines | *(empty line)* | Ignored (no effect) |
| Comments | `# note` | Lines starting with `#` are ignored |
| Trailing spaces | `foo ` | Stripped unless escaped with `\` |
| Negation | `!foo.txt` | Un-ignores a previously ignored path |
| Escaped `!` | `\!important` | Matches literal `!important` |
| Leading `/` | `/foo` | Anchored to the `.gitignore`'s directory |
| Middle `/` | `foo/bar` | Anchored to the `.gitignore`'s directory |
| Trailing `/` | `build/` | Matches directories only |
| Single `*` | `*.log` | Matches anything except `/` |
| `?` | `foo?.txt` | Matches any one character except `/` |
| Character range | `[a-z]` | Matches one character in range |
| `**/` prefix | `**/foo` | Matches `foo` at any depth |
| `/**` suffix | `abc/**` | Matches everything inside `abc/` |
| `/**/` middle | `a/**/b` | Matches zero or more directory levels |
| Backslash escape | `\*` | Matches a literal `*` |

A pattern with **no `/`** (other than a trailing one) matches at any depth
below the `.gitignore` file's directory.

A pattern with a `/` in the **beginning or middle** is relative to the
`.gitignore` file's directory only.

---

## `.gitignore` Discovery

During `Indexer._collect_files`, `.gitignore` files are discovered lazily as
the directory tree is walked. The walk must be depth-first so that a
directory's own `.gitignore` is loaded before its children are processed.

### Loading order

For a file at path `a/b/c/file.py`, the applicable `.gitignore` files (in
increasing precedence) are:

```
<project_root>/.gitignore        ← lowest precedence
<project_root>/a/.gitignore
<project_root>/a/b/.gitignore
<project_root>/a/b/c/.gitignore  ← highest precedence
```

This matches git's rule: **patterns in lower-level files override those in
higher-level files.**

Within a single `.gitignore` file, the **last matching pattern wins.**

### Caching

Each `.gitignore` file is parsed once and cached by its directory path for the
duration of an indexing run. The cache is not persisted across runs.

---

## Ignore Decision Algorithm

For each candidate file path, `_should_index_file` is extended as follows:

1. Collect all applicable `PathSpec` objects from the file's ancestor
   directories (from project root down to the file's parent), in that order.
2. For each spec (root-first), test the file's path relative to that spec's
   directory.
3. Apply the **last matching rule** across all specs (lower `.gitignore` takes
   priority over a higher one on the same path).
4. If the final decision is **ignored**, return `False`.
5. Otherwise, proceed to the existing `include`/`exclude` config rules.

Negation (`!`) within a single `.gitignore` file is handled internally by
`pathspec`. Cross-file negation follows git behavior: a lower `.gitignore`
can re-include a file ignored by a higher one **unless a parent directory has
been excluded**, in which case the re-include has no effect.

### Excluded parent directories

If a directory itself is matched by a gitignore pattern, its entire subtree is
excluded and no patterns inside nested `.gitignore` files within that directory
can re-include files. This mirrors git's behavior where excluded directories
are not recursed into.

---

## Interaction with Embecode Config

The evaluation order for a file is:

1. **Gitignore check** — if ignored, exclude immediately.
2. **Config `exclude` patterns** — if matched, exclude.
3. **Config `include` patterns** — if specified and not matched, exclude.
4. **Default** — include.

Gitignore is checked first so that config `include` patterns cannot
accidentally pull in gitignored files.

---

## Watcher Behavior

`watcher.py` fires `Indexer.update_file` and `Indexer.delete_file` on
filesystem events. `update_file` already calls `_should_index_file`, so
gitignore filtering applies automatically to incremental updates at no extra
cost.

A newly-created `.gitignore` file or a change to an existing one should trigger
a **full re-index**, since the ignore rules for many already-indexed files may
have changed. The watcher must detect writes to any file named `.gitignore`
and call `Indexer.start_full_index` in response.

---

## Configuration

No new config keys are added. Gitignore support is always active and cannot be
disabled via `.embecode.toml`. If a project truly needs to index gitignored
files, the user can add explicit `include` overrides in their embecode config
(though this does not override gitignore — see interaction rules above).

> **Future consideration:** A top-level `index.respect_gitignore = false` flag
> could be added later if a clear use case arises.

---

## Default Configuration Change

As part of this feature, the default `include` list is changed from
`["src/", "lib/", "tests/"]` to `[]` (empty), meaning **all files are indexed
by default**.

This is a better default because:
- The old default silently indexes nothing in projects that don't follow that
  exact directory convention (e.g., monorepos, Go projects, Rails apps).
- With gitignore support, the meaningful filtering now comes from `.gitignore`
  files rather than an opinionated hardcoded list.
- The existing `exclude` defaults (`node_modules/`, `.venv/`, `.git/`, etc.)
  continue to handle the obvious noise for non-git repos.

### Files to update
- `src/embecode/config.py` — change `IndexConfig.include` default from
  `["src/", "lib/", "tests/"]` to `[]`
- `.embecode.toml.example` — update the example `include` value to `[]` and
  add a comment explaining that an empty list means "index everything"

---

## Testing Requirements

All tests live in `tests/test_gitignore.py`. Tests use `pytest` with `tmp_path`
fixtures to create real filesystem trees. `Indexer` is instantiated with mock
DB and embedder (same pattern as `test_indexer.py`). No real embeddings are
generated.

### Class: `TestGitignoreBasic`

- `test_no_gitignore_indexes_all_files`
  Project has no `.gitignore`. All files are collected.

- `test_root_gitignore_excludes_matched_files`
  Root `.gitignore` contains `*.log`. `foo.log` is excluded, `foo.py` is
  included.

- `test_blank_lines_and_comments_ignored`
  `.gitignore` with only blank lines and `#` comments. All files still indexed.

- `test_empty_gitignore_has_no_effect`
  `.gitignore` exists but is completely empty. All files still indexed.

- `test_no_git_directory_still_respects_gitignore`
  Project has `.gitignore` but no `.git` directory. Gitignore is still applied.

### Class: `TestGitignorePatternAnchoring`

- `test_unanchored_pattern_matches_at_any_depth`
  Pattern `*.log` (no slash) in root `.gitignore` excludes `a/b/c/file.log`.

- `test_leading_slash_anchors_to_gitignore_dir`
  Pattern `/foo.txt` excludes `<root>/foo.txt` but NOT `<root>/sub/foo.txt`.

- `test_middle_slash_anchors_to_gitignore_dir`
  Pattern `foo/bar.txt` excludes `<root>/foo/bar.txt` but NOT
  `<root>/sub/foo/bar.txt`.

- `test_trailing_slash_matches_directories_only`
  Pattern `build/` excludes `build/output.js` but does NOT exclude a file
  literally named `build`.

- `test_double_star_prefix_matches_any_depth`
  Pattern `**/logs` matches `logs/` at root AND `sub/logs/` AND `a/b/logs/`.

- `test_double_star_suffix_matches_subtree`
  Pattern `abc/**` excludes all files inside `abc/` at any depth.

- `test_double_star_middle_matches_zero_or_more_dirs`
  Pattern `a/**/b` matches `a/b`, `a/x/b`, and `a/x/y/b`.

- `test_single_star_does_not_cross_slash`
  Pattern `foo/*` matches `foo/bar.txt` but NOT `foo/sub/bar.txt`.

- `test_question_mark_matches_single_non_slash_char`
  Pattern `foo?.txt` matches `food.txt` but NOT `foooo.txt` and NOT `fo/.txt`.

- `test_character_range_pattern`
  Pattern `[abc].py` matches `a.py`, `b.py`, `c.py` but NOT `d.py`.

### Class: `TestGitignoreNegation`

- `test_negation_re_includes_within_same_file`
  `.gitignore`: `*.log` then `!important.log`.
  `important.log` is included; `other.log` is excluded.

- `test_negation_order_matters`
  `.gitignore`: `!important.log` then `*.log`.
  Both `important.log` and `other.log` are excluded (last rule wins).

- `test_negation_in_child_overrides_parent_ignore`
  Root `.gitignore`: `*.log`
  `sub/.gitignore`: `!keep.log`
  `sub/keep.log` is included; `sub/other.log` is excluded;
  `root/other.log` is excluded.

- `test_negation_cannot_re_include_inside_excluded_dir`
  Root `.gitignore`: `build/`
  Even if `build/.gitignore` contained `!important.txt`, files inside `build/`
  remain excluded because the parent directory is excluded.

- `test_escaped_exclamation_mark`
  Pattern `\!readme` matches a file literally named `!readme` (not a negation).

### Class: `TestGitignoreNesting`

- `test_child_gitignore_overrides_parent`
  Root `.gitignore`: `*.py` (exclude all Python files)
  `sub/.gitignore`: `!main.py` (re-include `main.py` in `sub/`)
  `sub/main.py` is included; `root/other.py` is excluded;
  `sub/other.py` is excluded.

- `test_child_gitignore_scoped_to_its_directory`
  `sub/.gitignore`: `*.log`
  `root/file.log` is still included (child rule doesn't apply here).
  `sub/file.log` is excluded.

- `test_deeply_nested_gitignore`
  Three levels: root, `root/a`, `root/a/b` each with their own `.gitignore`.
  Rules apply correctly at each level with proper precedence.

- `test_multiple_gitignore_files_independent_scopes`
  Two sibling directories each with their own `.gitignore`.
  Rules from `dir_a/.gitignore` don't affect files in `dir_b/`.

### Class: `TestGitignoreAndEmbeCodeConfig`

- `test_gitignore_checked_before_config_include`
  File is gitignored. Config `include = []` (include everything).
  File is still excluded — gitignore takes priority.

- `test_config_exclude_still_applies_to_non_gitignored_files`
  File is not gitignored. Config `exclude` matches it.
  File is excluded.

- `test_config_include_cannot_override_gitignore`
  File is gitignored. Config `include` explicitly lists the file's directory.
  File is still excluded.

- `test_both_gitignore_and_config_exclude_file`
  File matches both a gitignore pattern and a config `exclude` pattern.
  File is excluded (no conflict, no error).

- `test_default_include_empty_indexes_all_non_gitignored`
  Config `include = []` (new default). No `.gitignore`.
  All files collected.

### Class: `TestGitignoreWatcher`

- `test_modifying_gitignore_triggers_full_reindex`
  Watcher observes a `Change.modified` event on `.gitignore`.
  `Indexer.start_full_index` is called.

- `test_creating_new_gitignore_triggers_full_reindex`
  Watcher observes a `Change.added` event on a new `.gitignore` file.
  `Indexer.start_full_index` is called.

- `test_gitignored_file_modified_not_reindexed`
  File matches a gitignore rule. Watcher observes `Change.modified`.
  `Indexer.update_file` is NOT called.

- `test_non_gitignored_file_change_still_processed_normally`
  File does not match any gitignore rule. Watcher observes `Change.modified`.
  `Indexer.update_file` IS called.

### Class: `TestGitignoreEdgeCases`

- `test_gitignore_with_windows_line_endings`
  `.gitignore` uses `\r\n` line endings. Patterns still parsed correctly.

- `test_trailing_space_stripped_unless_escaped`
  Pattern `"foo.txt  "` (trailing spaces) matches `foo.txt`.
  Pattern `"foo.txt\ "` (escaped space) matches `"foo.txt "` (with space).

- `test_backslash_escapes_special_chars`
  Pattern `\*.txt` matches a file literally named `*.txt`, not all `.txt` files.

- `test_gitignore_caching_across_collect_files`
  Each `.gitignore` file is only read from disk once per indexing run.
  Verified by asserting `Path.open` call count matches number of unique
  `.gitignore` files, not number of files checked.

- `test_project_with_only_gitignored_files`
  All files in project are gitignored. `_collect_files` returns empty list.
  No error is raised.
