"""Code chunking using cAST (Context-Aware Syntax Tree) algorithm."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import tree_sitter_languages as tsl
from tree_sitter import Node, Parser, Tree

from embecode.config import LanguageConfig


@dataclass
class Chunk:
    """A semantically meaningful chunk of code."""

    content: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    context: str  # Enrichment metadata (file path, scope, imports, etc.)
    hash: str  # SHA1 hash of content for incremental indexing

    @classmethod
    def create(
        cls,
        content: str,
        file_path: str,
        language: str,
        start_line: int,
        end_line: int,
        context: str,
    ) -> Chunk:
        """Create a chunk with computed hash."""
        content_hash = hashlib.sha1(content.encode("utf-8")).hexdigest()
        return cls(
            content=content,
            file_path=file_path,
            language=language,
            start_line=start_line,
            end_line=end_line,
            context=context,
            hash=content_hash,
        )


# Language name mappings from file extension to tree-sitter language
LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".cs": "c_sharp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".r": "r",
    ".lua": "lua",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".fish": "fish",
    ".sql": "sql",
    ".html": "html",
    ".xml": "xml",
    ".css": "css",
    ".scss": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".rst": "rst",
}


def get_language_for_file(file_path: Path) -> str | None:
    """Get the language name for a file based on its extension."""
    suffix = file_path.suffix.lower()
    return LANGUAGE_MAP.get(suffix)


def _count_non_whitespace(text: str) -> int:
    """Count non-whitespace characters in text."""
    return sum(1 for c in text if not c.isspace())


def _get_parser(language: str) -> Parser | None:
    """Get a tree-sitter parser for the given language."""
    try:
        parser = Parser()
        lang = tsl.get_language(language)
        parser.language = lang
        return parser
    except Exception:
        # Language not supported by tree-sitter-languages
        return None


def _extract_node_text(node: Node, source_code: bytes) -> str:
    """Extract the text content of a node."""
    return source_code[node.start_byte : node.end_byte].decode("utf-8")


def _get_context_info(file_path: str, language: str, source_code: str) -> str:
    """Generate context metadata for a chunk (file path, language, etc.)."""
    context_parts = [f"File: {file_path}", f"Language: {language}"]
    return "\n".join(context_parts)


def _merge_nodes_into_chunk(
    nodes: list[Node], source_code: bytes, file_path: str, language: str
) -> Chunk | None:
    """Merge a list of sibling nodes into a single chunk."""
    if not nodes:
        return None

    # Get the text spanning all nodes
    start_node = nodes[0]
    end_node = nodes[-1]
    content = source_code[start_node.start_byte : end_node.end_byte].decode("utf-8")

    # Calculate line numbers (tree-sitter uses 0-indexed lines)
    start_line = start_node.start_point[0] + 1
    end_line = end_node.end_point[0] + 1

    context = _get_context_info(file_path, language, content)

    return Chunk.create(
        content=content,
        file_path=file_path,
        language=language,
        start_line=start_line,
        end_line=end_line,
        context=context,
    )


def _chunk_node_recursively(
    node: Node,
    source_code: bytes,
    file_path: str,
    language: str,
    max_size: int,
    chunks: list[Chunk],
) -> None:
    """
    Recursively chunk a node using the cAST algorithm.

    Strategy:
    1. Try to merge sibling nodes into chunks up to max_size
    2. If a single node exceeds max_size, recurse into its children
    3. Chunk boundaries always align with complete syntactic units
    """
    children = list(node.children)

    if not children:
        # Leaf node - create a chunk from it
        content = _extract_node_text(node, source_code)
        if content.strip():  # Only create chunks with non-empty content
            chunk = _merge_nodes_into_chunk([node], source_code, file_path, language)
            if chunk:
                chunks.append(chunk)
        return

    # Filter out empty/whitespace-only children
    meaningful_children = [
        child for child in children if _extract_node_text(child, source_code).strip()
    ]

    if not meaningful_children:
        return

    current_batch: list[Node] = []
    current_size = 0

    for child in meaningful_children:
        child_text = _extract_node_text(child, source_code)
        child_size = _count_non_whitespace(child_text)

        # If this single child exceeds max_size, recurse into it
        if child_size > max_size and child.children:
            # First, flush any accumulated batch
            if current_batch:
                chunk = _merge_nodes_into_chunk(current_batch, source_code, file_path, language)
                if chunk:
                    chunks.append(chunk)
                current_batch = []
                current_size = 0

            # Recurse into the large child
            _chunk_node_recursively(child, source_code, file_path, language, max_size, chunks)
        else:
            # Try to add this child to the current batch
            if current_size + child_size <= max_size:
                current_batch.append(child)
                current_size += child_size
            else:
                # Flush the current batch and start a new one
                if current_batch:
                    chunk = _merge_nodes_into_chunk(current_batch, source_code, file_path, language)
                    if chunk:
                        chunks.append(chunk)
                current_batch = [child]
                current_size = child_size

    # Flush any remaining batch
    if current_batch:
        chunk = _merge_nodes_into_chunk(current_batch, source_code, file_path, language)
        if chunk:
            chunks.append(chunk)


def chunk_file(
    file_path: Path,
    language_config: LanguageConfig,
) -> list[Chunk]:
    """
    Chunk a file using the cAST algorithm.

    Args:
        file_path: Path to the file to chunk
        language_config: Language-specific chunk size configuration

    Returns:
        List of chunks extracted from the file
    """
    # Determine language
    language = get_language_for_file(file_path)
    if not language:
        # Unsupported file type - return empty list
        return []

    # Get max chunk size for this language
    if language == "python":
        max_size = language_config.python
    elif language in ("typescript", "tsx"):
        max_size = language_config.typescript
    elif language == "javascript":
        max_size = language_config.javascript
    else:
        max_size = language_config.default

    # Read file content
    try:
        source_code = file_path.read_bytes()
    except Exception:
        # File read error - return empty list
        return []

    # Parse with tree-sitter
    parser = _get_parser(language)
    if not parser:
        # Parser not available - fall back to single chunk
        content = source_code.decode("utf-8", errors="replace")
        context = _get_context_info(str(file_path), language, content)
        return [
            Chunk.create(
                content=content,
                file_path=str(file_path),
                language=language,
                start_line=1,
                end_line=len(content.splitlines()),
                context=context,
            )
        ]

    tree: Tree = parser.parse(source_code)
    root = tree.root_node

    # Recursively chunk the AST
    chunks: list[Chunk] = []
    _chunk_node_recursively(root, source_code, str(file_path), language, max_size, chunks)

    return chunks


def chunk_files(
    file_paths: list[Path],
    language_config: LanguageConfig,
) -> Iterator[Chunk]:
    """
    Chunk multiple files.

    Args:
        file_paths: List of file paths to chunk
        language_config: Language-specific chunk size configuration

    Yields:
        Chunks from all files
    """
    for file_path in file_paths:
        chunks = chunk_file(file_path, language_config)
        yield from chunks
