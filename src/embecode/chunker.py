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
    definitions: str = ""  # Comma-separated definition names (e.g. "function foo, class Bar")

    @classmethod
    def create(
        cls,
        content: str,
        file_path: str,
        language: str,
        start_line: int,
        end_line: int,
        context: str,
        definitions: str = "",
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
            definitions=definitions,
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


# Maps (language, node_type) -> label for definition extraction.
# Only languages and node types listed here produce definitions.
DEFINITION_NODE_TYPES: dict[str, dict[str, str]] = {
    "python": {
        "function_definition": "function",
        "class_definition": "class",
        "decorated_definition": "__decorated__",  # sentinel: unwrap to inner def/class
    },
    "typescript": {
        "function_declaration": "function",
        "class_declaration": "class",
        "interface_declaration": "interface",
        "type_alias_declaration": "type",
        "enum_declaration": "enum",
        "method_definition": "method",
    },
    "tsx": {
        "function_declaration": "function",
        "class_declaration": "class",
        "interface_declaration": "interface",
        "type_alias_declaration": "type",
        "enum_declaration": "enum",
        "method_definition": "method",
    },
    "javascript": {
        "function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
    },
    "go": {
        "function_declaration": "function",
        "method_declaration": "method",
        "type_declaration": "__go_type__",  # sentinel: name is on type_spec child
    },
    "rust": {
        "function_item": "function",
        "struct_item": "struct",
        "enum_item": "enum",
        "impl_item": "__rust_impl__",  # sentinel: name is in type field
        "trait_item": "trait",
    },
    "java": {
        "class_declaration": "class",
        "interface_declaration": "interface",
        "method_declaration": "method",
        "enum_declaration": "enum",
    },
}


def _extract_definition_names(nodes: list[Node], language: str) -> list[str]:
    """Extract named definitions from AST nodes.

    Walks the list of nodes being merged into a chunk and returns a list of
    definition strings like ``["function foo", "class Bar"]``.
    """
    lang_table = DEFINITION_NODE_TYPES.get(language)
    if not lang_table:
        return []

    definitions: list[str] = []

    for node in nodes:
        _collect_definitions(node, lang_table, language, definitions)

    return definitions


def _collect_definitions(
    node: Node,
    lang_table: dict[str, str],
    language: str,
    definitions: list[str],
) -> None:
    """Recursively collect definition names from a node and its children."""
    label = lang_table.get(node.type)

    if label is not None:
        name = _extract_def_name(node, label, language)
        if name:
            definitions.append(name)
        # Don't recurse into matched nodes — their children are part of
        # the same definition (avoids double-counting inner methods that
        # are already captured when the class node is recursed into at
        # chunk level).
        return

    # Recurse into children to find nested definitions (e.g. methods inside
    # a class body that was split during chunking).
    for child in node.children:
        _collect_definitions(child, lang_table, language, definitions)


def _extract_def_name(node: Node, label: str, language: str) -> str | None:
    """Extract a formatted definition name from a single AST node.

    Returns e.g. ``"function foo"`` or ``None`` if the name can't be
    determined.
    """
    # Python decorated_definition: unwrap to inner function/class
    if label == "__decorated__":
        return _extract_decorated_definition(node, language)

    # Go type_declaration: name is on the type_spec child
    if label == "__go_type__":
        return _extract_go_type_declaration(node)

    # Rust impl_item: name is in the "type" field
    if label == "__rust_impl__":
        return _extract_rust_impl(node)

    # General case: name is in the "name" field
    name_node = node.child_by_field_name("name")
    if name_node and name_node.text is not None:
        return f"{label} {name_node.text.decode('utf-8')}"

    return None


def _extract_decorated_definition(node: Node, language: str) -> str | None:
    """Handle Python ``decorated_definition`` by unwrapping to inner def/class."""
    lang_table = DEFINITION_NODE_TYPES.get(language, {})
    for child in node.children:
        inner_label = lang_table.get(child.type)
        if inner_label and inner_label != "__decorated__":
            name_node = child.child_by_field_name("name")
            if name_node and name_node.text is not None:
                return f"{inner_label} {name_node.text.decode('utf-8')}"
    return None


def _extract_go_type_declaration(node: Node) -> str | None:
    """Handle Go ``type_declaration`` — name is on the ``type_spec`` child."""
    for child in node.children:
        if child.type == "type_spec":
            name_node = child.child_by_field_name("name")
            if name_node and name_node.text is not None:
                return f"type {name_node.text.decode('utf-8')}"
    return None


def _extract_rust_impl(node: Node) -> str | None:
    """Handle Rust ``impl_item`` — name is in the ``type`` field."""
    type_node = node.child_by_field_name("type")
    if type_node and type_node.text is not None:
        return f"impl {type_node.text.decode('utf-8')}"
    return None


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

    # Extract definition names from AST nodes
    defs = _extract_definition_names(nodes, language)
    definitions = ", ".join(defs)

    context = _get_context_info(file_path, language, content)
    if definitions:
        context += f"\nDefines: {definitions}"

    return Chunk.create(
        content=content,
        file_path=file_path,
        language=language,
        start_line=start_line,
        end_line=end_line,
        context=context,
        definitions=definitions,
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
