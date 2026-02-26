"""Hybrid search with BM25 + dense vector and RRF fusion."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from embecode.db import Database
    from embecode.embedder import Embedder

logger = logging.getLogger(__name__)


class SearchError(Exception):
    """Base exception for search errors."""


class IndexNotReadyError(SearchError):
    """Raised when attempting to search before index is ready."""


@dataclass
class ChunkResult:
    """A search result containing a code chunk with metadata and score."""

    content: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    definitions: str
    score: float  # Relevance score (higher is better)

    def preview(self) -> str:
        """Generate a 2-line preview from chunk content."""
        lines = [line for line in self.content.splitlines() if line.strip()]
        preview = "\n".join(lines[:2])
        if len(preview) > 200:
            return preview[:197] + "..."
        return preview

    def to_dict(self) -> dict:
        """Convert result to concise dictionary for API responses (no full content)."""
        return {
            "file_path": self.file_path,
            "language": self.language,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "definitions": self.definitions,
            "preview": self.preview(),
            "score": self.score,
        }


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


class Searcher:
    """
    Hybrid search engine combining BM25 (keyword) and dense vector (semantic) search.

    Uses Reciprocal Rank Fusion (RRF) to combine results from both search methods.
    """

    # RRF constant (standard value, no tuning needed)
    RRF_K = 60

    def __init__(self, db: Database, embedder: Embedder) -> None:
        """
        Initialize searcher.

        Args:
            db: Database interface for querying chunks and embeddings.
            embedder: Embedder for generating query embeddings.
        """
        self.db = db
        self.embedder = embedder

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 10,
        path: str | None = None,
    ) -> list[ChunkResult]:
        """
        Search the codebase using keyword, semantic, or hybrid search.

        Args:
            query: Search query string (natural language or code).
            mode: Search mode - "semantic", "keyword", or "hybrid".
            top_k: Number of results to return.
            path: Optional path prefix filter (e.g., "src/", "apps/ui/").

        Returns:
            List of ChunkResult objects ordered by relevance (highest score first).

        Raises:
            IndexNotReadyError: If index is empty or not ready.
            SearchError: For other search-related errors.
        """
        # Validate mode
        if mode not in {"semantic", "keyword", "hybrid"}:
            msg = f"Invalid search mode: {mode}. Must be 'semantic', 'keyword', or 'hybrid'."
            raise SearchError(msg)

        # Check if index is ready
        if not self._is_index_ready():
            msg = "Index is not ready. Please wait for indexing to complete."
            raise IndexNotReadyError(msg)

        # Execute search based on mode
        if mode == "semantic":
            return self._search_semantic(query, top_k, path)
        elif mode == "keyword":
            return self._search_keyword(query, top_k, path)
        else:  # hybrid
            return self._search_hybrid(query, top_k, path)

    def _is_index_ready(self) -> bool:
        """Check if index has been built and is ready for search."""
        stats = self.db.get_index_stats()
        return stats["total_chunks"] > 0

    def _search_semantic(
        self,
        query: str,
        top_k: int,
        path: str | None = None,
    ) -> list[ChunkResult]:
        """
        Perform semantic search using dense vector embeddings.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            path: Optional path prefix filter.

        Returns:
            List of ChunkResult objects ordered by cosine similarity.
        """
        # Generate query embedding
        query_embedding = self.embedder.embed([query])[0]

        # Query database for nearest neighbors
        results = self.db.vector_search(
            query_embedding=query_embedding,
            top_k=top_k,
            path_prefix=path,
        )

        return [
            ChunkResult(
                content=row["content"],
                file_path=row["file_path"],
                language=row["language"],
                start_line=row["start_line"],
                end_line=row["end_line"],
                definitions=row.get("definitions", ""),
                score=row["score"],
            )
            for row in results
        ]

    def _search_keyword(
        self,
        query: str,
        top_k: int,
        path: str | None = None,
    ) -> list[ChunkResult]:
        """
        Perform keyword search using BM25 (full-text search).

        Args:
            query: Search query string.
            top_k: Number of results to return.
            path: Optional path prefix filter.

        Returns:
            List of ChunkResult objects ordered by BM25 score.
        """
        # Query database using FTS (BM25)
        results = self.db.bm25_search(
            query=query,
            top_k=top_k,
            path_prefix=path,
        )

        return [
            ChunkResult(
                content=row["content"],
                file_path=row["file_path"],
                language=row["language"],
                start_line=row["start_line"],
                end_line=row["end_line"],
                definitions=row.get("definitions", ""),
                score=row["score"],
            )
            for row in results
        ]

    def _search_hybrid(
        self,
        query: str,
        top_k: int,
        path: str | None = None,
    ) -> list[ChunkResult]:
        """
        Perform hybrid search using RRF fusion of semantic and keyword results.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            path: Optional path prefix filter.

        Returns:
            List of ChunkResult objects ordered by RRF score.
        """
        # Get results from both search methods (fetch more to improve fusion quality)
        # Typical heuristic: fetch 2-3x top_k from each leg
        fetch_k = top_k * 3

        semantic_results = self._search_semantic(query, fetch_k, path)
        keyword_results = self._search_keyword(query, fetch_k, path)

        # Build unique ID for each chunk (file_path + start_line)
        def chunk_id(result: ChunkResult) -> str:
            return f"{result.file_path}:{result.start_line}"

        # Compute RRF scores
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, ChunkResult] = {}

        # Add semantic results with RRF scoring
        for rank, result in enumerate(semantic_results, start=1):
            cid = chunk_id(result)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.RRF_K + rank)
            chunk_map[cid] = result

        # Add keyword results with RRF scoring
        for rank, result in enumerate(keyword_results, start=1):
            cid = chunk_id(result)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.RRF_K + rank)
            # Only add to chunk_map if not already present (prefer semantic version)
            if cid not in chunk_map:
                chunk_map[cid] = result

        # Sort by RRF score (descending) and take top_k
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        top_ids = sorted_ids[:top_k]

        # Build final results with updated scores
        return [
            ChunkResult(
                content=chunk_map[cid].content,
                file_path=chunk_map[cid].file_path,
                language=chunk_map[cid].language,
                start_line=chunk_map[cid].start_line,
                end_line=chunk_map[cid].end_line,
                definitions=chunk_map[cid].definitions,
                score=rrf_scores[cid],
            )
            for cid in top_ids
        ]
