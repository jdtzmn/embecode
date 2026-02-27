"""Hybrid search with BM25 + dense vector and RRF fusion."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from embecode.db import Database
    from embecode.embedder import Embedder

logger = logging.getLogger(__name__)

# Max match lines to report per chunk (prevents bloat on large chunks)
_MAX_MATCH_LINES = 8


def _tokenize_query(query: str) -> list[str]:
    """Extract deduplicated lowercase tokens from a query string.

    Tokens are ``[A-Za-z0-9_]+`` runs with single-character tokens dropped
    (too noisy).  Order is preserved; duplicates keep only the first
    occurrence.
    """
    raw = re.findall(r"[A-Za-z0-9_]+", query)
    seen: set[str] = set()
    tokens: list[str] = []
    for tok in raw:
        lower = tok.lower()
        if len(lower) <= 1 or lower in seen:
            continue
        seen.add(lower)
        tokens.append(lower)
    return tokens


def _find_match_lines(
    content: str,
    query: str,
    start_line: int,
) -> list[int]:
    """Return absolute line numbers in *content* that contain any query token.

    Lines are numbered starting from *start_line*.  The result is sorted
    ascending and capped at :data:`_MAX_MATCH_LINES`.
    """
    tokens = _tokenize_query(query)
    if not tokens:
        return []

    matches: list[int] = []
    for idx, line in enumerate(content.splitlines()):
        line_lower = line.lower()
        if any(tok in line_lower for tok in tokens):
            matches.append(start_line + idx)
            if len(matches) >= _MAX_MATCH_LINES:
                break
    return matches


def _pick_preview_lines(
    content: str,
    query: str | None = None,
) -> str:
    """Generate a 2-line preview, preferring lines that match *query* terms.

    When *query* is provided and produces lexical matches the preview shows
    the best matching line (highest token-hit count, earliest index as
    tiebreak) plus a second line (nearest other match, or nearest non-empty
    neighbour).

    Falls back to the first 2 non-empty lines when *query* is ``None`` or
    has no lexical overlap with *content*.  The result is capped at 200
    characters with ``...`` appended when truncated.
    """
    all_lines = content.splitlines()

    picked: list[tuple[int, str]] | None = None

    if query:
        tokens = _tokenize_query(query)
        if tokens:
            # Score each non-empty line by how many tokens it contains
            scored: list[tuple[int, str, int]] = []
            for idx, line in enumerate(all_lines):
                if not line.strip():
                    continue
                line_lower = line.lower()
                count = sum(1 for tok in tokens if tok in line_lower)
                if count > 0:
                    scored.append((idx, line, count))

            if scored:
                # Best match: highest count, earliest index
                best = max(scored, key=lambda t: (t[2], -t[0]))
                best_idx, best_line = best[0], best[1]

                # Second line: nearest other match, or nearest non-empty line
                second: tuple[int, str] | None = None
                other_matches = [(i, l) for i, l, c in scored if i != best_idx]
                if other_matches:
                    second = min(other_matches, key=lambda t: abs(t[0] - best_idx))
                else:
                    # Fall back to nearest non-empty neighbour
                    for dist in range(1, len(all_lines)):
                        for candidate_idx in (best_idx - dist, best_idx + dist):
                            if (
                                0 <= candidate_idx < len(all_lines)
                                and all_lines[candidate_idx].strip()
                            ):
                                second = (candidate_idx, all_lines[candidate_idx])
                                break
                        if second is not None:
                            break

                if second is not None:
                    pair = sorted([(best_idx, best_line), second], key=lambda t: t[0])
                    picked = pair
                else:
                    picked = [(best_idx, best_line)]

    # Fallback: first 2 non-empty lines
    if picked is None:
        non_empty = [(i, l) for i, l in enumerate(all_lines) if l.strip()]
        picked = non_empty[:2]

    preview = "\n".join(line for _, line in picked)
    if len(preview) > 200:
        return preview[:197] + "..."
    return preview


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

    def preview(self, query: str | None = None) -> str:
        """Generate a 2-line preview, preferring lines matching query terms."""
        return _pick_preview_lines(self.content, query)

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


@dataclass
class SearchResponse:
    """Search results with timing breakdown (internal use only)."""

    results: list[ChunkResult]
    timings: SearchTimings


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
    ) -> SearchResponse:
        """
        Search the codebase using keyword, semantic, or hybrid search.

        Args:
            query: Search query string (natural language or code).
            mode: Search mode - "semantic", "keyword", or "hybrid".
            top_k: Number of results to return.
            path: Optional path prefix filter (e.g., "src/", "apps/ui/").

        Returns:
            SearchResponse with results ordered by relevance and per-phase timings.

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
            response = self._search_semantic(query, top_k, path)
        elif mode == "keyword":
            response = self._search_keyword(query, top_k, path)
        else:  # hybrid
            response = self._search_hybrid(query, top_k, path)

        logger.info(
            "search query=%r mode=%s %s",
            query,
            mode,
            response.timings.to_dict(),
        )
        return response

    def _is_index_ready(self) -> bool:
        """Check if index has been built and is ready for search."""
        stats = self.db.get_index_stats()
        return stats["total_chunks"] > 0

    def _search_semantic(
        self,
        query: str,
        top_k: int,
        path: str | None = None,
    ) -> SearchResponse:
        """
        Perform semantic search using dense vector embeddings.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            path: Optional path prefix filter.

        Returns:
            SearchResponse with results ordered by cosine similarity and timings.
        """
        timings = SearchTimings()
        t0 = time.perf_counter()

        # Phase: embedding
        t = time.perf_counter()
        query_embedding = self.embedder.embed([query])[0]
        timings.embedding_ms = (time.perf_counter() - t) * 1000

        # Phase: vector search
        t = time.perf_counter()
        results = self.db.vector_search(
            query_embedding=query_embedding,
            top_k=top_k,
            path_prefix=path,
        )
        timings.vector_search_ms = (time.perf_counter() - t) * 1000

        timings.total_ms = (time.perf_counter() - t0) * 1000

        return SearchResponse(
            results=[
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
            ],
            timings=timings,
        )

    def _search_keyword(
        self,
        query: str,
        top_k: int,
        path: str | None = None,
    ) -> SearchResponse:
        """
        Perform keyword search using BM25 (full-text search).

        Args:
            query: Search query string.
            top_k: Number of results to return.
            path: Optional path prefix filter.

        Returns:
            SearchResponse with results ordered by BM25 score and timings.
        """
        timings = SearchTimings()
        t0 = time.perf_counter()

        # Phase: BM25 search
        t = time.perf_counter()
        results = self.db.bm25_search(
            query=query,
            top_k=top_k,
            path_prefix=path,
        )
        timings.bm25_search_ms = (time.perf_counter() - t) * 1000

        timings.total_ms = (time.perf_counter() - t0) * 1000

        return SearchResponse(
            results=[
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
            ],
            timings=timings,
        )

    def _search_hybrid(
        self,
        query: str,
        top_k: int,
        path: str | None = None,
    ) -> SearchResponse:
        """
        Perform hybrid search using RRF fusion of semantic and keyword results.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            path: Optional path prefix filter.

        Returns:
            SearchResponse with results ordered by RRF score and timings.
        """
        timings = SearchTimings()
        t0 = time.perf_counter()

        # Get results from both search methods (fetch more to improve fusion quality)
        # Typical heuristic: fetch 2-3x top_k from each leg
        fetch_k = top_k * 3

        sem_response = self._search_semantic(query, fetch_k, path)
        timings.embedding_ms = sem_response.timings.embedding_ms
        timings.vector_search_ms = sem_response.timings.vector_search_ms

        kw_response = self._search_keyword(query, fetch_k, path)
        timings.bm25_search_ms = kw_response.timings.bm25_search_ms

        # Phase: RRF fusion
        t = time.perf_counter()

        # Build unique ID for each chunk (file_path + start_line)
        def chunk_id(result: ChunkResult) -> str:
            return f"{result.file_path}:{result.start_line}"

        # Compute RRF scores
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, ChunkResult] = {}

        # Add semantic results with RRF scoring
        for rank, result in enumerate(sem_response.results, start=1):
            cid = chunk_id(result)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.RRF_K + rank)
            chunk_map[cid] = result

        # Add keyword results with RRF scoring
        for rank, result in enumerate(kw_response.results, start=1):
            cid = chunk_id(result)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.RRF_K + rank)
            # Only add to chunk_map if not already present (prefer semantic version)
            if cid not in chunk_map:
                chunk_map[cid] = result

        # Sort by RRF score (descending) and take top_k
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        top_ids = sorted_ids[:top_k]

        # Build final results with updated scores
        fused_results = [
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

        timings.fusion_ms = (time.perf_counter() - t) * 1000
        timings.total_ms = (time.perf_counter() - t0) * 1000

        return SearchResponse(results=fused_results, timings=timings)
