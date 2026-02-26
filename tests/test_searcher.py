"""Tests for searcher.py - hybrid search with RRF fusion."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from embecode.searcher import ChunkResult, IndexNotReadyError, Searcher, SearchError


class TestChunkResult:
    """Test suite for ChunkResult."""

    def test_to_dict(self) -> None:
        """Should convert result to dictionary."""
        result = ChunkResult(
            content="def foo(): pass",
            file_path="src/main.py",
            language="python",
            start_line=10,
            end_line=15,
            context="module: main\ndefines: foo",
            score=0.95,
        )

        result_dict = result.to_dict()

        assert result_dict["content"] == "def foo(): pass"
        assert result_dict["file_path"] == "src/main.py"
        assert result_dict["language"] == "python"
        assert result_dict["start_line"] == 10
        assert result_dict["end_line"] == 15
        assert result_dict["context"] == "module: main\ndefines: foo"
        assert result_dict["score"] == 0.95


class TestSearcher:
    """Test suite for Searcher."""

    @pytest.fixture
    def mock_db(self) -> Mock:
        """Create a mock database."""
        db = Mock()
        db.get_index_stats.return_value = {
            "files_indexed": 10,
            "total_chunks": 100,
            "last_updated": "2025-02-25T10:00:00",
        }
        return db

    @pytest.fixture
    def mock_embedder(self) -> Mock:
        """Create a mock embedder."""
        embedder = Mock()
        embedder.embed.return_value = [[0.1, 0.2, 0.3]]
        return embedder

    @pytest.fixture
    def searcher(self, mock_db: Mock, mock_embedder: Mock) -> Searcher:
        """Create a searcher instance with mocked dependencies."""
        return Searcher(db=mock_db, embedder=mock_embedder)

    def test_initialization(self, mock_db: Mock, mock_embedder: Mock) -> None:
        """Should initialize with database and embedder."""
        searcher = Searcher(db=mock_db, embedder=mock_embedder)

        assert searcher.db is mock_db
        assert searcher.embedder is mock_embedder
        assert searcher.RRF_K == 60

    def test_search_invalid_mode(self, searcher: Searcher) -> None:
        """Should raise error for invalid search mode."""
        with pytest.raises(SearchError, match="Invalid search mode"):
            searcher.search("query", mode="invalid")

    def test_search_index_not_ready(self, searcher: Searcher, mock_db: Mock) -> None:
        """Should raise error when index is empty."""
        mock_db.get_index_stats.return_value = {
            "files_indexed": 0,
            "total_chunks": 0,
            "last_updated": None,
        }

        with pytest.raises(IndexNotReadyError, match="Index is not ready"):
            searcher.search("query", mode="semantic")

    def test_search_semantic(self, searcher: Searcher, mock_db: Mock, mock_embedder: Mock) -> None:
        """Should perform semantic search using vector embeddings."""
        # Mock database response
        mock_db.vector_search.return_value = [
            {
                "content": "def authenticate(user): pass",
                "file_path": "src/auth.py",
                "language": "python",
                "start_line": 10,
                "end_line": 15,
                "context": "module: auth\ndefines: authenticate",
                "score": 0.95,
            },
            {
                "content": "def login(credentials): pass",
                "file_path": "src/user.py",
                "language": "python",
                "start_line": 20,
                "end_line": 25,
                "context": "module: user\ndefines: login",
                "score": 0.85,
            },
        ]

        results = searcher.search("authentication logic", mode="semantic", top_k=2)

        # Verify embedder was called
        mock_embedder.embed.assert_called_once_with(["authentication logic"])

        # Verify database query
        mock_db.vector_search.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3],
            top_k=2,
            path_prefix=None,
        )

        # Verify results
        assert len(results) == 2
        assert results[0].file_path == "src/auth.py"
        assert results[0].score == 0.95
        assert results[1].file_path == "src/user.py"
        assert results[1].score == 0.85

    def test_search_semantic_with_path_filter(
        self, searcher: Searcher, mock_db: Mock, mock_embedder: Mock
    ) -> None:
        """Should apply path prefix filter for semantic search."""
        mock_db.vector_search.return_value = []

        searcher.search("query", mode="semantic", top_k=5, path="src/auth/")

        mock_db.vector_search.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3],
            top_k=5,
            path_prefix="src/auth/",
        )

    def test_search_keyword(self, searcher: Searcher, mock_db: Mock) -> None:
        """Should perform keyword search using BM25."""
        # Mock database response
        mock_db.bm25_search.return_value = [
            {
                "content": "class UserAuthentication: pass",
                "file_path": "src/auth.py",
                "language": "python",
                "start_line": 5,
                "end_line": 10,
                "context": "module: auth\ndefines: UserAuthentication",
                "score": 10.5,
            },
            {
                "content": "def authenticate_user(): pass",
                "file_path": "src/user.py",
                "language": "python",
                "start_line": 15,
                "end_line": 20,
                "context": "module: user\ndefines: authenticate_user",
                "score": 8.2,
            },
        ]

        results = searcher.search("authenticate", mode="keyword", top_k=2)

        # Verify database query
        mock_db.bm25_search.assert_called_once_with(
            query="authenticate",
            top_k=2,
            path_prefix=None,
        )

        # Verify results
        assert len(results) == 2
        assert results[0].file_path == "src/auth.py"
        assert results[0].score == 10.5
        assert results[1].file_path == "src/user.py"
        assert results[1].score == 8.2

    def test_search_keyword_with_path_filter(self, searcher: Searcher, mock_db: Mock) -> None:
        """Should apply path prefix filter for keyword search."""
        mock_db.bm25_search.return_value = []

        searcher.search("query", mode="keyword", top_k=5, path="src/auth/")

        mock_db.bm25_search.assert_called_once_with(
            query="query",
            top_k=5,
            path_prefix="src/auth/",
        )

    def test_search_hybrid_rrf_fusion(
        self, searcher: Searcher, mock_db: Mock, mock_embedder: Mock
    ) -> None:
        """Should combine semantic and keyword results using RRF fusion."""
        # Mock semantic results
        mock_db.vector_search.return_value = [
            {
                "content": "def auth(): pass",
                "file_path": "src/auth.py",
                "language": "python",
                "start_line": 10,
                "end_line": 15,
                "context": "module: auth",
                "score": 0.95,
            },
            {
                "content": "def login(): pass",
                "file_path": "src/login.py",
                "language": "python",
                "start_line": 20,
                "end_line": 25,
                "context": "module: login",
                "score": 0.85,
            },
            {
                "content": "def verify(): pass",
                "file_path": "src/verify.py",
                "language": "python",
                "start_line": 30,
                "end_line": 35,
                "context": "module: verify",
                "score": 0.75,
            },
        ]

        # Mock keyword results (overlapping with semantic)
        mock_db.bm25_search.return_value = [
            {
                "content": "def login(): pass",  # Same as semantic #2
                "file_path": "src/login.py",
                "language": "python",
                "start_line": 20,
                "end_line": 25,
                "context": "module: login",
                "score": 12.5,
            },
            {
                "content": "def authenticate(): pass",  # New result
                "file_path": "src/security.py",
                "language": "python",
                "start_line": 40,
                "end_line": 45,
                "context": "module: security",
                "score": 10.2,
            },
            {
                "content": "def auth(): pass",  # Same as semantic #1
                "file_path": "src/auth.py",
                "language": "python",
                "start_line": 10,
                "end_line": 15,
                "context": "module: auth",
                "score": 8.5,
            },
        ]

        results = searcher.search("authentication", mode="hybrid", top_k=3)

        # Verify both search methods were called (fetching 3x = 9 results each)
        mock_db.vector_search.assert_called_once()
        assert mock_db.vector_search.call_args[1]["top_k"] == 9
        mock_db.bm25_search.assert_called_once()
        assert mock_db.bm25_search.call_args[1]["top_k"] == 9

        # Verify RRF scoring:
        # src/auth.py:10 appears in both (rank 1 semantic, rank 3 keyword)
        #   RRF = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
        # src/login.py:20 appears in both (rank 2 semantic, rank 1 keyword)
        #   RRF = 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325  (highest!)
        # src/verify.py:30 appears only in semantic (rank 3)
        #   RRF = 1/(60+3) = 0.0159
        # src/security.py:40 appears only in keyword (rank 2)
        #   RRF = 1/(60+2) = 0.0161

        # Expected order: login (0.0325), auth (0.0323), security (0.0161)
        assert len(results) == 3
        assert results[0].file_path == "src/login.py"
        assert results[1].file_path == "src/auth.py"
        assert results[2].file_path == "src/security.py"

        # Verify scores are RRF scores (not original scores)
        assert results[0].score == pytest.approx(1 / 62 + 1 / 61, abs=0.0001)
        assert results[1].score == pytest.approx(1 / 61 + 1 / 63, abs=0.0001)
        assert results[2].score == pytest.approx(1 / 62, abs=0.0001)

    def test_search_hybrid_with_path_filter(
        self, searcher: Searcher, mock_db: Mock, mock_embedder: Mock
    ) -> None:
        """Should apply path prefix filter for hybrid search."""
        mock_db.vector_search.return_value = []
        mock_db.bm25_search.return_value = []

        searcher.search("query", mode="hybrid", top_k=5, path="src/auth/")

        # Both searches should use path filter
        mock_db.vector_search.assert_called_once()
        assert mock_db.vector_search.call_args[1]["path_prefix"] == "src/auth/"
        mock_db.bm25_search.assert_called_once()
        assert mock_db.bm25_search.call_args[1]["path_prefix"] == "src/auth/"

    def test_search_hybrid_empty_results(
        self, searcher: Searcher, mock_db: Mock, mock_embedder: Mock
    ) -> None:
        """Should handle case when both search methods return no results."""
        mock_db.vector_search.return_value = []
        mock_db.bm25_search.return_value = []

        results = searcher.search("nonexistent query", mode="hybrid", top_k=5)

        assert len(results) == 0

    def test_search_hybrid_one_empty_leg(
        self, searcher: Searcher, mock_db: Mock, mock_embedder: Mock
    ) -> None:
        """Should handle case when only one search method returns results."""
        # Only semantic has results
        mock_db.vector_search.return_value = [
            {
                "content": "def foo(): pass",
                "file_path": "src/main.py",
                "language": "python",
                "start_line": 10,
                "end_line": 15,
                "context": "module: main",
                "score": 0.9,
            },
        ]
        mock_db.bm25_search.return_value = []

        results = searcher.search("query", mode="hybrid", top_k=5)

        # Should still return the semantic result
        assert len(results) == 1
        assert results[0].file_path == "src/main.py"
        assert results[0].score == pytest.approx(1 / 61, abs=0.0001)

    def test_search_default_parameters(self, searcher: Searcher, mock_db: Mock) -> None:
        """Should use default parameters (mode=hybrid, top_k=5)."""
        mock_db.vector_search.return_value = []
        mock_db.bm25_search.return_value = []

        searcher.search("query")

        # Verify default mode is hybrid (both methods called)
        mock_db.vector_search.assert_called_once()
        mock_db.bm25_search.assert_called_once()

        # Verify default top_k=5 (but fetches 15 for hybrid)
        assert mock_db.vector_search.call_args[1]["top_k"] == 15
        assert mock_db.bm25_search.call_args[1]["top_k"] == 15
