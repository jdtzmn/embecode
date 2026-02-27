"""Tests for searcher.py - hybrid search with RRF fusion."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from embecode.searcher import (
    ChunkResult,
    IndexNotReadyError,
    Searcher,
    SearchError,
    SearchResponse,
    SearchTimings,
    _find_match_lines,
    _pick_preview_lines,
    _tokenize_query,
)


class TestChunkResult:
    """Test suite for ChunkResult."""

    def test_to_dict(self) -> None:
        """Should convert result to concise dictionary without content or context."""
        result = ChunkResult(
            content="def foo(): pass\n    return 42",
            file_path="src/main.py",
            language="python",
            start_line=10,
            end_line=15,
            definitions="function foo",
            score=0.95,
        )

        result_dict = result.to_dict()

        assert result_dict["file_path"] == "src/main.py"
        assert result_dict["language"] == "python"
        assert result_dict["start_line"] == 10
        assert result_dict["end_line"] == 15
        assert result_dict["definitions"] == "function foo"
        assert result_dict["preview"] == "def foo(): pass\n    return 42"
        assert result_dict["score"] == 0.95
        assert "content" not in result_dict
        assert "context" not in result_dict

    def test_result_preview_skips_empty_lines(self) -> None:
        """Preview should skip leading blank lines and show first 2 non-empty lines."""
        result = ChunkResult(
            content="\n\n\n    \ndef first_real_line(): pass\n\ndef second_real_line(): pass\n\ndef third_real_line(): pass",
            file_path="src/main.py",
            language="python",
            start_line=1,
            end_line=10,
            definitions="",
            score=0.5,
        )

        d = result.to_dict()
        assert d["preview"] == "def first_real_line(): pass\ndef second_real_line(): pass"

    def test_result_preview_truncated_at_200_chars(self) -> None:
        """Preview should truncate to 200 chars total with '...' when content exceeds limit."""
        # Create a line that's 210 characters long (well over 200)
        long_line = "x" * 210
        result = ChunkResult(
            content=long_line,
            file_path="src/main.py",
            language="python",
            start_line=1,
            end_line=1,
            definitions="",
            score=0.5,
        )

        d = result.to_dict()
        preview = d["preview"]
        assert len(preview) == 200
        assert preview.endswith("...")
        assert preview == "x" * 197 + "..."

    def test_result_preview_single_line_chunk(self) -> None:
        """Preview of a single non-empty line chunk should be that single line."""
        result = ChunkResult(
            content="import os\n",
            file_path="src/main.py",
            language="python",
            start_line=1,
            end_line=1,
            definitions="",
            score=0.5,
        )

        d = result.to_dict()
        assert d["preview"] == "import os"


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
                "definitions": "function authenticate",
                "score": 0.95,
            },
            {
                "content": "def login(credentials): pass",
                "file_path": "src/user.py",
                "language": "python",
                "start_line": 20,
                "end_line": 25,
                "definitions": "function login",
                "score": 0.85,
            },
        ]

        response = searcher.search("authentication logic", mode="semantic", top_k=2)

        # Verify embedder was called
        mock_embedder.embed.assert_called_once_with(["authentication logic"])

        # Verify database query
        mock_db.vector_search.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3],
            top_k=2,
            path_prefix=None,
        )

        # Verify results
        assert len(response.results) == 2
        assert response.results[0].file_path == "src/auth.py"
        assert response.results[0].score == 0.95
        assert response.results[1].file_path == "src/user.py"
        assert response.results[1].score == 0.85

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
                "definitions": "class UserAuthentication",
                "score": 10.5,
            },
            {
                "content": "def authenticate_user(): pass",
                "file_path": "src/user.py",
                "language": "python",
                "start_line": 15,
                "end_line": 20,
                "definitions": "function authenticate_user",
                "score": 8.2,
            },
        ]

        response = searcher.search("authenticate", mode="keyword", top_k=2)

        # Verify database query
        mock_db.bm25_search.assert_called_once_with(
            query="authenticate",
            top_k=2,
            path_prefix=None,
        )

        # Verify results
        assert len(response.results) == 2
        assert response.results[0].file_path == "src/auth.py"
        assert response.results[0].score == 10.5
        assert response.results[1].file_path == "src/user.py"
        assert response.results[1].score == 8.2

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
                "definitions": "",
                "score": 0.95,
            },
            {
                "content": "def login(): pass",
                "file_path": "src/login.py",
                "language": "python",
                "start_line": 20,
                "end_line": 25,
                "definitions": "",
                "score": 0.85,
            },
            {
                "content": "def verify(): pass",
                "file_path": "src/verify.py",
                "language": "python",
                "start_line": 30,
                "end_line": 35,
                "definitions": "",
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
                "definitions": "",
                "score": 12.5,
            },
            {
                "content": "def authenticate(): pass",  # New result
                "file_path": "src/security.py",
                "language": "python",
                "start_line": 40,
                "end_line": 45,
                "definitions": "",
                "score": 10.2,
            },
            {
                "content": "def auth(): pass",  # Same as semantic #1
                "file_path": "src/auth.py",
                "language": "python",
                "start_line": 10,
                "end_line": 15,
                "definitions": "",
                "score": 8.5,
            },
        ]

        response = searcher.search("authentication", mode="hybrid", top_k=3)

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
        assert len(response.results) == 3
        assert response.results[0].file_path == "src/login.py"
        assert response.results[1].file_path == "src/auth.py"
        assert response.results[2].file_path == "src/security.py"

        # Verify scores are RRF scores (not original scores)
        assert response.results[0].score == pytest.approx(1 / 62 + 1 / 61, abs=0.0001)
        assert response.results[1].score == pytest.approx(1 / 61 + 1 / 63, abs=0.0001)
        assert response.results[2].score == pytest.approx(1 / 62, abs=0.0001)

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

        response = searcher.search("nonexistent query", mode="hybrid", top_k=5)

        assert len(response.results) == 0

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
                "definitions": "",
                "score": 0.9,
            },
        ]
        mock_db.bm25_search.return_value = []

        response = searcher.search("query", mode="hybrid", top_k=5)

        # Should still return the semantic result
        assert len(response.results) == 1
        assert response.results[0].file_path == "src/main.py"
        assert response.results[0].score == pytest.approx(1 / 61, abs=0.0001)

    def test_search_default_parameters(self, searcher: Searcher, mock_db: Mock) -> None:
        """Should use default parameters (mode=hybrid, top_k=10)."""
        mock_db.vector_search.return_value = []
        mock_db.bm25_search.return_value = []

        searcher.search("query")

        # Verify default mode is hybrid (both methods called)
        mock_db.vector_search.assert_called_once()
        mock_db.bm25_search.assert_called_once()

        # Verify default top_k=10 (but fetches 30 for hybrid)
        assert mock_db.vector_search.call_args[1]["top_k"] == 30
        assert mock_db.bm25_search.call_args[1]["top_k"] == 30

    def test_search_hybrid_all_duplicates(
        self, searcher: Searcher, mock_db: Mock, mock_embedder: Mock
    ) -> None:
        """Should handle case when semantic and keyword return identical results."""
        # Both searches return the same results
        same_results = [
            {
                "content": "def foo(): pass",
                "file_path": "src/main.py",
                "language": "python",
                "start_line": 10,
                "end_line": 15,
                "definitions": "",
                "score": 0.9,
            },
            {
                "content": "def bar(): pass",
                "file_path": "src/utils.py",
                "language": "python",
                "start_line": 20,
                "end_line": 25,
                "definitions": "",
                "score": 0.8,
            },
        ]
        mock_db.vector_search.return_value = same_results
        mock_db.bm25_search.return_value = same_results

        response = searcher.search("query", mode="hybrid", top_k=2)

        # Should deduplicate and boost scores via RRF
        # Each result appears in both rank 1 and rank 2
        # foo: RRF = 1/(60+1) + 1/(60+1) = 2 * 1/61 = 0.0328
        # bar: RRF = 1/(60+2) + 1/(60+2) = 2 * 1/62 = 0.0323
        assert len(response.results) == 2
        assert response.results[0].file_path == "src/main.py"
        assert response.results[1].file_path == "src/utils.py"
        assert response.results[0].score == pytest.approx(2 / 61, abs=0.0001)
        assert response.results[1].score == pytest.approx(2 / 62, abs=0.0001)

    def test_search_hybrid_different_chunks_same_location(
        self, searcher: Searcher, mock_db: Mock, mock_embedder: Mock
    ) -> None:
        """Should use chunk ID (file_path:start_line) to identify duplicates."""
        # Same location (file_path:start_line) but different content/context
        mock_db.vector_search.return_value = [
            {
                "content": "def foo(): pass",
                "file_path": "src/main.py",
                "language": "python",
                "start_line": 10,
                "end_line": 15,
                "definitions": "function foo",
                "score": 0.95,
            },
        ]
        mock_db.bm25_search.return_value = [
            {
                "content": "def foo(): return True",  # Different content
                "file_path": "src/main.py",
                "language": "python",
                "start_line": 10,  # Same start line
                "end_line": 15,
                "definitions": "function foo",  # Different definitions
                "score": 12.5,
            },
        ]

        response = searcher.search("query", mode="hybrid", top_k=1)

        # Should treat as duplicate and use semantic version (added first)
        assert len(response.results) == 1
        assert response.results[0].content == "def foo(): pass"  # From semantic
        assert response.results[0].definitions == "function foo"  # From semantic
        # RRF score combines both: 1/(60+1) + 1/(60+1) = 2/61
        assert response.results[0].score == pytest.approx(2 / 61, abs=0.0001)

    def test_search_hybrid_top_k_smaller_than_results(
        self, searcher: Searcher, mock_db: Mock, mock_embedder: Mock
    ) -> None:
        """Should return only top_k results even when more are available."""
        # Create 5 unique results from each search
        mock_db.vector_search.return_value = [
            {
                "content": f"def func{i}(): pass",
                "file_path": f"src/file{i}.py",
                "language": "python",
                "start_line": i * 10,
                "end_line": i * 10 + 5,
                "definitions": "",
                "score": 1.0 - (i * 0.1),
            }
            for i in range(5)
        ]
        mock_db.bm25_search.return_value = [
            {
                "content": f"def func{i + 5}(): pass",
                "file_path": f"src/file{i + 5}.py",
                "language": "python",
                "start_line": (i + 5) * 10,
                "end_line": (i + 5) * 10 + 5,
                "definitions": "",
                "score": 10.0 - i,
            }
            for i in range(5)
        ]

        response = searcher.search("query", mode="hybrid", top_k=3)

        # Should return exactly 3 results
        assert len(response.results) == 3
        # All results should have RRF scores
        for result in response.results:
            assert result.score > 0

    def test_search_hybrid_rrf_k_constant(self, searcher: Searcher) -> None:
        """Should use RRF_K constant of 60 for fusion."""
        assert searcher.RRF_K == 60

    def test_search_semantic_empty_results(
        self, searcher: Searcher, mock_db: Mock, mock_embedder: Mock
    ) -> None:
        """Should return empty list when semantic search finds no results."""
        mock_db.vector_search.return_value = []

        response = searcher.search("nonexistent", mode="semantic", top_k=5)

        assert len(response.results) == 0
        assert response.results == []

    def test_search_keyword_empty_results(self, searcher: Searcher, mock_db: Mock) -> None:
        """Should return empty list when keyword search finds no results."""
        mock_db.bm25_search.return_value = []

        response = searcher.search("nonexistent", mode="keyword", top_k=5)

        assert len(response.results) == 0
        assert response.results == []

    def test_search_hybrid_large_top_k(
        self, searcher: Searcher, mock_db: Mock, mock_embedder: Mock
    ) -> None:
        """Should handle large top_k values correctly."""
        mock_db.vector_search.return_value = []
        mock_db.bm25_search.return_value = []

        searcher.search("query", mode="hybrid", top_k=100)

        # Should fetch 3x top_k = 300 from each leg
        mock_db.vector_search.assert_called_once()
        assert mock_db.vector_search.call_args[1]["top_k"] == 300
        mock_db.bm25_search.assert_called_once()
        assert mock_db.bm25_search.call_args[1]["top_k"] == 300

    def test_chunk_result_all_fields(self) -> None:
        """Should preserve all fields in ChunkResult."""
        result = ChunkResult(
            content="test content",
            file_path="/path/to/file.py",
            language="python",
            start_line=42,
            end_line=100,
            definitions="function test",
            score=0.123456,
        )

        assert result.content == "test content"
        assert result.file_path == "/path/to/file.py"
        assert result.language == "python"
        assert result.start_line == 42
        assert result.end_line == 100
        assert result.definitions == "function test"
        assert result.score == 0.123456

    def test_search_exceptions_are_searcherror_subclass(self) -> None:
        """Should verify exception hierarchy."""
        assert issubclass(IndexNotReadyError, SearchError)
        assert issubclass(SearchError, Exception)

    # --- New timing tests ---

    def test_search_returns_search_response(self, searcher: Searcher, mock_db: Mock) -> None:
        """Searcher.search() should return a SearchResponse with .results and .timings."""
        mock_db.bm25_search.return_value = []
        mock_db.vector_search.return_value = []

        response = searcher.search("query", mode="keyword")

        assert isinstance(response, SearchResponse)
        assert hasattr(response, "results")
        assert hasattr(response, "timings")
        assert isinstance(response.results, list)
        assert isinstance(response.timings, SearchTimings)

    def test_timings_hybrid_has_all_phases(
        self, searcher: Searcher, mock_db: Mock, mock_embedder: Mock
    ) -> None:
        """Hybrid search should populate all timing fields > 0."""
        mock_db.vector_search.return_value = [
            {
                "content": "def foo(): pass",
                "file_path": "src/a.py",
                "language": "python",
                "start_line": 1,
                "end_line": 5,
                "definitions": "",
                "score": 0.9,
            }
        ]
        mock_db.bm25_search.return_value = [
            {
                "content": "def bar(): pass",
                "file_path": "src/b.py",
                "language": "python",
                "start_line": 1,
                "end_line": 5,
                "definitions": "",
                "score": 5.0,
            }
        ]

        response = searcher.search("query", mode="hybrid", top_k=2)

        t = response.timings
        assert t.embedding_ms > 0
        assert t.vector_search_ms > 0
        assert t.bm25_search_ms > 0
        assert t.fusion_ms > 0
        assert t.total_ms > 0

    def test_timings_semantic_has_embedding_and_vector(
        self, searcher: Searcher, mock_db: Mock, mock_embedder: Mock
    ) -> None:
        """Semantic search should populate embedding_ms and vector_search_ms > 0; others remain 0."""
        mock_db.vector_search.return_value = []

        response = searcher.search("query", mode="semantic", top_k=5)

        t = response.timings
        assert t.embedding_ms > 0
        assert t.vector_search_ms > 0
        assert t.bm25_search_ms == 0.0
        assert t.fusion_ms == 0.0
        assert t.total_ms > 0

    def test_timings_keyword_has_bm25_only(self, searcher: Searcher, mock_db: Mock) -> None:
        """Keyword search should populate bm25_search_ms > 0; other phase fields remain 0."""
        mock_db.bm25_search.return_value = []

        response = searcher.search("query", mode="keyword", top_k=5)

        t = response.timings
        assert t.bm25_search_ms > 0
        assert t.embedding_ms == 0.0
        assert t.vector_search_ms == 0.0
        assert t.fusion_ms == 0.0
        assert t.total_ms > 0

    def test_timings_total_gte_sum_of_parts(
        self, searcher: Searcher, mock_db: Mock, mock_embedder: Mock
    ) -> None:
        """total_ms should be >= sum of all phase durations."""
        mock_db.vector_search.return_value = []
        mock_db.bm25_search.return_value = []

        response = searcher.search("query", mode="hybrid", top_k=5)

        t = response.timings
        phase_sum = t.embedding_ms + t.vector_search_ms + t.bm25_search_ms + t.fusion_ms
        assert t.total_ms >= phase_sum

    def test_timings_to_dict_rounds_to_two_decimals(self) -> None:
        """SearchTimings.to_dict() values should be rounded to 2 decimal places."""
        timings = SearchTimings(
            embedding_ms=12.3456789,
            vector_search_ms=0.0012345,
            bm25_search_ms=99.9999,
            fusion_ms=1.005,
            total_ms=113.351789,
        )

        d = timings.to_dict()

        assert d["embedding_ms"] == round(12.3456789, 2)
        assert d["vector_search_ms"] == round(0.0012345, 2)
        assert d["bm25_search_ms"] == round(99.9999, 2)
        assert d["fusion_ms"] == round(1.005, 2)
        assert d["total_ms"] == round(113.351789, 2)
        # Verify they are indeed rounded (at most 2 decimal places)
        for key, val in d.items():
            assert val == round(val, 2), f"{key} not rounded to 2 decimals"

    def test_timings_logged_at_info_level(
        self, searcher: Searcher, mock_db: Mock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """After calling search(), a log record at INFO level should be emitted
        containing the query text, mode, and timing dict."""
        mock_db.bm25_search.return_value = []
        mock_db.vector_search.return_value = []

        import logging

        with caplog.at_level(logging.INFO, logger="embecode.searcher"):
            searcher.search("my test query", mode="keyword", top_k=5)

        assert len(caplog.records) >= 1
        record = caplog.records[-1]
        assert record.levelno == logging.INFO
        assert "my test query" in record.getMessage()
        assert "keyword" in record.getMessage()
        # Timing dict keys should appear in the message
        assert "bm25_search_ms" in record.getMessage()


class TestTokenizeQuery:
    """Tests for _tokenize_query()."""

    def test_tokenize_query_basic(self) -> None:
        """Standard multi-word query produces lowercase tokens."""
        assert _tokenize_query("UserAvatar component") == ["useravatar", "component"]

    def test_tokenize_query_drops_single_chars(self) -> None:
        """Single-character tokens are dropped as too noisy."""
        assert _tokenize_query("a b foo") == ["foo"]

    def test_tokenize_query_deduplicates(self) -> None:
        """Duplicate tokens keep only the first occurrence."""
        assert _tokenize_query("foo bar foo") == ["foo", "bar"]

    def test_tokenize_query_empty_string(self) -> None:
        """Empty string produces empty list."""
        assert _tokenize_query("") == []

    def test_tokenize_query_special_characters(self) -> None:
        """Non-word characters act as separators."""
        assert _tokenize_query("user-avatar.tsx") == ["user", "avatar", "tsx"]


class TestPickPreviewLines:
    """Tests for _pick_preview_lines() — match-aware preview selection."""

    def test_preview_shows_matching_lines_when_query_provided(self) -> None:
        """Preview prefers lines containing query tokens over chunk-opening lines."""
        content = (
            "import React from 'react'\n"
            "import { cn } from '@/lib/utils'\n"
            "\n"
            "// Component for user avatars\n"
            "export function UserAvatar({\n"
        )
        preview = _pick_preview_lines(content, query="UserAvatar")
        assert "UserAvatar" in preview
        # Should NOT start with the import line
        assert not preview.startswith("import React")

    def test_preview_falls_back_when_no_lexical_match(self) -> None:
        """Semantic-only query with no text overlap falls back to first 2 lines."""
        content = 'def verify_credentials(user, password):\n    """Verify user credentials."""\n    return True\n'
        preview = _pick_preview_lines(content, query="authentication logic")
        assert (
            preview == 'def verify_credentials(user, password):\n    """Verify user credentials."""'
        )

    def test_preview_falls_back_when_no_query(self) -> None:
        """query=None uses first-2-lines fallback."""
        content = "line one\nline two\nline three\n"
        preview = _pick_preview_lines(content, query=None)
        assert preview == "line one\nline two"

    def test_preview_case_insensitive_match(self) -> None:
        """Matching is case-insensitive."""
        content = "import os\nexport function UserAvatar() {\n"
        preview = _pick_preview_lines(content, query="useravatar")
        assert "UserAvatar" in preview

    def test_preview_picks_best_match_and_nearest(self) -> None:
        """Best match (highest count) + nearest other match are selected."""
        lines = [
            "line 0",
            "line 1",
            "line 2 foo",
            "line 3",
            "line 4",
            "line 5",
            "line 6",
            "line 7 foo",
            "line 8",
            "line 9",
            "line 10",
            "line 11",
            "line 12",
            "line 13",
            "line 14 foo",
        ]
        content = "\n".join(lines)
        preview = _pick_preview_lines(content, query="foo")
        # Best match is line 2 (first, all have count 1), nearest other is line 7
        assert "line 2 foo" in preview
        assert "line 7 foo" in preview

    def test_preview_single_match_with_context(self) -> None:
        """Single matching line picks a non-empty neighbour as context."""
        lines = [
            "line 0",
            "line 1",
            "line 2",
            "line 3",
            "line 4 target",
            "line 5",
            "line 6",
            "line 7",
            "line 8",
            "line 9",
        ]
        content = "\n".join(lines)
        preview = _pick_preview_lines(content, query="target")
        assert "line 4 target" in preview
        # Should include a neighbor (line 3 or line 5)
        lines_in_preview = preview.split("\n")
        assert len(lines_in_preview) == 2

    def test_preview_match_preserves_200_char_limit(self) -> None:
        """Matching lines longer than 200 chars are truncated."""
        long_line = "target " + "x" * 250
        content = "short line\n" + long_line
        preview = _pick_preview_lines(content, query="target")
        assert len(preview) <= 200
        assert preview.endswith("...")


class TestFindMatchLines:
    """Tests for _find_match_lines()."""

    def test_match_lines_absolute_line_numbers(self) -> None:
        """Lines are numbered from start_line, not zero."""
        content = "alpha foo beta\ngamma\ndelta foo\n"
        result = _find_match_lines(content, "foo", start_line=10)
        assert result == [10, 12]

    def test_match_lines_empty_when_no_match(self) -> None:
        """Returns empty list when no tokens match."""
        content = "def verify_credentials():\n    pass\n"
        result = _find_match_lines(content, "authentication logic", start_line=1)
        assert result == []

    def test_match_lines_capped_at_8(self) -> None:
        """At most 8 match lines are returned."""
        # 20 lines all containing the token
        content = "\n".join(f"line {i} foo" for i in range(20))
        result = _find_match_lines(content, "foo", start_line=1)
        assert len(result) == 8

    def test_match_lines_empty_query_tokens(self) -> None:
        """Query with only single-char tokens returns empty list."""
        content = "a b c d e\n"
        result = _find_match_lines(content, "a b c", start_line=1)
        assert result == []


class TestChunkResultPreviewWithQuery:
    """Tests for ChunkResult.preview(query=...) integration."""

    def test_preview_with_query_shows_matching_line(self) -> None:
        """ChunkResult.preview(query=...) delegates to _pick_preview_lines."""
        result = ChunkResult(
            content="import os\nimport sys\ndef UserAvatar(): pass\n",
            file_path="test.py",
            language="python",
            start_line=1,
            end_line=3,
            definitions="",
            score=0.5,
        )
        preview = result.preview(query="UserAvatar")
        assert "UserAvatar" in preview

    def test_preview_without_query_uses_fallback(self) -> None:
        """ChunkResult.preview() without query still uses first 2 non-empty lines."""
        result = ChunkResult(
            content="first line\nsecond line\nthird line\n",
            file_path="test.py",
            language="python",
            start_line=1,
            end_line=3,
            definitions="",
            score=0.5,
        )
        assert result.preview() == "first line\nsecond line"

    def test_to_dict_with_query_has_match_lines(self) -> None:
        """to_dict(query=...) includes match_lines when there are lexical matches."""
        result = ChunkResult(
            content="import os\ndef foo(): pass\nreturn foo\n",
            file_path="test.py",
            language="python",
            start_line=10,
            end_line=12,
            definitions="",
            score=0.5,
        )
        d = result.to_dict(query="foo")
        assert "match_lines" in d
        assert d["match_lines"] == [11, 12]

    def test_to_dict_without_query_no_match_lines(self) -> None:
        """to_dict() without query omits match_lines key entirely."""
        result = ChunkResult(
            content="import os\ndef foo(): pass\n",
            file_path="test.py",
            language="python",
            start_line=1,
            end_line=2,
            definitions="",
            score=0.5,
        )
        d = result.to_dict()
        assert "match_lines" not in d

    def test_to_dict_match_lines_omitted_when_empty(self) -> None:
        """to_dict(query=...) omits match_lines when no tokens match content."""
        result = ChunkResult(
            content="def verify_credentials():\n    pass\n",
            file_path="test.py",
            language="python",
            start_line=1,
            end_line=2,
            definitions="",
            score=0.5,
        )
        d = result.to_dict(query="authentication logic")
        assert "match_lines" not in d

    def test_to_dict_with_query_uses_match_aware_preview(self) -> None:
        """to_dict(query=...) generates a match-aware preview."""
        result = ChunkResult(
            content="import os\nimport sys\ndef UserAvatar(): pass\n",
            file_path="test.py",
            language="python",
            start_line=1,
            end_line=3,
            definitions="",
            score=0.5,
        )
        d = result.to_dict(query="UserAvatar")
        assert "UserAvatar" in d["preview"]
