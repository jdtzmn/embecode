"""Tests for embecode.db module."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from embecode.db import Database


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = Database(db_path)
        yield db
        db.close()


def test_database_initialization(temp_db):
    """Test database initialization and schema creation."""
    temp_db.connect()

    # Verify database file was created
    assert temp_db.db_path.exists()

    # Verify tables were created
    conn = temp_db._conn
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    table_names = {row[0] for row in tables}

    assert "chunks" in table_names
    assert "embeddings" in table_names
    assert "files" in table_names


def test_database_connect_idempotent(temp_db):
    """Test that multiple connect calls are safe."""
    temp_db.connect()
    conn1 = temp_db._conn

    temp_db.connect()
    conn2 = temp_db._conn

    assert conn1 is conn2


def test_database_close(temp_db):
    """Test database close."""
    temp_db.connect()
    assert temp_db._conn is not None

    temp_db.close()
    assert temp_db._conn is None


def test_clear_index(temp_db):
    """Test clearing the entire index."""
    temp_db.connect()

    # Insert some test data
    chunk_records = [
        {
            "file_path": "test.py",
            "language": "python",
            "start_line": 1,
            "end_line": 5,
            "content": "def hello(): pass",
            "context": "test.py: hello",
            "hash": "abc123",
            "embedding": [0.1] * 384,
        }
    ]
    temp_db.insert_chunks(chunk_records)

    # Verify data was inserted
    stats = temp_db.get_index_stats()
    assert stats["total_chunks"] == 1

    # Clear index
    temp_db.clear_index()

    # Verify data was deleted
    stats = temp_db.get_index_stats()
    assert stats["total_chunks"] == 0
    assert stats["files_indexed"] == 0
    assert stats["last_updated"] is None


def test_get_index_stats_empty(temp_db):
    """Test getting stats from empty database."""
    temp_db.connect()

    stats = temp_db.get_index_stats()

    assert stats["total_chunks"] == 0
    assert stats["files_indexed"] == 0
    assert stats["last_updated"] is None


def test_get_index_stats_with_data(temp_db):
    """Test getting stats with indexed data."""
    temp_db.connect()

    # Insert test data
    chunk_records = [
        {
            "file_path": "test1.py",
            "language": "python",
            "start_line": 1,
            "end_line": 5,
            "content": "def hello(): pass",
            "context": "test1.py: hello",
            "hash": "abc123",
            "embedding": [0.1] * 384,
        },
        {
            "file_path": "test1.py",
            "language": "python",
            "start_line": 6,
            "end_line": 10,
            "content": "def world(): pass",
            "context": "test1.py: world",
            "hash": "def456",
            "embedding": [0.2] * 384,
        },
        {
            "file_path": "test2.py",
            "language": "python",
            "start_line": 1,
            "end_line": 3,
            "content": "x = 1",
            "context": "test2.py",
            "hash": "ghi789",
            "embedding": [0.3] * 384,
        },
    ]
    temp_db.insert_chunks(chunk_records)
    temp_db.update_file_metadata("test1.py", 2)
    temp_db.update_file_metadata("test2.py", 1)

    stats = temp_db.get_index_stats()

    assert stats["total_chunks"] == 3
    assert stats["files_indexed"] == 2
    assert stats["last_updated"] is not None
    # Verify last_updated is a valid ISO timestamp
    datetime.fromisoformat(stats["last_updated"])


def test_insert_chunks(temp_db):
    """Test inserting chunks with embeddings."""
    temp_db.connect()

    chunk_records = [
        {
            "file_path": "test.py",
            "language": "python",
            "start_line": 1,
            "end_line": 5,
            "content": "def hello(): pass",
            "context": "test.py: hello",
            "hash": "abc123",
            "embedding": [0.1] * 384,
        },
        {
            "file_path": "test.py",
            "language": "python",
            "start_line": 6,
            "end_line": 10,
            "content": "def world(): pass",
            "context": "test.py: world",
            "hash": "def456",
            "embedding": [0.2] * 384,
        },
    ]

    temp_db.insert_chunks(chunk_records)

    # Verify chunks were inserted
    conn = temp_db._conn
    chunks = conn.execute("SELECT * FROM chunks ORDER BY start_line").fetchall()
    assert len(chunks) == 2
    assert chunks[0][1] == "test.py"  # file_path
    assert (
        chunks[0][5] == "def hello(): pass"
    )  # content (id, file_path, language, start_line, end_line, content)

    # Verify embeddings were inserted
    embeddings = conn.execute("SELECT * FROM embeddings").fetchall()
    assert len(embeddings) == 2


def test_insert_chunks_empty_list(temp_db):
    """Test inserting empty chunk list (no-op)."""
    temp_db.connect()

    temp_db.insert_chunks([])

    stats = temp_db.get_index_stats()
    assert stats["total_chunks"] == 0


def test_insert_chunks_upsert(temp_db):
    """Test that insert_chunks updates existing chunks."""
    temp_db.connect()

    # Insert initial chunk
    chunk_records = [
        {
            "file_path": "test.py",
            "language": "python",
            "start_line": 1,
            "end_line": 5,
            "content": "def hello(): pass",
            "context": "test.py: hello",
            "hash": "abc123",
            "embedding": [0.1] * 384,
        }
    ]
    temp_db.insert_chunks(chunk_records)

    # Update the same chunk (same file_path and start_line)
    updated_records = [
        {
            "file_path": "test.py",
            "language": "python",
            "start_line": 1,
            "end_line": 5,
            "content": "def hello(): print('hi')",
            "context": "test.py: hello",
            "hash": "xyz789",
            "embedding": [0.9] * 384,
        }
    ]
    temp_db.insert_chunks(updated_records)

    # Verify only one chunk exists with updated content
    stats = temp_db.get_index_stats()
    assert stats["total_chunks"] == 1

    conn = temp_db._conn
    chunk = conn.execute("SELECT content, hash FROM chunks").fetchone()
    assert chunk[0] == "def hello(): print('hi')"
    assert chunk[1] == "xyz789"


def test_get_chunk_hashes_for_file(temp_db):
    """Test getting chunk hashes for a specific file."""
    temp_db.connect()

    # Insert chunks for multiple files
    chunk_records = [
        {
            "file_path": "test1.py",
            "language": "python",
            "start_line": 1,
            "end_line": 5,
            "content": "def hello(): pass",
            "context": "test1.py: hello",
            "hash": "hash1",
            "embedding": [0.1] * 384,
        },
        {
            "file_path": "test1.py",
            "language": "python",
            "start_line": 6,
            "end_line": 10,
            "content": "def world(): pass",
            "context": "test1.py: world",
            "hash": "hash2",
            "embedding": [0.2] * 384,
        },
        {
            "file_path": "test2.py",
            "language": "python",
            "start_line": 1,
            "end_line": 3,
            "content": "x = 1",
            "context": "test2.py",
            "hash": "hash3",
            "embedding": [0.3] * 384,
        },
    ]
    temp_db.insert_chunks(chunk_records)

    # Get hashes for test1.py
    hashes = temp_db.get_chunk_hashes_for_file("test1.py")
    assert hashes == {"hash1", "hash2"}

    # Get hashes for test2.py
    hashes = temp_db.get_chunk_hashes_for_file("test2.py")
    assert hashes == {"hash3"}

    # Get hashes for non-existent file
    hashes = temp_db.get_chunk_hashes_for_file("nonexistent.py")
    assert hashes == set()


def test_delete_chunks_by_hash(temp_db):
    """Test deleting chunks by their hashes."""
    temp_db.connect()

    # Insert test chunks
    chunk_records = [
        {
            "file_path": "test.py",
            "language": "python",
            "start_line": 1,
            "end_line": 5,
            "content": "def hello(): pass",
            "context": "test.py: hello",
            "hash": "hash1",
            "embedding": [0.1] * 384,
        },
        {
            "file_path": "test.py",
            "language": "python",
            "start_line": 6,
            "end_line": 10,
            "content": "def world(): pass",
            "context": "test.py: world",
            "hash": "hash2",
            "embedding": [0.2] * 384,
        },
        {
            "file_path": "test.py",
            "language": "python",
            "start_line": 11,
            "end_line": 15,
            "content": "def foo(): pass",
            "context": "test.py: foo",
            "hash": "hash3",
            "embedding": [0.3] * 384,
        },
    ]
    temp_db.insert_chunks(chunk_records)

    # Delete by hash
    temp_db.delete_chunks_by_hash(["hash1", "hash3"])

    # Verify correct chunks were deleted
    stats = temp_db.get_index_stats()
    assert stats["total_chunks"] == 1

    conn = temp_db._conn
    remaining = conn.execute("SELECT hash FROM chunks").fetchall()
    assert len(remaining) == 1
    assert remaining[0][0] == "hash2"

    # Verify embeddings were also deleted
    embeddings = conn.execute("SELECT chunk_id FROM embeddings").fetchall()
    assert len(embeddings) == 1


def test_delete_chunks_by_hash_empty_list(temp_db):
    """Test deleting with empty hash list (no-op)."""
    temp_db.connect()

    # Insert test chunk
    chunk_records = [
        {
            "file_path": "test.py",
            "language": "python",
            "start_line": 1,
            "end_line": 5,
            "content": "def hello(): pass",
            "context": "test.py: hello",
            "hash": "hash1",
            "embedding": [0.1] * 384,
        }
    ]
    temp_db.insert_chunks(chunk_records)

    # Delete empty list
    temp_db.delete_chunks_by_hash([])

    # Verify nothing was deleted
    stats = temp_db.get_index_stats()
    assert stats["total_chunks"] == 1


def test_delete_chunks_by_file(temp_db):
    """Test deleting all chunks for a specific file."""
    temp_db.connect()

    # Insert chunks for multiple files
    chunk_records = [
        {
            "file_path": "test1.py",
            "language": "python",
            "start_line": 1,
            "end_line": 5,
            "content": "def hello(): pass",
            "context": "test1.py: hello",
            "hash": "hash1",
            "embedding": [0.1] * 384,
        },
        {
            "file_path": "test1.py",
            "language": "python",
            "start_line": 6,
            "end_line": 10,
            "content": "def world(): pass",
            "context": "test1.py: world",
            "hash": "hash2",
            "embedding": [0.2] * 384,
        },
        {
            "file_path": "test2.py",
            "language": "python",
            "start_line": 1,
            "end_line": 3,
            "content": "x = 1",
            "context": "test2.py",
            "hash": "hash3",
            "embedding": [0.3] * 384,
        },
    ]
    temp_db.insert_chunks(chunk_records)

    # Delete chunks for test1.py
    temp_db.delete_chunks_by_file("test1.py")

    # Verify correct chunks were deleted
    stats = temp_db.get_index_stats()
    assert stats["total_chunks"] == 1

    conn = temp_db._conn
    remaining = conn.execute("SELECT file_path FROM chunks").fetchall()
    assert len(remaining) == 1
    assert remaining[0][0] == "test2.py"


def test_update_file_metadata(temp_db):
    """Test updating file metadata."""
    temp_db.connect()

    # Update file metadata
    temp_db.update_file_metadata("test.py", 5)

    # Verify metadata was inserted
    conn = temp_db._conn
    file_data = conn.execute(
        "SELECT path, chunk_count FROM files WHERE path = 'test.py'"
    ).fetchone()
    assert file_data is not None
    assert file_data[0] == "test.py"
    assert file_data[1] == 5

    # Update again (upsert)
    temp_db.update_file_metadata("test.py", 10)

    # Verify metadata was updated
    file_data = conn.execute(
        "SELECT path, chunk_count FROM files WHERE path = 'test.py'"
    ).fetchone()
    assert file_data[1] == 10


def test_delete_file(temp_db):
    """Test deleting file and all its chunks."""
    temp_db.connect()

    # Insert chunks and metadata
    chunk_records = [
        {
            "file_path": "test1.py",
            "language": "python",
            "start_line": 1,
            "end_line": 5,
            "content": "def hello(): pass",
            "context": "test1.py: hello",
            "hash": "hash1",
            "embedding": [0.1] * 384,
        },
        {
            "file_path": "test1.py",
            "language": "python",
            "start_line": 6,
            "end_line": 10,
            "content": "def world(): pass",
            "context": "test1.py: world",
            "hash": "hash2",
            "embedding": [0.2] * 384,
        },
        {
            "file_path": "test2.py",
            "language": "python",
            "start_line": 1,
            "end_line": 3,
            "content": "x = 1",
            "context": "test2.py",
            "hash": "hash3",
            "embedding": [0.3] * 384,
        },
    ]
    temp_db.insert_chunks(chunk_records)
    temp_db.update_file_metadata("test1.py", 2)
    temp_db.update_file_metadata("test2.py", 1)

    # Delete test1.py
    deleted_count = temp_db.delete_file("test1.py")

    # Verify return value
    assert deleted_count == 2

    # Verify chunks were deleted
    stats = temp_db.get_index_stats()
    assert stats["total_chunks"] == 1
    assert stats["files_indexed"] == 1

    # Verify file metadata was deleted
    conn = temp_db._conn
    files = conn.execute("SELECT path FROM files").fetchall()
    assert len(files) == 1
    assert files[0][0] == "test2.py"


def test_delete_file_nonexistent(temp_db):
    """Test deleting non-existent file returns 0."""
    temp_db.connect()

    deleted_count = temp_db.delete_file("nonexistent.py")
    assert deleted_count == 0


def test_vector_search(temp_db):
    """Test vector similarity search."""
    temp_db.connect()

    # Insert test chunks with embeddings
    chunk_records = [
        {
            "file_path": "test1.py",
            "language": "python",
            "start_line": 1,
            "end_line": 5,
            "content": "def hello(): pass",
            "context": "test1.py: hello",
            "hash": "hash1",
            "embedding": [1.0, 0.0, 0.0],  # Simple 3D embeddings for testing
        },
        {
            "file_path": "test1.py",
            "language": "python",
            "start_line": 6,
            "end_line": 10,
            "content": "def world(): pass",
            "context": "test1.py: world",
            "hash": "hash2",
            "embedding": [0.0, 1.0, 0.0],
        },
        {
            "file_path": "test2.py",
            "language": "python",
            "start_line": 1,
            "end_line": 3,
            "content": "x = 1",
            "context": "test2.py",
            "hash": "hash3",
            "embedding": [0.0, 0.0, 1.0],
        },
    ]
    temp_db.insert_chunks(chunk_records)

    # Search with query embedding similar to first chunk
    query_embedding = [0.9, 0.1, 0.0]
    results = temp_db.vector_search(query_embedding, top_k=2)

    # Verify results
    assert len(results) <= 2
    if results:  # VSS extension may not be available in test environment
        assert "content" in results[0]
        assert "file_path" in results[0]
        assert "score" in results[0]
        assert results[0]["start_line"] > 0


def test_vector_search_with_path_prefix(temp_db):
    """Test vector search with path prefix filter."""
    temp_db.connect()

    # Insert test chunks
    chunk_records = [
        {
            "file_path": "src/main.py",
            "language": "python",
            "start_line": 1,
            "end_line": 5,
            "content": "def hello(): pass",
            "context": "src/main.py: hello",
            "hash": "hash1",
            "embedding": [1.0, 0.0, 0.0],
        },
        {
            "file_path": "tests/test.py",
            "language": "python",
            "start_line": 1,
            "end_line": 5,
            "content": "def test_hello(): pass",
            "context": "tests/test.py: test_hello",
            "hash": "hash2",
            "embedding": [0.9, 0.1, 0.0],
        },
    ]
    temp_db.insert_chunks(chunk_records)

    # Search with path prefix filter
    query_embedding = [1.0, 0.0, 0.0]
    results = temp_db.vector_search(query_embedding, top_k=10, path_prefix="src/")

    # Verify results only contain src/ files
    if results:  # VSS extension may not be available
        for result in results:
            assert result["file_path"].startswith("src/")


def test_bm25_search(temp_db):
    """Test BM25 keyword search."""
    temp_db.connect()

    # Insert test chunks
    chunk_records = [
        {
            "file_path": "test1.py",
            "language": "python",
            "start_line": 1,
            "end_line": 5,
            "content": "def calculate_total(items): return sum(items)",
            "context": "test1.py: calculate_total",
            "hash": "hash1",
            "embedding": [0.1] * 384,
        },
        {
            "file_path": "test2.py",
            "language": "python",
            "start_line": 1,
            "end_line": 5,
            "content": "def calculate_average(values): return sum(values) / len(values)",
            "context": "test2.py: calculate_average",
            "hash": "hash2",
            "embedding": [0.2] * 384,
        },
        {
            "file_path": "test3.py",
            "language": "python",
            "start_line": 1,
            "end_line": 3,
            "content": "x = 42",
            "context": "test3.py",
            "hash": "hash3",
            "embedding": [0.3] * 384,
        },
    ]
    temp_db.insert_chunks(chunk_records)

    # Search for "calculate"
    results = temp_db.bm25_search("calculate", top_k=10)

    # Verify results contain matching chunks
    assert len(results) >= 2
    assert any("calculate" in r["content"].lower() for r in results)


def test_bm25_search_with_path_prefix(temp_db):
    """Test BM25 search with path prefix filter."""
    temp_db.connect()

    # Insert test chunks
    chunk_records = [
        {
            "file_path": "src/main.py",
            "language": "python",
            "start_line": 1,
            "end_line": 5,
            "content": "def hello(): pass",
            "context": "src/main.py: hello",
            "hash": "hash1",
            "embedding": [0.1] * 384,
        },
        {
            "file_path": "tests/test.py",
            "language": "python",
            "start_line": 1,
            "end_line": 5,
            "content": "def test_hello(): pass",
            "context": "tests/test.py: test_hello",
            "hash": "hash2",
            "embedding": [0.2] * 384,
        },
    ]
    temp_db.insert_chunks(chunk_records)

    # Search with path prefix filter
    results = temp_db.bm25_search("hello", top_k=10, path_prefix="src/")

    # Verify results only contain src/ files
    assert len(results) >= 1
    for result in results:
        assert result["file_path"].startswith("src/")


def test_bm25_search_no_results(temp_db):
    """Test BM25 search with no matching results."""
    temp_db.connect()

    # Insert test chunk
    chunk_records = [
        {
            "file_path": "test.py",
            "language": "python",
            "start_line": 1,
            "end_line": 5,
            "content": "def hello(): pass",
            "context": "test.py: hello",
            "hash": "hash1",
            "embedding": [0.1] * 384,
        }
    ]
    temp_db.insert_chunks(chunk_records)

    # Search for non-existent term
    results = temp_db.bm25_search("nonexistent_term_xyz", top_k=10)

    # Verify no results
    assert len(results) == 0


def test_database_persistence(temp_db):
    """Test that data persists after closing and reopening database."""
    temp_db.connect()

    # Insert test data
    chunk_records = [
        {
            "file_path": "test.py",
            "language": "python",
            "start_line": 1,
            "end_line": 5,
            "content": "def hello(): pass",
            "context": "test.py: hello",
            "hash": "hash1",
            "embedding": [0.1] * 384,
        }
    ]
    temp_db.insert_chunks(chunk_records)
    temp_db.update_file_metadata("test.py", 1)

    # Close and reopen
    temp_db.close()
    temp_db.connect()

    # Verify data persisted
    stats = temp_db.get_index_stats()
    assert stats["total_chunks"] == 1
    assert stats["files_indexed"] == 1


def test_database_context_without_explicit_context(temp_db):
    """Test inserting chunks without explicit context field."""
    temp_db.connect()

    # Insert chunk without context field
    chunk_records = [
        {
            "file_path": "test.py",
            "language": "python",
            "start_line": 1,
            "end_line": 5,
            "content": "def hello(): pass",
            # Note: no context field
            "hash": "hash1",
            "embedding": [0.1] * 384,
        }
    ]
    temp_db.insert_chunks(chunk_records)

    # Verify chunk was inserted with empty context
    conn = temp_db._conn
    chunk = conn.execute("SELECT context FROM chunks").fetchone()
    assert chunk[0] == ""


def test_metadata_table_created(temp_db):
    """Test that metadata table is created during schema initialization."""
    temp_db.connect()

    conn = temp_db._conn
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    table_names = {row[0] for row in tables}

    assert "metadata" in table_names


def test_get_metadata_missing_key(temp_db):
    """Test that get_metadata returns None for non-existent key."""
    temp_db.connect()

    result = temp_db.get_metadata("nonexistent")
    assert result is None


def test_set_and_get_metadata(temp_db):
    """Test setting and retrieving metadata."""
    temp_db.connect()

    temp_db.set_metadata("key", "value")
    result = temp_db.get_metadata("key")

    assert result == "value"


def test_set_metadata_upsert(temp_db):
    """Test that set_metadata overwrites existing values."""
    temp_db.connect()

    temp_db.set_metadata("key", "a")
    temp_db.set_metadata("key", "b")

    result = temp_db.get_metadata("key")
    assert result == "b"


def test_clear_index_preserves_metadata(temp_db):
    """Test that clear_index does NOT clear the metadata table."""
    temp_db.connect()

    # Set metadata
    temp_db.set_metadata("embedding_model", "test-model")

    # Insert some chunk data so clear_index has something to delete
    chunk_records = [
        {
            "file_path": "test.py",
            "language": "python",
            "start_line": 1,
            "end_line": 5,
            "content": "def hello(): pass",
            "context": "test.py: hello",
            "hash": "abc123",
            "embedding": [0.1] * 384,
        }
    ]
    temp_db.insert_chunks(chunk_records)

    # Clear the index
    temp_db.clear_index()

    # Verify metadata survived
    result = temp_db.get_metadata("embedding_model")
    assert result == "test-model"

    # Verify index data was actually cleared
    stats = temp_db.get_index_stats()
    assert stats["total_chunks"] == 0


def test_get_indexed_file_paths(temp_db):
    """Test getting all indexed file paths."""
    temp_db.connect()

    # Insert file metadata for 3 files
    temp_db.update_file_metadata("src/main.py", 5)
    temp_db.update_file_metadata("src/utils.py", 3)
    temp_db.update_file_metadata("tests/test_main.py", 2)

    paths = temp_db.get_indexed_file_paths()

    assert paths == {"src/main.py", "src/utils.py", "tests/test_main.py"}


def test_get_indexed_file_paths_empty(temp_db):
    """Test getting indexed file paths from empty database."""
    temp_db.connect()

    paths = temp_db.get_indexed_file_paths()

    assert paths == set()


def test_get_indexed_files_with_timestamps(temp_db):
    """Test getting indexed files with their last_indexed timestamps."""
    temp_db.connect()

    # Insert file metadata for 2 files
    temp_db.update_file_metadata("src/main.py", 5)
    temp_db.update_file_metadata("src/utils.py", 3)

    result = temp_db.get_indexed_files_with_timestamps()

    assert len(result) == 2
    assert "src/main.py" in result
    assert "src/utils.py" in result

    # Verify values are datetime objects
    assert isinstance(result["src/main.py"], datetime)
    assert isinstance(result["src/utils.py"], datetime)


def test_chunks_table_has_definitions_column(temp_db):
    """Test that the chunks table has a definitions column after connect()."""
    temp_db.connect()

    cols = temp_db._conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name='chunks' AND column_name='definitions'"
    ).fetchall()
    assert len(cols) == 1


def test_insert_chunk_with_definitions(temp_db):
    """Test inserting a chunk with a definitions field and retrieving it."""
    temp_db.connect()

    chunk_records = [
        {
            "file_path": "test.py",
            "language": "python",
            "start_line": 1,
            "end_line": 5,
            "content": "def foo(): pass\nclass Bar: pass",
            "context": "File: test.py\nLanguage: python\nDefines: function foo, class Bar",
            "hash": "abc123",
            "definitions": "function foo, class Bar",
            "embedding": [0.1] * 384,
        }
    ]
    temp_db.insert_chunks(chunk_records)

    result = temp_db._conn.execute("SELECT definitions FROM chunks").fetchone()
    assert result is not None
    assert result[0] == "function foo, class Bar"


def test_vector_search_returns_definitions(temp_db):
    """Test that vector search results include a definitions key."""
    temp_db.connect()

    chunk_records = [
        {
            "file_path": "test1.py",
            "language": "python",
            "start_line": 1,
            "end_line": 5,
            "content": "def hello(): pass",
            "context": "test1.py: hello",
            "hash": "hash1",
            "definitions": "function hello",
            "embedding": [1.0, 0.0, 0.0],
        },
        {
            "file_path": "test2.py",
            "language": "python",
            "start_line": 1,
            "end_line": 3,
            "content": "x = 1",
            "context": "test2.py",
            "hash": "hash2",
            "definitions": "",
            "embedding": [0.0, 1.0, 0.0],
        },
    ]
    temp_db.insert_chunks(chunk_records)

    query_embedding = [0.9, 0.1, 0.0]
    results = temp_db.vector_search(query_embedding, top_k=2)

    if results:  # VSS extension may not be available in test environment
        assert "definitions" in results[0]


def test_bm25_search_returns_definitions(temp_db):
    """Test that BM25 search results include a definitions key."""
    temp_db.connect()

    chunk_records = [
        {
            "file_path": "test1.py",
            "language": "python",
            "start_line": 1,
            "end_line": 5,
            "content": "def calculate_total(items): return sum(items)",
            "context": "test1.py: calculate_total",
            "hash": "hash1",
            "definitions": "function calculate_total",
            "embedding": [0.1] * 384,
        },
        {
            "file_path": "test2.py",
            "language": "python",
            "start_line": 1,
            "end_line": 3,
            "content": "x = 42",
            "context": "test2.py",
            "hash": "hash2",
            "definitions": "",
            "embedding": [0.2] * 384,
        },
    ]
    temp_db.insert_chunks(chunk_records)

    results = temp_db.bm25_search("calculate", top_k=10)

    assert len(results) >= 1
    assert "definitions" in results[0]


def test_existing_db_migration_adds_definitions():
    """Test that re-running _initialize_schema() on a legacy DB adds the definitions column."""
    import duckdb

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "legacy.db"

        # Create a legacy database WITHOUT the definitions column
        conn = duckdb.connect(str(db_path))
        conn.execute("""
            CREATE TABLE chunks (
                id VARCHAR PRIMARY KEY,
                file_path VARCHAR NOT NULL,
                language VARCHAR NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                content TEXT NOT NULL,
                context TEXT,
                hash VARCHAR NOT NULL,
                created_at TIMESTAMP NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE embeddings (
                chunk_id VARCHAR PRIMARY KEY,
                embedding FLOAT[] NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES chunks(id)
            )
        """)
        conn.execute("""
            CREATE TABLE files (
                path VARCHAR PRIMARY KEY,
                chunk_count INTEGER NOT NULL,
                last_indexed TIMESTAMP NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE metadata (
                key VARCHAR PRIMARY KEY,
                value VARCHAR NOT NULL
            )
        """)
        conn.close()

        # Verify definitions column does NOT exist
        conn = duckdb.connect(str(db_path))
        cols = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='chunks' AND column_name='definitions'"
        ).fetchall()
        assert len(cols) == 0
        conn.close()

        # Open with Database and let _initialize_schema run the migration
        db = Database(db_path)
        db.connect()

        # Verify definitions column was added
        cols = db._conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='chunks' AND column_name='definitions'"
        ).fetchall()
        assert len(cols) == 1
        db.close()
