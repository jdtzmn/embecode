"""Database module for embecode.

Provides DuckDB setup, schema management, and integration with VSS (Vector Similarity
Search) and FTS (Full-Text Search) extensions for hybrid code search.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

logger = logging.getLogger(__name__)


class Database:
    """Database interface for embecode index storage.

    Uses DuckDB with VSS extension for vector similarity search and FTS extension
    for BM25 keyword search. All data is stored in a single database file.

    Thread safety: all public methods acquire ``_db_lock`` before touching the
    underlying DuckDB connection so that multiple threads (catch-up indexer,
    file watcher, MCP request handler) can safely share a single instance.
    """

    def __init__(self, db_path: Path) -> None:
        """Initialize database connection.

        Args:
            db_path: Path to the DuckDB database file. Will be created if it doesn't exist.
        """
        self.db_path = db_path
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._is_initialized = False
        self._read_only = False
        self._db_lock = threading.Lock()

    @property
    def is_read_only(self) -> bool:
        """Return True if the current connection was opened in read-only mode."""
        return self._read_only

    def connect(self, read_only: bool = False) -> None:
        """Open database connection and initialize schema if needed.

        Args:
            read_only: If True, open the connection in read-only mode.  A
                read-only connection cannot write to the database but can
                coexist with a separate read-write connection from another
                process (DuckDB supports this).
        """
        with self._db_lock:
            if self._conn is not None:
                return

            # Create parent directory if it doesn't exist
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Open connection
            self._conn = duckdb.connect(str(self.db_path), read_only=read_only)
            self._read_only = read_only

            if not read_only:
                # Cap DuckDB's buffer pool so it doesn't consume all available RAM.
                # Without this, DuckDB defaults to ~80% of system RAM and never shrinks
                # its buffer pool voluntarily, causing the post-indexing memory floor.
                try:
                    self._conn.execute("SET memory_limit='2GB'")
                except Exception as e:
                    logger.warning(f"Failed to set DuckDB memory limit: {e}")

                # Install and load VSS extension for vector similarity search
                try:
                    self._conn.execute("INSTALL vss")
                    self._conn.execute("LOAD vss")
                except Exception as e:
                    logger.warning(f"Failed to load VSS extension: {e}")

                # Install and load FTS extension for full-text search
                try:
                    self._conn.execute("INSTALL fts")
                    self._conn.execute("LOAD fts")
                except Exception as e:
                    logger.warning(f"Failed to load FTS extension: {e}")
            else:
                # Extensions are already installed; just load them
                try:
                    self._conn.execute("LOAD vss")
                except Exception as e:
                    logger.warning(f"Failed to load VSS extension (read-only): {e}")
                try:
                    self._conn.execute("LOAD fts")
                except Exception as e:
                    logger.warning(f"Failed to load FTS extension (read-only): {e}")

            # Initialize schema if needed (skip in read-only mode — the owner
            # is responsible for schema creation)
            if not read_only and not self._is_initialized:
                self._initialize_schema()
                self._is_initialized = True

    def reconnect(self, read_only: bool = False) -> None:
        """Close the current connection and reopen with the given mode.

        This is used during reader → owner promotion: the reader must close
        its read-only connection before reopening read-write.

        Args:
            read_only: Mode for the new connection.
        """
        with self._db_lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
                self._is_initialized = False
                self._read_only = False

        # connect() acquires the lock itself, so call it outside the lock
        self.connect(read_only=read_only)

    def close(self) -> None:
        """Close database connection."""
        with self._db_lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
                self._is_initialized = False
                self._read_only = False

    def shrink_memory(self) -> None:
        """Release DuckDB's buffer pool pages back to the OS after a bulk write.

        DuckDB retains its buffer pool indefinitely by default, which causes a
        large memory footprint to persist after indexing completes. Calling
        ``checkpoint()`` flushes all dirty buffer-pool pages to the ``.db``
        file so the OS can reclaim the backing physical RAM.  The
        ``memory_limit`` set in :meth:`connect` then prevents the pool from
        refilling beyond 2 GB on subsequent reads.
        """
        if self._conn is None:
            return
        try:
            self._conn.checkpoint()
        except Exception as e:
            logger.warning(f"Failed to checkpoint DuckDB: {e}")

    def _initialize_schema(self) -> None:
        """Create tables and indexes if they don't exist.

        Must be called with ``_db_lock`` already held (called from ``connect``).
        """
        if self._conn is None:
            msg = "Database connection not open"
            raise RuntimeError(msg)

        # Chunks table: stores code chunks with metadata
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id VARCHAR PRIMARY KEY,
                file_path VARCHAR NOT NULL,
                language VARCHAR NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                content TEXT NOT NULL,
                context TEXT,
                hash VARCHAR NOT NULL,
                created_at TIMESTAMP NOT NULL,
                definitions TEXT NOT NULL DEFAULT ''
            )
        """)

        # Embeddings table: stores vector embeddings for chunks
        # Using FLOAT[] array for embeddings (VSS extension supports this)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id VARCHAR PRIMARY KEY,
                embedding FLOAT[] NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES chunks(id)
            )
        """)

        # Files table: tracks indexed files with metadata
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS files (
                path VARCHAR PRIMARY KEY,
                chunk_count INTEGER NOT NULL,
                last_indexed TIMESTAMP NOT NULL
            )
        """)

        # Metadata table: stores key-value pairs (e.g., embedding model name)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key VARCHAR PRIMARY KEY,
                value VARCHAR NOT NULL
            )
        """)

        # Create indexes for common queries
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_file_path
            ON chunks(file_path)
        """)

        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_hash
            ON chunks(hash)
        """)

        # Migration: add definitions column if it doesn't exist (for existing DBs)
        try:
            self._conn.execute("""
                ALTER TABLE chunks ADD COLUMN definitions TEXT DEFAULT ''
            """)
        except Exception:
            # Column already exists — skip
            pass

        # Create FTS index for keyword search
        # Note: We use the simple PRAGMA approach which creates the index dynamically
        try:
            self._conn.execute("""
                PRAGMA create_fts_index('chunks', 'id', 'content', overwrite=1)
            """)
        except Exception as e:
            logger.warning(f"Failed to create FTS index: {e}")

    def clear_index(self) -> None:
        """Clear the entire index (delete all chunks, embeddings, and files)."""
        self.connect()
        with self._db_lock:
            if self._conn is None:
                msg = "Database connection not open"
                raise RuntimeError(msg)

            self._conn.execute("DELETE FROM embeddings")
            self._conn.execute("DELETE FROM chunks")
            self._conn.execute("DELETE FROM files")

    def get_index_stats(self) -> dict[str, Any]:
        """Get index statistics.

        Returns:
            Dictionary with keys:
            - total_chunks: Total number of chunks in the index
            - files_indexed: Total number of files indexed
            - last_updated: ISO timestamp of most recent update (or None if empty)
        """
        self.connect()
        with self._db_lock:
            if self._conn is None:
                msg = "Database connection not open"
                raise RuntimeError(msg)

            total_chunks = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            files_indexed = self._conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]

            # Get most recent update timestamp
            last_updated = None
            if total_chunks > 0:
                result = self._conn.execute("SELECT MAX(created_at) FROM chunks").fetchone()
                if result and result[0]:
                    last_updated = result[0].isoformat()

        return {
            "total_chunks": total_chunks,
            "files_indexed": files_indexed,
            "last_updated": last_updated,
        }

    def get_chunk_hashes_for_file(self, file_path: str) -> set[str]:
        """Get all chunk hashes for a specific file.

        Args:
            file_path: Path to the file

        Returns:
            Set of chunk hashes (SHA1) for the file
        """
        self.connect()
        with self._db_lock:
            if self._conn is None:
                msg = "Database connection not open"
                raise RuntimeError(msg)

            result = self._conn.execute(
                "SELECT hash FROM chunks WHERE file_path = ?",
                [file_path],
            ).fetchall()

        return {row[0] for row in result}

    def insert_chunks(self, chunk_records: list[dict[str, Any]]) -> None:
        """Insert chunks and their embeddings into the database.

        Args:
            chunk_records: List of chunk records, each containing:
                - file_path: Path to the source file
                - language: Programming language
                - start_line: Starting line number
                - end_line: Ending line number
                - content: Chunk content
                - context: Contextual metadata for embedding
                - hash: SHA1 hash of the chunk content
                - definitions: Comma-separated definition names
                - embedding: Vector embedding (list of floats)
        """
        self.connect()

        if not chunk_records:
            return

        # Generate chunk IDs and prepare records (outside lock — pure Python work)
        now = datetime.now(UTC)
        chunk_rows = []
        embedding_rows = []

        for record in chunk_records:
            chunk_id = f"{record['file_path']}:{record['start_line']}"
            chunk_rows.append(
                (
                    chunk_id,
                    record["file_path"],
                    record["language"],
                    record["start_line"],
                    record["end_line"],
                    record["content"],
                    record.get("context", ""),
                    record["hash"],
                    record.get("definitions", ""),
                    now,
                )
            )
            embedding_rows.append((chunk_id, record["embedding"]))

        with self._db_lock:
            if self._conn is None:
                msg = "Database connection not open"
                raise RuntimeError(msg)

            # Check which chunks already exist (for upsert handling)
            existing_ids = {
                row[0]
                for row in self._conn.execute(
                    f"SELECT id FROM chunks WHERE id IN ({','.join('?' * len(chunk_rows))})",
                    [row[0] for row in chunk_rows],
                ).fetchall()
            }

            # Delete embeddings for existing chunks first (foreign key constraint)
            if existing_ids:
                placeholders = ",".join("?" * len(existing_ids))
                self._conn.execute(
                    f"DELETE FROM embeddings WHERE chunk_id IN ({placeholders})",
                    list(existing_ids),
                )

            # Insert/update chunks
            self._conn.executemany(
                """
                INSERT INTO chunks (id, file_path, language, start_line, end_line,
                                  content, context, hash, definitions, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    content = excluded.content,
                    hash = excluded.hash,
                    definitions = excluded.definitions,
                    created_at = excluded.created_at
                """,
                chunk_rows,
            )

            # Insert embeddings (all new after deletion above)
            self._conn.executemany(
                """
                INSERT INTO embeddings (chunk_id, embedding)
                VALUES (?, ?)
                """,
                embedding_rows,
            )

    def delete_chunks_by_hash(self, hashes: list[str]) -> None:
        """Delete chunks by their hashes.

        Args:
            hashes: List of SHA1 hashes to delete
        """
        self.connect()

        if not hashes:
            return

        with self._db_lock:
            if self._conn is None:
                msg = "Database connection not open"
                raise RuntimeError(msg)

            # Get chunk IDs to delete
            placeholders = ",".join("?" * len(hashes))
            chunk_ids = self._conn.execute(
                f"SELECT id FROM chunks WHERE hash IN ({placeholders})",
                hashes,
            ).fetchall()

            if not chunk_ids:
                return

            chunk_id_list = [row[0] for row in chunk_ids]

            # Delete embeddings first (foreign key constraint)
            placeholders = ",".join("?" * len(chunk_id_list))
            self._conn.execute(
                f"DELETE FROM embeddings WHERE chunk_id IN ({placeholders})",
                chunk_id_list,
            )

            # Delete chunks
            placeholders = ",".join("?" * len(hashes))
            self._conn.execute(
                f"DELETE FROM chunks WHERE hash IN ({placeholders})",
                hashes,
            )

    def delete_chunks_by_file(self, file_path: str) -> None:
        """Delete all chunks for a specific file.

        Args:
            file_path: Path to the file
        """
        self.connect()
        with self._db_lock:
            if self._conn is None:
                msg = "Database connection not open"
                raise RuntimeError(msg)

            # Get chunk IDs to delete
            chunk_ids = self._conn.execute(
                "SELECT id FROM chunks WHERE file_path = ?",
                [file_path],
            ).fetchall()

            if not chunk_ids:
                return

            chunk_id_list = [row[0] for row in chunk_ids]

            # Delete embeddings first (foreign key constraint)
            placeholders = ",".join("?" * len(chunk_id_list))
            self._conn.execute(
                f"DELETE FROM embeddings WHERE chunk_id IN ({placeholders})",
                chunk_id_list,
            )

            # Delete chunks
            self._conn.execute(
                "DELETE FROM chunks WHERE file_path = ?",
                [file_path],
            )

    def update_file_metadata(self, file_path: str, chunk_count: int) -> None:
        """Update file metadata after indexing.

        Args:
            file_path: Path to the file
            chunk_count: Number of chunks for this file
        """
        self.connect()
        with self._db_lock:
            if self._conn is None:
                msg = "Database connection not open"
                raise RuntimeError(msg)

            now = datetime.now(UTC)
            self._conn.execute(
                """
                INSERT INTO files (path, chunk_count, last_indexed)
                VALUES (?, ?, ?)
                ON CONFLICT (path) DO UPDATE SET
                    chunk_count = excluded.chunk_count,
                    last_indexed = excluded.last_indexed
                """,
                [file_path, chunk_count, now],
            )

    def delete_file(self, file_path: str) -> int:
        """Delete all chunks for a file and return count deleted.

        Args:
            file_path: Path to the file

        Returns:
            Number of chunks deleted
        """
        self.connect()
        with self._db_lock:
            if self._conn is None:
                msg = "Database connection not open"
                raise RuntimeError(msg)

            # Get count before deletion
            count_result = self._conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE file_path = ?",
                [file_path],
            ).fetchone()
            count_before = count_result[0] if count_result else 0

        # delete_chunks_by_file acquires the lock itself — call outside lock
        self.delete_chunks_by_file(file_path)

        with self._db_lock:
            if self._conn is None:
                msg = "Database connection not open"
                raise RuntimeError(msg)

            # Delete file metadata
            self._conn.execute(
                "DELETE FROM files WHERE path = ?",
                [file_path],
            )

        return count_before

    def vector_search(
        self,
        query_embedding: list[float],
        top_k: int,
        path_prefix: str | None = None,
    ) -> list[dict[str, Any]]:
        """Perform vector similarity search.

        Args:
            query_embedding: Query vector embedding
            top_k: Maximum number of results to return
            path_prefix: Optional path prefix filter (e.g., "src/")

        Returns:
            List of search results, each containing:
            - content: Chunk content
            - file_path: Path to source file
            - language: Programming language
            - start_line: Starting line number
            - end_line: Ending line number
            - context: Contextual metadata
            - definitions: Comma-separated definition names
            - score: Similarity score (higher is better)
        """
        self.connect()

        # Build query with optional path filter (pure Python — outside lock)
        sql = """
            SELECT
                c.content,
                c.file_path,
                c.language,
                c.start_line,
                c.end_line,
                c.context,
                c.definitions,
                array_cosine_similarity(e.embedding, ?::FLOAT[]) as score
            FROM chunks c
            JOIN embeddings e ON c.id = e.chunk_id
        """

        params: list[Any] = [query_embedding]

        if path_prefix:
            sql += " WHERE c.file_path LIKE ?"
            params.append(f"{path_prefix}%")

        sql += " ORDER BY score DESC LIMIT ?"
        params.append(top_k)

        with self._db_lock:
            if self._conn is None:
                msg = "Database connection not open"
                raise RuntimeError(msg)

            try:
                results = self._conn.execute(sql, params).fetchall()
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
                # Fall back to no results if VSS extension is not available
                return []

        return [
            {
                "content": row[0],
                "file_path": row[1],
                "language": row[2],
                "start_line": row[3],
                "end_line": row[4],
                "context": row[5],
                "definitions": row[6],
                "score": float(row[7]) if row[7] is not None else 0.0,
            }
            for row in results
        ]

    def bm25_search(
        self,
        query: str,
        top_k: int,
        path_prefix: str | None = None,
    ) -> list[dict[str, Any]]:
        """Perform BM25 keyword search.

        Args:
            query: Search query string
            top_k: Maximum number of results to return
            path_prefix: Optional path prefix filter (e.g., "src/")

        Returns:
            List of search results, each containing:
            - content: Chunk content
            - file_path: Path to source file
            - language: Programming language
            - start_line: Starting line number
            - end_line: Ending line number
            - context: Contextual metadata
            - definitions: Comma-separated definition names
            - score: BM25 relevance score (higher is better)
        """
        self.connect()

        # Build FTS query with optional path filter (pure Python — outside lock)
        if path_prefix:
            fts_query = """
                SELECT
                    c.content,
                    c.file_path,
                    c.language,
                    c.start_line,
                    c.end_line,
                    c.context,
                    c.definitions,
                    fts.score
                FROM (
                    SELECT fts_main_chunks.match_bm25(id, ?) AS score, id
                    FROM chunks
                ) fts
                JOIN chunks c ON c.id = fts.id
                WHERE fts.score IS NOT NULL AND c.file_path LIKE ?
                ORDER BY fts.score DESC
                LIMIT ?
            """
            fts_params: list[Any] = [query, f"{path_prefix}%", top_k]
        else:
            fts_query = """
                SELECT
                    c.content,
                    c.file_path,
                    c.language,
                    c.start_line,
                    c.end_line,
                    c.context,
                    c.definitions,
                    fts.score
                FROM (
                    SELECT fts_main_chunks.match_bm25(id, ?) AS score, id
                    FROM chunks
                ) fts
                JOIN chunks c ON c.id = fts.id
                WHERE fts.score IS NOT NULL
                ORDER BY fts.score DESC
                LIMIT ?
            """
            fts_params = [query, top_k]

        with self._db_lock:
            if self._conn is None:
                msg = "Database connection not open"
                raise RuntimeError(msg)

            try:
                results = self._conn.execute(fts_query, fts_params).fetchall()

                # If FTS returns no results, fall back to substring search
                if not results:
                    logger.debug("FTS returned no results, falling back to substring search")
                    # _fallback_keyword_search acquires the lock itself — call
                    # outside the lock
                    results = None
            except Exception as e:
                logger.warning(f"BM25 search failed: {e}")
                results = None

        if results is None:
            # Fall back to simple substring search if FTS is not available or
            # returned no results
            return self._fallback_keyword_search(query, top_k, path_prefix)

        return [
            {
                "content": row[0],
                "file_path": row[1],
                "language": row[2],
                "start_line": row[3],
                "end_line": row[4],
                "context": row[5],
                "definitions": row[6],
                "score": float(row[7]) if row[7] is not None else 0.0,
            }
            for row in results
        ]

    def get_metadata(self, key: str) -> str | None:
        """Read a value from the metadata table.

        Args:
            key: Metadata key to look up

        Returns:
            The value if found, None otherwise
        """
        self.connect()
        with self._db_lock:
            if self._conn is None:
                msg = "Database connection not open"
                raise RuntimeError(msg)

            result = self._conn.execute(
                "SELECT value FROM metadata WHERE key = ?",
                [key],
            ).fetchone()

        return result[0] if result else None

    def set_metadata(self, key: str, value: str) -> None:
        """Write (upsert) a value into the metadata table.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.connect()
        with self._db_lock:
            if self._conn is None:
                msg = "Database connection not open"
                raise RuntimeError(msg)

            self._conn.execute(
                """
                INSERT INTO metadata (key, value)
                VALUES (?, ?)
                ON CONFLICT (key) DO UPDATE SET
                    value = excluded.value
                """,
                [key, value],
            )

    def get_indexed_file_paths(self) -> set[str]:
        """Get all indexed file paths.

        Returns:
            Set of all file paths in the files table
        """
        self.connect()
        with self._db_lock:
            if self._conn is None:
                msg = "Database connection not open"
                raise RuntimeError(msg)

            result = self._conn.execute("SELECT path FROM files").fetchall()

        return {row[0] for row in result}

    def get_indexed_files_with_timestamps(self) -> dict[str, datetime]:
        """Get all indexed files with their last_indexed timestamps.

        Returns:
            Dictionary mapping file paths to their last_indexed datetime (UTC)
        """
        self.connect()
        with self._db_lock:
            if self._conn is None:
                msg = "Database connection not open"
                raise RuntimeError(msg)

            result = self._conn.execute("SELECT path, last_indexed FROM files").fetchall()

        return {row[0]: row[1] for row in result}

    def _fallback_keyword_search(
        self,
        query: str,
        top_k: int,
        path_prefix: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fallback keyword search using simple substring matching.

        Args:
            query: Search query string
            top_k: Maximum number of results to return
            path_prefix: Optional path prefix filter

        Returns:
            List of search results (same format as bm25_search)
        """
        query_sql = """
            SELECT content, file_path, language, start_line, end_line, context, definitions
            FROM chunks
            WHERE LOWER(content) LIKE ?
        """

        params: list[Any] = [f"%{query.lower()}%"]

        if path_prefix:
            query_sql += " AND file_path LIKE ?"
            params.append(f"{path_prefix}%")

        query_sql += " LIMIT ?"
        params.append(top_k)

        with self._db_lock:
            if self._conn is None:
                msg = "Database connection not open"
                raise RuntimeError(msg)

            results = self._conn.execute(query_sql, params).fetchall()

        return [
            {
                "content": row[0],
                "file_path": row[1],
                "language": row[2],
                "start_line": row[3],
                "end_line": row[4],
                "context": row[5],
                "definitions": row[6],
                "score": 1.0,  # Uniform score for fallback
            }
            for row in results
        ]
