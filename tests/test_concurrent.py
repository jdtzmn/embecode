"""Integration tests for concurrent DuckDB access (owner + reader).

Verifies that two embecode processes pointed at the same DuckDB index can run
simultaneously — one as OWNER (read-write) and one as READER (read-only) — and
that the READER can serve search queries while the OWNER holds its connection.

Run with:
    pytest tests/test_concurrent.py -v --no-cov
"""

from __future__ import annotations

import math
import os
import random
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from embecode.config import EmbeCodeConfig, load_config
from embecode.db import Database
from embecode.indexer import Indexer

# ---------------------------------------------------------------------------
# FixedVectorEmbedder — deterministic embedder for testing
# ---------------------------------------------------------------------------

_DIM = 768
random.seed(42)
_raw = [random.gauss(0, 1) for _ in range(_DIM)]
_norm = math.sqrt(sum(x * x for x in _raw))
_FIXED_UNIT_VECTOR: list[float] = [x / _norm for x in _raw]


class FixedVectorEmbedder:
    """Embedder that always returns the same pre-computed unit vector."""

    dimension = _DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_FIXED_UNIT_VECTOR[:] for _ in texts]

    def unload(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_OWNER_PROCESS_MODULE = "tests.helpers.owner_process"
_OWNER_READY_TIMEOUT = 30  # seconds


@pytest.fixture()
def indexed_db(tmp_path: Path) -> Path:
    """Build a real DuckDB index in *tmp_path* and return the DB file path.

    Creates a small project with a Python source file, indexes it using a
    ``FixedVectorEmbedder``, then closes the database so the owner subprocess
    can open it cleanly.
    """
    # -- tiny project -------------------------------------------------------
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    src_file = project_dir / "example.py"
    src_file.write_text(
        textwrap.dedent("""\
            def greet(name: str) -> str:
                \"\"\"Return a friendly greeting.\"\"\"
                return f"Hello, {name}!"

            def add(a: int, b: int) -> int:
                \"\"\"Add two numbers and return the sum.\"\"\"
                return a + b

            class Calculator:
                \"\"\"A simple calculator class.\"\"\"

                def multiply(self, x: int, y: int) -> int:
                    return x * y
        """),
    )

    # -- index --------------------------------------------------------------
    db_path = tmp_path / "index.db"
    db = Database(db_path)
    db.connect(read_only=False)
    try:
        config = load_config(project_dir)
        embedder = FixedVectorEmbedder()
        indexer = Indexer(project_dir, config, db, embedder)  # type: ignore[arg-type]
        indexer.start_full_index(background=False)

        # Verify something was actually indexed
        stats = db.get_index_stats()
        assert stats["total_chunks"] > 0, "Indexer produced no chunks"
    finally:
        db.close()

    return db_path


def _spawn_owner(db_path: Path) -> subprocess.Popen:
    """Launch the owner helper process and wait for its "ready" signal."""
    proc = subprocess.Popen(
        [sys.executable, "-m", _OWNER_PROCESS_MODULE, str(db_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        # Run from repo root so ``-m tests.helpers.owner_process`` resolves
        cwd=str(Path(__file__).resolve().parent.parent),
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "src")},
    )
    try:
        assert proc.stdout is not None
        # Read the first line — must be "ready"
        line = proc.stdout.readline()
        if not line.strip():
            # Process may have crashed — capture stderr for diagnostics
            stderr = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(
                f"Owner process exited prematurely (rc={proc.poll()}). stderr:\n{stderr}"
            )
        assert line.strip() == "ready", f"Expected 'ready', got: {line!r}"
    except Exception:
        proc.kill()
        proc.wait()
        raise
    return proc


def _kill_owner(proc: subprocess.Popen) -> None:
    """Terminate the owner process cleanly."""
    if proc.stdin:
        proc.stdin.close()
    proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
def test_reader_can_search_while_owner_holds_connection(indexed_db: Path) -> None:
    """A read-only Database can execute searches while another process holds
    the read-write connection on the same DuckDB file."""
    owner = _spawn_owner(indexed_db)
    try:
        # -- open reader (read-only) in *this* process ----------------------
        reader = Database(indexed_db)
        reader.connect(read_only=True)
        try:
            # Vector search
            results = reader.vector_search(_FIXED_UNIT_VECTOR, top_k=5)
            assert isinstance(results, list)
            assert len(results) > 0, "vector_search returned no results"

            # BM25 / keyword search
            results_bm25 = reader.bm25_search("greet", top_k=5)
            assert isinstance(results_bm25, list)
            # bm25 may use fallback substring search — either way, should succeed
            assert len(results_bm25) > 0, "bm25_search returned no results"
        finally:
            reader.close()
    finally:
        _kill_owner(owner)
