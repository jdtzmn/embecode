"""Subprocess helper that indexes a project directory into a DuckDB database.

Usage:
    python -m tests.helpers.indexer_process <db_path> <project_dir>

Opens the database in read-write mode, indexes all files under *project_dir*
using a deterministic ``FixedVectorEmbedder``, prints ``done`` to stdout on
success, then exits (releasing the DuckDB lock).

Exit codes:
    0 — indexing succeeded (stdout contains "done")
    1 — error (stderr contains the traceback)
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: indexer_process.py <db_path> <project_dir>", file=sys.stderr)
        sys.exit(1)

    db_path = Path(sys.argv[1])
    project_dir = Path(sys.argv[2])

    # Import here so the module can be imported without side effects
    from embecode.config import load_config
    from embecode.db import Database
    from embecode.indexer import Indexer

    # FixedVectorEmbedder — must match the one in test_concurrent.py
    _DIM = 768
    random.seed(42)
    _raw = [random.gauss(0, 1) for _ in range(_DIM)]
    _norm = math.sqrt(sum(x * x for x in _raw))
    _fixed_vector = [x / _norm for x in _raw]

    class FixedVectorEmbedder:
        dimension = _DIM

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [_fixed_vector[:] for _ in texts]

        def unload(self) -> None:
            pass

    db = Database(db_path)
    try:
        db.connect(read_only=False)
        config = load_config(project_dir)
        embedder = FixedVectorEmbedder()
        indexer = Indexer(project_dir, config, db, embedder)  # type: ignore[arg-type]
        indexer.start_full_index(background=False)
        print("done", flush=True)
    finally:
        db.close()


if __name__ == "__main__":
    main()
