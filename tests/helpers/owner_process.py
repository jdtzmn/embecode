"""Minimal owner process helper for concurrent integration tests.

Usage:
    python -m tests.helpers.owner_process <db_path>

Opens a real DuckDB ``Database`` connection in read-write mode, prints
``ready`` to stdout once the connection is established, then blocks on
``sys.stdin.read()`` until stdin is closed by the parent process.

The parent test controls this process's lifetime via
``process.communicate()``.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: owner_process.py <db_path>", file=sys.stderr)
        sys.exit(1)

    db_path = Path(sys.argv[1])

    # Import here so the module can be imported without side effects
    from embecode.db import Database

    db = Database(db_path)
    try:
        db.connect()
        # Signal the parent that the read-write connection is open and held
        print("ready", flush=True)
        # Block until the parent closes our stdin (i.e. the test is done)
        sys.stdin.read()
    finally:
        db.close()


if __name__ == "__main__":
    main()
