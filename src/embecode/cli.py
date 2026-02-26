"""Entry point for `uvx embecode` / `embecode` CLI."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    """Parse CLI arguments and start the MCP server."""
    parser = argparse.ArgumentParser(
        prog="embecode",
        description="Local-first MCP server for semantic + keyword hybrid code search.",
    )
    parser.add_argument(
        "--path",
        default=".",
        metavar="PATH",
        help="Root directory to index (default: current working directory).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )

    args = parser.parse_args()

    # Placeholder — server startup will be wired in once core modules exist.
    print(f"embecode: indexing {args.path!r} — server not yet implemented", file=sys.stderr)
    sys.exit(1)


def _get_version() -> str:
    from embecode import __version__

    return __version__


if __name__ == "__main__":
    main()
