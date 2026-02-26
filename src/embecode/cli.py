"""Entry point for `uvx embecode` / `embecode` CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from embecode.cache import CacheManager


def main() -> None:
    """Parse CLI arguments and start the MCP server or run cache commands."""
    parser = argparse.ArgumentParser(
        prog="embecode",
        description="Local-first MCP server for semantic + keyword hybrid code search.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Main server command (default behavior when no subcommand)
    server_parser = subparsers.add_parser(
        "server",
        help="Start the MCP server (default command)",
        add_help=False,
    )
    server_parser.add_argument(
        "--path",
        default=".",
        metavar="PATH",
        help="Root directory to index (default: current working directory).",
    )

    # Cache management subcommands
    cache_parser = subparsers.add_parser("cache", help="Cache management commands")
    cache_subparsers = cache_parser.add_subparsers(dest="cache_command", help="Cache operations")

    # cache status
    cache_subparsers.add_parser(
        "status", help="Show all cached projects with sizes and last accessed times"
    )

    # cache clean
    cache_subparsers.add_parser("clean", help="Evict LRU entries until cache is under size limit")

    # cache purge [path]
    purge_parser = cache_subparsers.add_parser("purge", help="Delete cache(s)")
    purge_parser.add_argument(
        "path",
        nargs="?",
        metavar="PATH",
        help="Project path to purge (use '.' for current directory). If omitted, purges all caches.",
    )

    args = parser.parse_args()

    # Handle cache commands
    if args.command == "cache":
        if args.cache_command is None:
            cache_parser.print_help()
            sys.exit(0)
        _handle_cache_command(args)
        return

    # Default behavior: start server
    # If no command specified, treat as server command with --path from remaining args
    if args.command is None:
        # Re-parse with server-specific args
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
        path = args.path
    else:
        path = args.path

    # Start the MCP server
    from embecode.server import run_server

    run_server(Path(path).resolve())


def _handle_cache_command(args: argparse.Namespace) -> None:
    """Handle cache subcommands.

    Args:
        args: Parsed command-line arguments
    """
    from embecode.cache import CacheManager

    cache = CacheManager()

    if args.cache_command == "status":
        _cache_status(cache)
    elif args.cache_command == "clean":
        _cache_clean(cache)
    elif args.cache_command == "purge":
        _cache_purge(cache, args.path)


def _cache_status(cache: CacheManager) -> None:
    """Show cache status.

    Args:
        cache: CacheManager instance
    """
    status = cache.get_cache_status()

    print("Cache Status:")
    print(f"  Total size: {status['total_size_human']} / {status['size_limit_human']}")
    print(f"  Projects cached: {status['project_count']}")
    print()

    if not status["projects"]:
        print("No cached projects.")
        return

    print("Cached projects (most recent first):")
    for proj in status["projects"]:
        print(f"  • {proj['project_path']}")
        print(f"    Size: {proj['size_human']}")
        print(f"    Last accessed: {proj['last_accessed']}")
        print(f"    Hash: {proj['hash']}")
        print()


def _cache_clean(cache: CacheManager) -> None:
    """Evict LRU entries until cache is under size limit.

    Args:
        cache: CacheManager instance
    """
    evicted = cache.evict_lru()

    if not evicted:
        print("Cache is already under size limit. No eviction needed.")
        return

    print(f"Evicted {len(evicted)} project(s):")
    for path in evicted:
        print(f"  • {path}")


def _cache_purge(cache: CacheManager, path: str | None) -> None:
    """Purge cache(s).

    Args:
        cache: CacheManager instance
        path: Project path to purge (use '.' for current directory), or None to purge all
    """
    if path is None:
        # Purge all caches
        count = cache.purge_all()
        print(f"Purged all caches ({count} project(s)).")
    else:
        # Purge specific project
        project_path = Path(path).resolve()
        existed = cache.purge_project(project_path)

        if existed:
            print(f"Purged cache for: {project_path}")
        else:
            print(f"No cache found for: {project_path}")
            sys.exit(1)


def _get_version() -> str:
    from embecode import __version__

    return __version__


if __name__ == "__main__":
    main()
