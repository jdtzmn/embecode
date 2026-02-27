"""Cache management for embecode.

Handles cache directory resolution, registry.json management, and LRU eviction.

Cache structure:
    ~/.cache/embecode/
        registry.json          # metadata for all cached projects
        a3f9b2c1/              # hash of /Users/john/projects/myapp
            index.db           # DuckDB file (chunks, embeddings, FTS index)
            daemon.lock        # PID lock file (reserved for v2 daemon)
"""

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class CacheEntry:
    """Metadata for a cached project."""

    project_path: str
    last_accessed: str  # ISO 8601 timestamp
    size_bytes: int


class CacheManager:
    """Manages cache directory resolution, registry, and eviction."""

    def __init__(
        self,
        cache_root: Path | None = None,
        size_limit_bytes: int = 2 * 1024 * 1024 * 1024,  # 2GB default
    ):
        """Initialize cache manager.

        Args:
            cache_root: Root cache directory. Defaults to ~/.cache/embecode
            size_limit_bytes: Maximum total cache size in bytes. Default 2GB.
        """
        self.cache_root = cache_root or Path.home() / ".cache" / "embecode"
        self.size_limit_bytes = size_limit_bytes
        self.registry_path = self.cache_root / "registry.json"

        # Ensure cache root exists
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def get_cache_dir(self, project_path: str | Path) -> Path:
        """Get cache directory for a project.

        Args:
            project_path: Absolute path to the project

        Returns:
            Path to the cache directory for this project
        """
        project_path = Path(project_path).resolve()
        cache_hash = self._hash_path(str(project_path))
        cache_dir = self.cache_root / cache_hash

        # Ensure cache dir exists
        cache_dir.mkdir(parents=True, exist_ok=True)

        return cache_dir

    def update_access_time(self, project_path: str | Path) -> None:
        """Update last_accessed timestamp for a project.

        Args:
            project_path: Absolute path to the project
        """
        project_path = Path(project_path).resolve()
        cache_hash = self._hash_path(str(project_path))
        cache_dir = self.cache_root / cache_hash

        if not cache_dir.exists():
            return

        registry = self._load_registry()
        size_bytes = self._calculate_dir_size(cache_dir)

        registry[cache_hash] = CacheEntry(
            project_path=str(project_path),
            last_accessed=datetime.now().isoformat(),
            size_bytes=size_bytes,
        )

        self._save_registry(registry)

    def get_cache_status(self) -> dict[str, Any]:
        """Get cache status for all projects.

        Returns:
            Dictionary with cache statistics and project details
        """
        registry = self._load_registry()

        # Clean stale entries first
        self._clean_stale_entries(registry)

        total_size = sum(entry.size_bytes for entry in registry.values())
        project_count = len(registry)

        projects = []
        for cache_hash, entry in sorted(
            registry.items(), key=lambda x: x[1].last_accessed, reverse=True
        ):
            projects.append(
                {
                    "hash": cache_hash,
                    "project_path": entry.project_path,
                    "last_accessed": entry.last_accessed,
                    "size_bytes": entry.size_bytes,
                    "size_human": self._human_readable_size(entry.size_bytes),
                }
            )

        return {
            "total_size_bytes": total_size,
            "total_size_human": self._human_readable_size(total_size),
            "size_limit_bytes": self.size_limit_bytes,
            "size_limit_human": self._human_readable_size(self.size_limit_bytes),
            "project_count": project_count,
            "projects": projects,
        }

    def evict_lru(self) -> list[str]:
        """Evict least recently used projects until under size limit.

        Returns:
            List of evicted project paths
        """
        registry = self._load_registry()

        # Clean stale entries first
        self._clean_stale_entries(registry)

        total_size = sum(entry.size_bytes for entry in registry.values())

        if total_size <= self.size_limit_bytes:
            return []

        # Sort by last_accessed (oldest first)
        sorted_entries = sorted(registry.items(), key=lambda x: x[1].last_accessed)

        evicted = []
        for cache_hash, entry in sorted_entries:
            if total_size <= self.size_limit_bytes:
                break

            cache_dir = self.cache_root / cache_hash
            if cache_dir.exists():
                shutil.rmtree(cache_dir)

            evicted.append(entry.project_path)
            total_size -= entry.size_bytes
            del registry[cache_hash]

        self._save_registry(registry)
        return evicted

    def purge_all(self) -> int:
        """Delete all caches.

        Returns:
            Number of projects purged
        """
        registry = self._load_registry()
        count = len(registry)

        # Delete all cache directories
        for cache_hash in registry.keys():
            cache_dir = self.cache_root / cache_hash
            if cache_dir.exists():
                shutil.rmtree(cache_dir)

        # Clear registry
        self._save_registry({})

        return count

    def purge_project(self, project_path: str | Path) -> bool:
        """Delete cache for a specific project.

        Args:
            project_path: Absolute path to the project

        Returns:
            True if cache was deleted, False if it didn't exist
        """
        project_path = Path(project_path).resolve()
        cache_hash = self._hash_path(str(project_path))
        cache_dir = self.cache_root / cache_hash

        existed = cache_dir.exists()

        if existed:
            shutil.rmtree(cache_dir)

        # Remove from registry (even if cache dir didn't exist)
        registry = self._load_registry()
        if cache_hash in registry:
            del registry[cache_hash]
            self._save_registry(registry)
            existed = True  # Consider it existed if it was in registry

        return existed

    def _clean_stale_entries(self, registry: dict[str, CacheEntry]) -> None:
        """Remove entries for projects that no longer exist on disk.

        Args:
            registry: Registry dictionary to clean (modified in place)
        """
        stale_hashes = []

        for cache_hash, entry in registry.items():
            project_path = Path(entry.project_path)
            if not project_path.exists():
                stale_hashes.append(cache_hash)

                # Delete cache directory
                cache_dir = self.cache_root / cache_hash
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)

        # Remove stale entries from registry
        for cache_hash in stale_hashes:
            del registry[cache_hash]

        if stale_hashes:
            self._save_registry(registry)

    def _load_registry(self) -> dict[str, CacheEntry]:
        """Load registry.json.

        Returns:
            Dictionary mapping cache_hash to CacheEntry
        """
        if not self.registry_path.exists():
            return {}

        try:
            with open(self.registry_path) as f:
                data = json.load(f)

            registry = {}
            for cache_hash, entry_data in data.items():
                registry[cache_hash] = CacheEntry(
                    project_path=entry_data["project_path"],
                    last_accessed=entry_data["last_accessed"],
                    size_bytes=entry_data["size_bytes"],
                )

            return registry
        except (json.JSONDecodeError, KeyError, OSError):
            # If registry is corrupted, start fresh
            return {}

    def _save_registry(self, registry: dict[str, CacheEntry]) -> None:
        """Save registry.json.

        Args:
            registry: Dictionary mapping cache_hash to CacheEntry
        """
        data = {}
        for cache_hash, entry in registry.items():
            data[cache_hash] = {
                "project_path": entry.project_path,
                "last_accessed": entry.last_accessed,
                "size_bytes": entry.size_bytes,
            }

        try:
            with open(self.registry_path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError:
            # Fail silently - registry is not critical
            pass

    def get_lock_path(self, project_path: str | Path) -> Path:
        """Get the daemon lock file path for a project.

        Args:
            project_path: Absolute path to the project

        Returns:
            Path to the daemon.lock file for this project's cache directory.
            The lock file lives at ~/.cache/embecode/<hash>/daemon.lock.
            Note: This assumes the cache directory is on local disk.
            O_EXCL is not atomic on NFSv2, so NFS cache directories are not supported.
        """
        cache_dir = self.get_cache_dir(project_path)
        return cache_dir / "daemon.lock"

    @staticmethod
    def _hash_path(path: str) -> str:
        """Generate 8-character hash of a path.

        Args:
            path: Absolute path to hash

        Returns:
            First 8 characters of SHA1 hex digest
        """
        return hashlib.sha1(path.encode()).hexdigest()[:8]

    @staticmethod
    def _calculate_dir_size(path: Path) -> int:
        """Calculate total size of a directory in bytes.

        Args:
            path: Directory path

        Returns:
            Total size in bytes
        """
        total = 0
        try:
            for item in path.rglob("*"):
                if item.is_file():
                    total += item.stat().st_size
        except OSError:
            pass
        return total

    @staticmethod
    def _human_readable_size(size_bytes: int) -> str:
        """Convert bytes to human-readable size.

        Args:
            size_bytes: Size in bytes

        Returns:
            Human-readable string (e.g., "1.5 GB")
        """
        size = float(size_bytes)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"
