"""Configuration loading and resolution for embecode."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


@dataclass
class LanguageConfig:
    """Per-language chunk size configuration (measured in non-whitespace characters)."""

    python: int = 1500
    typescript: int = 1200
    javascript: int = 1200
    default: int = 1000


@dataclass
class IndexConfig:
    """Index behavior configuration."""

    include: list[str] = field(default_factory=lambda: ["src/", "lib/", "tests/"])
    exclude: list[str] = field(
        default_factory=lambda: [
            "node_modules/",
            "dist/",
            "build/",
            ".git/",
            "*.min.js",
            "**/__pycache__/",
            "**/.pytest_cache/",
            "**/venv/",
            "**/.venv/",
        ]
    )
    languages: LanguageConfig = field(default_factory=LanguageConfig)


@dataclass
class EmbeddingsConfig:
    """Embedding model configuration."""

    model: str = "nomic-embed-text-v1.5"
    api_key_env: str | None = None


@dataclass
class SearchConfig:
    """Search behavior configuration."""

    default_mode: str = "hybrid"
    top_k: int = 10


@dataclass
class DaemonConfig:
    """File watching and daemon configuration."""

    debounce_ms: int = 500
    auto_watch: bool = True


@dataclass
class CacheConfig:
    """Cache behavior configuration (user-global only)."""

    max_size_bytes: int = 2_000_000_000  # 2GB default


@dataclass
class EmbeCodeConfig:
    """Complete embecode configuration."""

    index: IndexConfig = field(default_factory=IndexConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)


def load_config(
    project_path: Path | None = None, cli_overrides: dict[str, Any] | None = None
) -> EmbeCodeConfig:
    """
    Load configuration with priority order (highest to lowest):
    1. CLI args (via cli_overrides)
    2. .embecode.toml in project root
    3. ~/.config/embecode/config.toml (user-global)
    4. Built-in defaults

    Args:
        project_path: Path to the project root (for finding .embecode.toml)
        cli_overrides: Dictionary of CLI overrides (e.g., {"index": {"exclude": ["dist/"]}})

    Returns:
        Fully resolved EmbeCodeConfig
    """
    # Start with defaults
    config = EmbeCodeConfig()

    # Layer 3: User-global config
    user_config_path = Path.home() / ".config" / "embecode" / "config.toml"
    if user_config_path.exists():
        _merge_config_from_file(config, user_config_path)

    # Layer 2: Project-level config
    if project_path:
        project_config_path = project_path / ".embecode.toml"
        if project_config_path.exists():
            _merge_config_from_file(config, project_config_path)

    # Layer 1: CLI overrides
    if cli_overrides:
        _merge_config_from_dict(config, cli_overrides)

    return config


def _merge_config_from_file(config: EmbeCodeConfig, path: Path) -> None:
    """Load TOML file and merge into existing config."""
    with path.open("rb") as f:
        data = tomllib.load(f)
    _merge_config_from_dict(config, data)


def _merge_config_from_dict(config: EmbeCodeConfig, data: dict[str, Any]) -> None:
    """Merge dictionary data into config object."""
    if "index" in data:
        index_data = data["index"]
        if "include" in index_data:
            config.index.include = index_data["include"]
        if "exclude" in index_data:
            config.index.exclude = index_data["exclude"]
        if "languages" in index_data:
            lang_data = index_data["languages"]
            if "python" in lang_data:
                config.index.languages.python = lang_data["python"]
            if "typescript" in lang_data:
                config.index.languages.typescript = lang_data["typescript"]
            if "javascript" in lang_data:
                config.index.languages.javascript = lang_data["javascript"]
            if "default" in lang_data:
                config.index.languages.default = lang_data["default"]

    if "embeddings" in data:
        emb_data = data["embeddings"]
        if "model" in emb_data:
            config.embeddings.model = emb_data["model"]
        if "api_key_env" in emb_data:
            config.embeddings.api_key_env = emb_data["api_key_env"]

    if "search" in data:
        search_data = data["search"]
        if "default_mode" in search_data:
            config.search.default_mode = search_data["default_mode"]
        if "top_k" in search_data:
            config.search.top_k = search_data["top_k"]

    if "daemon" in data:
        daemon_data = data["daemon"]
        if "debounce_ms" in daemon_data:
            config.daemon.debounce_ms = daemon_data["debounce_ms"]
        if "auto_watch" in daemon_data:
            config.daemon.auto_watch = daemon_data["auto_watch"]

    if "cache" in data:
        cache_data = data["cache"]
        if "max_size_bytes" in cache_data:
            config.cache.max_size_bytes = cache_data["max_size_bytes"]


def get_chunk_size_for_language(config: EmbeCodeConfig, language: str) -> int:
    """
    Get the chunk size budget for a specific language.

    Args:
        config: The loaded configuration
        language: Language name (e.g., "python", "typescript")

    Returns:
        Chunk size in non-whitespace characters
    """
    lang_lower = language.lower()
    if lang_lower == "python":
        return config.index.languages.python
    elif lang_lower in ("typescript", "tsx"):
        return config.index.languages.typescript
    elif lang_lower in ("javascript", "jsx"):
        return config.index.languages.javascript
    else:
        return config.index.languages.default
