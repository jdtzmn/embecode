"""Tests for config loading and resolution."""

from __future__ import annotations

import tempfile
from pathlib import Path

from embecode.config import (
    EmbeCodeConfig,
    get_chunk_size_for_language,
    load_config,
)


def test_default_config():
    """Test that default config loads correctly."""
    config = load_config()

    assert config.index.include == ["src/", "lib/", "tests/"]
    assert "node_modules/" in config.index.exclude
    assert config.index.languages.python == 1500
    assert config.index.languages.typescript == 1200
    assert config.index.languages.javascript == 1200
    assert config.index.languages.default == 1000

    assert config.embeddings.model == "nomic-embed-text-v1.5"
    assert config.embeddings.api_key_env is None

    assert config.search.default_mode == "hybrid"
    assert config.search.top_k == 10

    assert config.daemon.debounce_ms == 500
    assert config.daemon.auto_watch is True

    assert config.cache.max_size_bytes == 2_000_000_000


def test_project_config_override():
    """Test that project-level .embecode.toml overrides defaults."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        config_file = project_path / ".embecode.toml"

        config_file.write_text("""
[index]
include = ["custom/src/"]
exclude = ["custom/dist/"]

[index.languages]
python = 2000

[embeddings]
model = "BAAI/bge-base-en-v1.5"

[search]
top_k = 20
""")

        config = load_config(project_path=project_path)

        assert config.index.include == ["custom/src/"]
        assert config.index.exclude == ["custom/dist/"]
        assert config.index.languages.python == 2000
        assert config.index.languages.typescript == 1200  # Still default
        assert config.embeddings.model == "BAAI/bge-base-en-v1.5"
        assert config.search.top_k == 20
        assert config.search.default_mode == "hybrid"  # Still default


def test_cli_overrides():
    """Test that CLI overrides take highest priority."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        config_file = project_path / ".embecode.toml"

        config_file.write_text("""
[search]
top_k = 20
""")

        cli_overrides = {
            "search": {"top_k": 5},
            "index": {"include": ["cli/override/"]},
        }

        config = load_config(project_path=project_path, cli_overrides=cli_overrides)

        assert config.search.top_k == 5  # CLI override wins
        assert config.index.include == ["cli/override/"]  # CLI override wins


def test_get_chunk_size_for_language():
    """Test chunk size resolution for different languages."""
    config = EmbeCodeConfig()

    assert get_chunk_size_for_language(config, "python") == 1500
    assert get_chunk_size_for_language(config, "Python") == 1500  # Case insensitive
    assert get_chunk_size_for_language(config, "typescript") == 1200
    assert get_chunk_size_for_language(config, "tsx") == 1200
    assert get_chunk_size_for_language(config, "javascript") == 1200
    assert get_chunk_size_for_language(config, "jsx") == 1200
    assert get_chunk_size_for_language(config, "rust") == 1000  # Default
    assert get_chunk_size_for_language(config, "go") == 1000  # Default


def test_missing_project_config():
    """Test that missing project config doesn't break anything."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        # No .embecode.toml file

        config = load_config(project_path=project_path)

        # Should just use defaults
        assert config.index.include == ["src/", "lib/", "tests/"]


def test_partial_config():
    """Test that partial config files work (only override specific fields)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        config_file = project_path / ".embecode.toml"

        # Only override one field
        config_file.write_text("""
[embeddings]
model = "custom-model"
""")

        config = load_config(project_path=project_path)

        # Overridden field
        assert config.embeddings.model == "custom-model"

        # Everything else should be defaults
        assert config.index.include == ["src/", "lib/", "tests/"]
        assert config.search.top_k == 10
