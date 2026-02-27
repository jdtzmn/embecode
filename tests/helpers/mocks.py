"""Shared mock objects for the embecode test suite.

Provides common mock classes used across multiple test modules to avoid
duplication and ensure consistency.
"""

from __future__ import annotations


class MockEmbedder:
    """Fast mock embedder that returns zero vectors.

    Avoids loading a real sentence-transformers model while still exercising
    embedding storage and database code paths.

    Used by: test_performance.py, test_integration.py, test_memory.py
    """

    def __init__(self, dimension: int = 384) -> None:
        """Initialize mock embedder with configurable dimension."""
        self._dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate zero-vector embeddings (fast, deterministic)."""
        return [[0.0] * self._dimension for _ in texts]

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    def unload(self) -> None:
        """No-op: mock embedder has nothing to unload."""
