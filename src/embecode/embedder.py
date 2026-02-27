"""Embedding generation using sentence-transformers (local) or API providers."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from embecode.config import EmbeddingsConfig


class EmbedderError(Exception):
    """Base exception for embedder errors."""


class ModelNotFoundError(EmbedderError):
    """Raised when a model cannot be loaded."""


class APIKeyMissingError(EmbedderError):
    """Raised when an API key is required but not found."""


class Embedder:
    """
    Lazy-loading embedder supporting local and API-based models.

    Local models use sentence-transformers and run entirely offline.
    API models require an API key in the environment.
    """

    def __init__(self, config: EmbeddingsConfig) -> None:
        """
        Initialize embedder (does not load model yet).

        Args:
            config: Embedding configuration with model name and optional API key env var.
        """
        self.config = config
        self._model = None
        self._is_api_model = self._detect_api_model(config.model)

    @staticmethod
    def _detect_api_model(model_name: str) -> bool:
        """Detect if model requires API access based on name patterns."""
        api_patterns = [
            "voyage-",
            "text-embedding-",  # OpenAI
            "cohere-",
        ]
        return any(pattern in model_name for pattern in api_patterns)

    def _load_local_model(self) -> None:
        """Load a local sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            msg = (
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise ModelNotFoundError(msg) from e

        try:
            # Lazy load model on first use
            # trust_remote_code=True is required for models with custom architectures
            # (e.g. nomic-ai/nomic-embed-text-v1.5 uses nomic_bert)
            self._model = SentenceTransformer(self.config.model, trust_remote_code=True)
        except Exception as e:
            msg = f"Failed to load model '{self.config.model}': {e}"
            raise ModelNotFoundError(msg) from e

    def _load_api_model(self) -> None:
        """Set up API-based embedding client."""
        if self.config.api_key_env is None:
            msg = (
                f"API model '{self.config.model}' requires an API key. "
                "Set api_key_env in config to specify the environment variable name."
            )
            raise APIKeyMissingError(msg)

        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            msg = (
                f"API key not found in environment variable '{self.config.api_key_env}'. "
                f"Set it with: export {self.config.api_key_env}=your-key-here"
            )
            raise APIKeyMissingError(msg)

        # Determine provider from model name and set up client
        if "voyage-" in self.config.model:
            try:
                import voyageai  # type: ignore[import-untyped]

                self._model = voyageai.Client(api_key=api_key)
            except ImportError as e:
                msg = "voyageai not installed. Install with: pip install voyageai"
                raise ModelNotFoundError(msg) from e

        elif "text-embedding-" in self.config.model:
            try:
                import openai

                self._model = openai.OpenAI(api_key=api_key)
            except ImportError as e:
                msg = "openai not installed. Install with: pip install openai"
                raise ModelNotFoundError(msg) from e

        elif "cohere-" in self.config.model:
            try:
                import cohere  # type: ignore[import-untyped]

                self._model = cohere.Client(api_key=api_key)
            except ImportError as e:
                msg = "cohere not installed. Install with: pip install cohere"
                raise ModelNotFoundError(msg) from e

        else:
            msg = (
                f"Unknown API model provider for '{self.config.model}'. "
                "Supported: voyage-*, text-embedding-* (OpenAI), cohere-*"
            )
            raise ModelNotFoundError(msg)

    def _ensure_loaded(self) -> None:
        """Lazy load the model on first embed() call."""
        if self._model is not None:
            return

        if self._is_api_model:
            self._load_api_model()
        else:
            self._load_local_model()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed (typically chunk content + context).

        Returns:
            List of embedding vectors (one per input text).

        Raises:
            ModelNotFoundError: If model cannot be loaded.
            APIKeyMissingError: If API key is required but missing.
        """
        if not texts:
            return []

        self._ensure_loaded()

        if self._is_api_model:
            return self._embed_api(texts)
        else:
            return self._embed_local(texts)

    def unload(self) -> None:
        """Release the loaded model from memory.

        Drops the reference to the model object so the Python GC (and the
        underlying framework, e.g. PyTorch) can free the weights.  The model
        will be reloaded lazily on the next :meth:`embed` call, so this is safe
        to call after a bulk indexing run when no immediate searches are
        expected.
        """
        self._model = None

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using local sentence-transformers model."""
        # Convert numpy arrays to Python lists for JSON serialization
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [embedding.tolist() for embedding in embeddings]

    def _embed_api(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using API provider."""
        if "voyage-" in self.config.model:
            # Voyage AI
            result = self._model.embed(texts, model=self.config.model)
            return result.embeddings

        elif "text-embedding-" in self.config.model:
            # OpenAI
            response = self._model.embeddings.create(
                input=texts,
                model=self.config.model,
            )
            return [item.embedding for item in response.data]

        elif "cohere-" in self.config.model:
            # Cohere
            response = self._model.embed(
                texts=texts,
                model=self.config.model,
                input_type="search_document",
            )
            return response.embeddings

        else:
            msg = f"API embedding not implemented for model: {self.config.model}"
            raise NotImplementedError(msg)

    @property
    def dimension(self) -> int:
        """
        Get embedding dimension for the configured model.

        Returns:
            Embedding vector dimension.
        """
        self._ensure_loaded()

        # Known dimensions for common models
        known_dimensions = {
            "nomic-ai/nomic-embed-text-v1.5": 768,
            "BAAI/bge-base-en-v1.5": 768,
            "all-MiniLM-L6-v2": 384,
            "voyage-code-3": 1024,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }

        if self.config.model in known_dimensions:
            return known_dimensions[self.config.model]

        # For local models, get dimension from model
        if not self._is_api_model:
            return self._model.get_sentence_embedding_dimension()

        # For unknown API models, embed a test string
        test_embedding = self.embed(["test"])
        return len(test_embedding[0])
