"""Tests for embedder.py - embedding generation."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

from embecode.config import EmbeddingsConfig
from embecode.embedder import APIKeyMissingError, Embedder, ModelNotFoundError


class TestEmbedderLocalModels:
    """Test suite for local sentence-transformers models."""

    @patch("sentence_transformers.SentenceTransformer")
    def test_lazy_loading(self, mock_st: Mock) -> None:
        """Model should not load until first embed() call."""
        config = EmbeddingsConfig(model="all-MiniLM-L6-v2")
        embedder = Embedder(config)

        # Model not loaded yet
        assert embedder._model is None
        mock_st.assert_not_called()

        # Trigger lazy load
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_st.return_value = mock_model

        embedder.embed(["test"])

        # Now model is loaded
        assert embedder._model is not None
        mock_st.assert_called_once_with("all-MiniLM-L6-v2", trust_remote_code=True)

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_single_text(self, mock_st: Mock) -> None:
        """Should embed a single text correctly."""
        import numpy as np

        config = EmbeddingsConfig(model="all-MiniLM-L6-v2")
        embedder = Embedder(config)

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_st.return_value = mock_model

        result = embedder.embed(["test text"])

        assert len(result) == 1
        assert result[0] == pytest.approx([0.1, 0.2, 0.3])
        mock_model.encode.assert_called_once_with(["test text"], convert_to_numpy=True)

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_multiple_texts(self, mock_st: Mock) -> None:
        """Should embed multiple texts in batch."""
        config = EmbeddingsConfig(model="all-MiniLM-L6-v2")
        embedder = Embedder(config)

        mock_model = MagicMock()
        # Simulate numpy array conversion to list
        import numpy as np

        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        mock_st.return_value = mock_model

        result = embedder.embed(["text1", "text2", "text3"])

        assert len(result) == 3
        assert result[0] == pytest.approx([0.1, 0.2])
        assert result[1] == pytest.approx([0.3, 0.4])
        assert result[2] == pytest.approx([0.5, 0.6])

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_empty_list(self, mock_st: Mock) -> None:
        """Should handle empty input gracefully."""
        config = EmbeddingsConfig(model="all-MiniLM-L6-v2")
        embedder = Embedder(config)

        result = embedder.embed([])

        assert result == []
        mock_st.assert_not_called()  # Should not even load model

    @patch("sentence_transformers.SentenceTransformer")
    def test_model_not_found(self, mock_st: Mock) -> None:
        """Should raise ModelNotFoundError if model fails to load."""
        config = EmbeddingsConfig(model="nonexistent-model")
        embedder = Embedder(config)

        mock_st.side_effect = Exception("Model not found")

        with pytest.raises(ModelNotFoundError, match="Failed to load model"):
            embedder.embed(["test"])

    def test_sentence_transformers_not_installed(self) -> None:
        """Should raise ModelNotFoundError if sentence-transformers not installed."""
        config = EmbeddingsConfig(model="all-MiniLM-L6-v2")
        embedder = Embedder(config)

        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(ModelNotFoundError, match="sentence-transformers not installed"):
                embedder.embed(["test"])

    @patch("sentence_transformers.SentenceTransformer")
    def test_dimension_property_local(self, mock_st: Mock) -> None:
        """Should return correct dimension for local models."""
        config = EmbeddingsConfig(model="all-MiniLM-L6-v2")
        embedder = Embedder(config)

        # all-MiniLM-L6-v2 is in the known_dimensions dict, so it returns 384
        # without needing to query the model
        assert embedder.dimension == 384

    @patch("sentence_transformers.SentenceTransformer")
    def test_dimension_known_model(self, mock_st: Mock) -> None:
        """Should use known dimensions without loading model."""
        # Use the full model name that matches the known_dimensions dict key
        config = EmbeddingsConfig(model="nomic-ai/nomic-embed-text-v1.5")
        embedder = Embedder(config)

        mock_model = MagicMock()
        mock_st.return_value = mock_model

        # Access dimension (triggers lazy load but uses known value)
        dim = embedder.dimension
        assert dim == 768


class TestEmbedderAPIModels:
    """Test suite for API-based embedding models."""

    def test_detect_voyage_model(self) -> None:
        """Should detect Voyage AI models as API models."""
        assert Embedder._detect_api_model("voyage-code-3") is True
        assert Embedder._detect_api_model("voyage-large-2") is True

    def test_detect_openai_model(self) -> None:
        """Should detect OpenAI models as API models."""
        assert Embedder._detect_api_model("text-embedding-3-large") is True
        assert Embedder._detect_api_model("text-embedding-ada-002") is True

    def test_detect_cohere_model(self) -> None:
        """Should detect Cohere models as API models."""
        assert Embedder._detect_api_model("cohere-embed-v3") is True

    def test_detect_local_model(self) -> None:
        """Should not detect local models as API models."""
        assert Embedder._detect_api_model("nomic-embed-text-v1.5") is False
        assert Embedder._detect_api_model("all-MiniLM-L6-v2") is False

    def test_api_key_env_not_set(self) -> None:
        """Should raise APIKeyMissingError if api_key_env not configured."""
        config = EmbeddingsConfig(model="voyage-code-3", api_key_env=None)
        embedder = Embedder(config)

        with pytest.raises(APIKeyMissingError, match="requires an API key"):
            embedder.embed(["test"])

    def test_api_key_missing_from_env(self) -> None:
        """Should raise APIKeyMissingError if env var not set."""
        config = EmbeddingsConfig(model="voyage-code-3", api_key_env="VOYAGE_API_KEY")
        embedder = Embedder(config)

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(APIKeyMissingError, match="not found in environment variable"):
                embedder.embed(["test"])

    def test_voyage_embed(self) -> None:
        """Should use Voyage AI client for voyage-* models."""
        config = EmbeddingsConfig(model="voyage-code-3", api_key_env="VOYAGE_API_KEY")
        embedder = Embedder(config)

        mock_voyageai = MagicMock()
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1, 0.2], [0.3, 0.4]]
        mock_client.embed.return_value = mock_result
        mock_voyageai.Client.return_value = mock_client

        with patch.dict(os.environ, {"VOYAGE_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {"voyageai": mock_voyageai}):
                result = embedder.embed(["text1", "text2"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_client.embed.assert_called_once_with(["text1", "text2"], model="voyage-code-3")

    def test_openai_embed(self) -> None:
        """Should use OpenAI client for text-embedding-* models."""
        config = EmbeddingsConfig(model="text-embedding-3-large", api_key_env="OPENAI_API_KEY")
        embedder = Embedder(config)

        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2]),
            MagicMock(embedding=[0.3, 0.4]),
        ]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {"openai": mock_openai}):
                result = embedder.embed(["text1", "text2"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_client.embeddings.create.assert_called_once_with(
            input=["text1", "text2"], model="text-embedding-3-large"
        )

    def test_cohere_embed(self) -> None:
        """Should use Cohere client for cohere-* models."""
        config = EmbeddingsConfig(model="cohere-embed-v3", api_key_env="COHERE_API_KEY")
        embedder = Embedder(config)

        mock_cohere = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2], [0.3, 0.4]]
        mock_client.embed.return_value = mock_response
        mock_cohere.Client.return_value = mock_client

        with patch.dict(os.environ, {"COHERE_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {"cohere": mock_cohere}):
                result = embedder.embed(["text1", "text2"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_client.embed.assert_called_once_with(
            texts=["text1", "text2"],
            model="cohere-embed-v3",
            input_type="search_document",
        )

    def test_unknown_api_provider(self) -> None:
        """Should raise error for unknown API model patterns."""
        # "unknown-api-model" doesn't match any API pattern, so _is_api_model is False.
        # It will try to load as a local model and fail with ModelNotFoundError.
        config = EmbeddingsConfig(model="unknown-api-model", api_key_env="SOME_API_KEY")
        embedder = Embedder(config)

        with patch.dict(os.environ, {"SOME_API_KEY": "test-key"}):
            with pytest.raises(ModelNotFoundError):
                embedder.embed(["test"])

    def test_dimension_known_api_model(self) -> None:
        """Should use known dimension for API models."""
        config = EmbeddingsConfig(model="voyage-code-3", api_key_env="VOYAGE_API_KEY")
        embedder = Embedder(config)

        mock_voyageai = MagicMock()
        mock_client = MagicMock()
        mock_voyageai.Client.return_value = mock_client

        with patch.dict(os.environ, {"VOYAGE_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {"voyageai": mock_voyageai}):
                assert embedder.dimension == 1024

        # Should not make API call for known dimension
        mock_client.embed.assert_not_called()


class TestEmbedderIntegration:
    """Integration tests with real models (optional, requires models installed)."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_real_local_model(self) -> None:
        """Test with real all-MiniLM-L6-v2 model (small, fast download)."""
        pytest.importorskip("sentence_transformers")

        config = EmbeddingsConfig(model="all-MiniLM-L6-v2")
        embedder = Embedder(config)

        result = embedder.embed(["Hello world", "Test embedding"])

        assert len(result) == 2
        assert len(result[0]) == 384  # all-MiniLM-L6-v2 dimension
        assert len(result[1]) == 384
        assert isinstance(result[0][0], float)

        # Verify embeddings are different
        assert result[0] != result[1]

        # Verify dimension property
        assert embedder.dimension == 384
