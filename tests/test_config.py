"""Tests for configuration module."""
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from backend.config import Settings
from backend.rag.vectorstore import get_embeddings, get_qdrant_client


class TestSettings:
    """Test Settings validation."""

    def test_valid_settings(self, monkeypatch):
        """Test that valid settings are accepted."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-1234567890")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        
        settings = Settings()
        assert settings.openai_api_key == "sk-test-key-1234567890"
        assert settings.redis_url == "redis://localhost:6379/0"
        assert settings.qdrant_url == "http://localhost:6333"

    def test_invalid_openai_key_placeholder(self, monkeypatch):
        """Test that placeholder API key is rejected."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-placeholder")
        
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        
        assert "OPENAI_API_KEY must be a valid OpenAI API key" in str(exc_info.value)

    def test_invalid_openai_key_format(self, monkeypatch):
        """Test that invalid API key format is rejected."""
        monkeypatch.setenv("OPENAI_API_KEY", "invalid-key")
        
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        
        assert "OPENAI_API_KEY must start with 'sk-'" in str(exc_info.value)

    def test_invalid_redis_url(self, monkeypatch):
        """Test that invalid Redis URL is rejected."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-1234567890")
        monkeypatch.setenv("REDIS_URL", "http://localhost:6379")
        
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        
        assert "REDIS_URL must start with 'redis://'" in str(exc_info.value)

    def test_invalid_qdrant_url(self, monkeypatch):
        """Test that invalid Qdrant URL is rejected."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-1234567890")
        monkeypatch.setenv("QDRANT_URL", "redis://localhost:6333")
        
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        
        assert "QDRANT_URL must be a valid HTTP URL" in str(exc_info.value)

    def test_similarity_threshold_bounds(self, monkeypatch):
        """Test that similarity threshold is validated."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-1234567890")
        
        # Valid
        monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.5")
        settings = Settings()
        assert settings.similarity_threshold == 0.5
        
        # Invalid - too high
        monkeypatch.setenv("SIMILARITY_THRESHOLD", "1.5")
        with pytest.raises(ValidationError):
            Settings()

    def test_chunk_size_bounds(self, monkeypatch):
        """Test that chunk size is validated."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-1234567890")
        
        # Valid
        monkeypatch.setenv("CHUNK_SIZE", "1024")
        settings = Settings()
        assert settings.chunk_size == 1024
        
        # Invalid - too large
        monkeypatch.setenv("CHUNK_SIZE", "100000")
        with pytest.raises(ValidationError):
            Settings()

    def test_qdrant_client_initialization_is_compatible(self, monkeypatch):
        """Ensure the Qdrant client can be instantiated with the installed version."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-1234567890")
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

        client = get_qdrant_client()
        assert client is not None

    def test_embeddings_fallback_when_openai_call_fails(self, monkeypatch):
        """Ensure a local embedding fallback is used when OpenAI embeddings fail."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        with patch("backend.rag.vectorstore.OpenAIEmbeddings") as mock_openai_embeddings:
            instance = mock_openai_embeddings.return_value
            instance.embed_documents.side_effect = Exception("openai unavailable")
            instance.embed_query.side_effect = Exception("openai unavailable")

            embeddings = get_embeddings()
            vectors = embeddings.embed_documents(["hello world"])

            assert len(vectors) == 1
            assert len(vectors[0]) == 1536