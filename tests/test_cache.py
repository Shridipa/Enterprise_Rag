"""Tests for semantic cache module."""
import pytest
from unittest.mock import patch, MagicMock

from backend.rag.cache import SemanticCache, CacheResult


class TestSemanticCache:
    """Test semantic cache functionality."""

    @pytest.fixture
    def cache(self, monkeypatch):
        """Create a cache instance with mocked dependencies."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-1234567890")
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        
        with patch("backend.rag.cache.get_embeddings") as mock_embed:
            mock_embedder = MagicMock()
            mock_embedder.embed_query.return_value = [0.1] * 1536
            mock_embed.return_value = mock_embedder
            
            with patch("backend.rag.cache.QdrantClient") as mock_client:
                mock_qdrant = MagicMock()
                mock_qdrant.get_collections.return_value = MagicMock(collections=[])
                mock_qdrant.search.return_value = []
                mock_qdrant.get_collection.return_value = MagicMock(points_count=0)
                mock_client.return_value = mock_qdrant
                
                return SemanticCache()

    def test_cache_result_dataclass(self):
        """Test CacheResult dataclass."""
        result = CacheResult(
            hit=True,
            answer="Test answer",
            similarity=0.95,
            latency_ms=5.2,
            source="CACHE HIT",
            cached_query="What is the answer?",
        )
        assert result.hit is True
        assert result.answer == "Test answer"
        assert result.similarity == 0.95

    def test_cache_lookup_miss(self, cache):
        """Test cache lookup returns miss when no results."""
        result = cache.lookup("What is the refund policy?")
        assert result.hit is False
        assert result.source == "LLM GENERATED"
        assert result.similarity == 0.0

    def test_cache_lookup_hit(self, cache):
        """Test cache lookup returns hit when similar query found."""
        # Mock search to return a result
        mock_result = MagicMock()
        mock_result.score = 0.95
        mock_result.payload = {
            "query": "What is the refund policy?",
            "answer": "Refunds are processed within 5 business days.",
        }
        cache._client.search.return_value = [mock_result]
        
        result = cache.lookup("What is the refund policy?")
        assert result.hit is True
        assert result.source == "CACHE HIT"
        assert result.similarity == 0.95

    def test_cache_store(self, cache):
        """Test cache store method."""
        cache.store("What is the refund policy?", "Refunds are processed within 5 business days.")
        
        # Verify upsert was called
        cache._client.upsert.assert_called_once()
        call_args = cache._client.upsert.call_args
        assert call_args[1]["collection_name"] == "semantic_cache"
        assert len(call_args[1]["points"]) == 1

    def test_cache_flush(self, cache):
        """Test cache flush method."""
        mock_info = MagicMock()
        mock_info.points_count = 100
        cache._client.get_collection.return_value = mock_info
        
        result = cache.flush()
        assert result == 100
        cache._client.delete_collection.assert_called_once_with("semantic_cache")

    def test_cache_stats(self, cache):
        """Test cache stats method."""
        mock_info = MagicMock()
        mock_info.points_count = 50
        cache._client.get_collection.return_value = mock_info
        
        stats = cache.stats()
        assert stats["cached_entries"] == 50
        assert stats["threshold"] == 0.90

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        assert SemanticCache._cosine(v1, v2) == pytest.approx(1.0)
        
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        assert SemanticCache._cosine(v1, v2) == pytest.approx(0.0)
        
        v1 = [1.0, 1.0, 0.0]
        v2 = [1.0, 1.0, 0.0]
        assert SemanticCache._cosine(v1, v2) == pytest.approx(1.0)
