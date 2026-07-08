"""Tests for FastAPI main module."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from backend.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_api_key(monkeypatch):
    """Set a test API key."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-1234567890")


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_endpoint(self, client, mock_api_key):
        """Test that health endpoint returns status."""
        with patch("backend.main.redis.from_url") as mock_redis:
            mock_redis_client = MagicMock()
            mock_redis_client.ping.return_value = True
            mock_redis.return_value = mock_redis_client
            
            with patch("backend.main.collection_info") as mock_info:
                mock_info.return_value = {"status": "ok", "vectors_count": 0}
                
                with patch("backend.main.semantic_cache.stats") as mock_stats:
                    mock_stats.return_value = {"cached_entries": 0}
                    
                    response = client.get("/v1/health")
                    assert response.status_code == 200
                    data = response.json()
                    assert "ok" in data
                    assert data["redis"] == "ok"


class TestQueryEndpoint:
    """Test query endpoint."""

    def test_query_empty_question(self, client, mock_api_key):
        """Test that empty question is rejected."""
        response = client.post(
            "/v1/query",
            json={"question": ""},
            headers={"X-API-Key": "test-api-key-12345"},
        )
        # FastAPI returns 422 for validation errors (min_length=1)
        assert response.status_code in (400, 422)

    def test_query_missing_api_key(self, client, mock_api_key):
        """Test that missing API key is rejected."""
        response = client.post(
            "/v1/query",
            json={"question": "What is the refund policy?"},
        )
        assert response.status_code == 401

    def test_query_invalid_api_key(self, client, mock_api_key):
        """Test that invalid API key is rejected."""
        response = client.post(
            "/v1/query",
            json={"question": "What is the refund policy?"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 403


class TestIngestEndpoint:
    """Test ingest endpoint."""

    def test_ingest_missing_api_key(self, client, mock_api_key):
        """Test that missing API key is rejected."""
        response = client.post(
            "/v1/ingest",
            files={"file": ("test.pdf", b"%PDF-1.4 test", "application/pdf")},
        )
        assert response.status_code == 401

    def test_ingest_invalid_file_type(self, client, mock_api_key):
        """Test that non-PDF files are rejected."""
        response = client.post(
            "/v1/ingest",
            files={"file": ("test.txt", b"not a pdf", "text/plain")},
            headers={"X-API-Key": "test-api-key-12345"},
        )
        assert response.status_code == 400

    def test_ingest_invalid_pdf_content(self, client, mock_api_key):
        """Test that invalid PDF content is rejected."""
        response = client.post(
            "/v1/ingest",
            files={"file": ("test.pdf", b"not a pdf", "application/pdf")},
            headers={"X-API-Key": "test-api-key-12345"},
        )
        assert response.status_code == 400

    def test_ingest_valid_pdf(self, client, mock_api_key):
        """Test that valid PDF is accepted."""
        with patch("backend.main.celery_app.send_task") as mock_task:
            mock_task.return_value = MagicMock(id="test-task-id")
            
            response = client.post(
                "/v1/ingest",
                files={"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")},
                headers={"X-API-Key": "test-api-key-12345"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "task_id" in data
            assert data["status"] == "queued"


class TestCacheEndpoint:
    """Test cache flush endpoint."""

    def test_cache_flush_requires_auth(self, client, mock_api_key):
        """Test that cache flush requires authentication."""
        response = client.delete("/v1/cache")
        assert response.status_code == 401

    def test_cache_flush_invalid_key(self, client, mock_api_key):
        """Test that cache flush rejects invalid API key."""
        response = client.delete(
            "/v1/cache",
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 403