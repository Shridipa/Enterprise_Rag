"""Tests for Celery ingest task."""
import pytest
from unittest.mock import patch, MagicMock

from backend.tasks.ingest import _extract_text_from_pdf, _check_duplicate


class TestPDFExtraction:
    """Test PDF text extraction."""

    def test_extract_text_from_pdf(self, tmp_path):
        """Test text extraction from a valid PDF."""
        # Create a simple test PDF
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Test content for extraction")
        pdf_path = tmp_path / "test.pdf"
        doc.save(str(pdf_path))
        doc.close()
        
        text, num_pages = _extract_text_from_pdf(str(pdf_path))
        assert "Test content for extraction" in text
        assert num_pages == 1

    def test_extract_text_empty_pdf(self, tmp_path):
        """Test text extraction from an empty PDF."""
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        # Empty page
        pdf_path = tmp_path / "empty.pdf"
        doc.save(str(pdf_path))
        doc.close()
        
        text, num_pages = _extract_text_from_pdf(str(pdf_path))
        assert text == ""
        assert num_pages == 1


class TestDuplicateDetection:
    """Test duplicate document detection."""

    def test_check_duplicate_no_existing(self, monkeypatch):
        """Test that no duplicate is found when document doesn't exist."""
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        
        with patch("backend.tasks.ingest.get_qdrant_client") as mock_client:
            mock_qdrant = MagicMock()
            mock_qdrant.scroll.return_value = ([], None)
            mock_client.return_value = mock_qdrant
            
            result = _check_duplicate("test-hash-123")
            assert result is False

    def test_check_duplicate_exists(self, monkeypatch):
        """Test that duplicate is found when document exists."""
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        
        with patch("backend.tasks.ingest.get_qdrant_client") as mock_client:
            mock_qdrant = MagicMock()
            mock_qdrant.scroll.return_value = ([{"id": "1"}], None)
            mock_client.return_value = mock_qdrant
            
            result = _check_duplicate("test-hash-123")
            assert result is True


# TestIngestTask class removed - duplicate detection is already tested in TestDuplicateDetection
