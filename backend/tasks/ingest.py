"""
tasks/ingest.py — Celery task: PDF → chunks → embeddings → Qdrant.

The task is bound (bind=True) so it can update its own state, which lets
the FastAPI /task/{id} endpoint stream live progress to the frontend.

Features:
  - Idempotent ingestion (deduplication by file hash)
  - OCR support for scanned PDFs
  - File validation
  - Retry with jitter
"""
from __future__ import annotations

import logging
import os
import random

import fitz  # PyMuPDF

from ..celery_app import celery_app
from ..config import settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..rag.vectorstore import get_vectorstore, get_qdrant_client

logger = logging.getLogger(__name__)

# Try to import OCR support
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("OCR not available. Install pytesseract and Pillow for scanned PDF support.")


def _extract_text_with_ocr(page) -> str:
    """Extract text from a page using OCR as fallback."""
    if not OCR_AVAILABLE:
        return ""
    
    try:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return pytesseract.image_to_string(img)
    except Exception as e:
        logger.warning("OCR extraction failed: %s", e)
        return ""


def _extract_text_from_pdf(file_path: str) -> tuple[str, int]:
    """Extract text from PDF, with OCR fallback for scanned pages."""
    doc = fitz.open(file_path)
    pages_text: list[str] = []
    
    for page in doc:
        text = page.get_text()
        # If no text extracted, try OCR
        if not text.strip() and OCR_AVAILABLE:
            text = _extract_text_with_ocr(page)
        pages_text.append(text)
    
    doc.close()
    return "\n\n".join(pages_text), len(pages_text)


def _check_duplicate(file_hash: str) -> bool:
    """Check if a document with this hash already exists."""
    try:
        client = get_qdrant_client()
        # Search for existing document by hash
        results = client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter={"must": [{"key": "file_hash", "match": {"value": file_hash}}]},
            limit=1,
        )
        return len(results[0]) > 0
    except Exception:
        return False


@celery_app.task(
    bind=True,
    max_retries=3,
    name="tasks.ingest.ingest_document",
    time_limit=600,  # Hard limit: 10 minutes max
    soft_time_limit=540,  # Soft limit: 9 minutes
    acks_late=True,
)
def ingest_document(self, file_path: str, doc_id: str, file_hash: str) -> dict:
    """
    Celery task that processes a PDF end-to-end:
      PARSING  → extract raw text with PyMuPDF (with OCR fallback)
      CHUNKING → split into overlapping chunks
      EMBEDDING → embed each chunk via OpenAI
      UPSERTING → write to Qdrant
    Returns a dict with status and chunk count on success.
    """
    try:
        # ── Step 1: Check for duplicate ───────────────────────────────────────────
        self.update_state(state="PARSING", meta={"doc_id": doc_id, "step": 1, "total_steps": 4})
        
        if _check_duplicate(file_hash):
            logger.info("[%s] Duplicate file detected, skipping ingestion", doc_id)
            return {
                "status": "SUCCESS",
                "doc_id": doc_id,
                "chunks_ingested": 0,
                "pages": 0,
                "message": "Duplicate file - skipped",
            }

        # ── Step 2: Parse PDF ─────────────────────────────────────────────────
        logger.info("[%s] Opening PDF: %s", doc_id, file_path)

        full_text, num_pages = _extract_text_from_pdf(file_path)
        
        if not full_text.strip():
            raise ValueError("PDF contained no extractable text. It may be a scanned image PDF without OCR support.")

        logger.info("[%s] Extracted %d chars from %d pages", doc_id, len(full_text), num_pages)

        # ── Step 3: Chunk ─────────────────────────────────────────────────────
        self.update_state(state="CHUNKING", meta={"doc_id": doc_id, "step": 2, "total_steps": 4})

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.create_documents(
            [full_text],
            metadatas=[{"doc_id": doc_id, "source": os.path.basename(file_path), "file_hash": file_hash}],
        )
        logger.info("[%s] Created %d chunks", doc_id, len(chunks))

        # ── Step 4: Embed + Upsert ────────────────────────────────────────────
        self.update_state(
            state="EMBEDDING",
            meta={"doc_id": doc_id, "step": 3, "total_steps": 4, "chunks": len(chunks)},
        )

        try:
            vs = get_vectorstore()
            # add_documents handles batching internally
            vs.add_documents(chunks)
        except Exception as e:
            logger.error("[%s] Failed to embed/upsert: %s", doc_id, e, exc_info=True)
            self.update_state(state="FAILURE", meta={"error": f"Embedding/Upsert failed: {e}"})
            raise

        # ── Step 5: Cleanup ───────────────────────────────────────────────────
        self.update_state(state="CLEANUP", meta={"doc_id": doc_id, "step": 4, "total_steps": 4})
        try:
            os.remove(file_path)
        except OSError:
            pass

        result = {
            "status": "SUCCESS",
            "doc_id": doc_id,
            "chunks_ingested": len(chunks),
            "pages": num_pages,
        }
        logger.info("[%s] Ingestion complete: %s", doc_id, result)
        return result

    except Exception as exc:
        logger.error("[%s] Ingestion failed: %s", doc_id, exc, exc_info=True)
        self.update_state(state="FAILURE", meta={"error": str(exc)})
        # Exponential back-off with jitter
        countdown = (2 ** self.request.retries) + random.randint(0, 10)
        raise self.retry(exc=exc, countdown=countdown)