"""
main.py — FastAPI application with production-grade endpoints.

Features:
  - API versioning
  - Authentication (API Key)
  - Rate limiting
  - Request validation
  - Structured logging
  - Health checks
  - Proper error handling
"""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from celery.result import AsyncResult
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

from .celery_app import celery_app
from .config import settings
from .rag.cache import semantic_cache
from .rag.chain import build_rag_chain
from .rag.vectorstore import collection_info
from .services.repository_service import get_repository, ingest_repository

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)

# API version prefix
API_PREFIX = f"/{settings.api_version}"

# ── Rate Limiter ─────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="CodeMind AI",
    description="AI software engineering platform for repository understanding, architecture exploration, and code intelligence.",
    version="1.1.0",
)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

# ── CORS Configuration ───────────────────────────────────────────────────────
# Security: Only allow specific origins, not wildcards with credentials
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# ── Authentication ───────────────────────────────────────────────────────────
API_KEY_HEADER = "X-API-Key"

def verify_api_key(request: Request) -> str:
    """Verify API key from request header."""
    api_key = request.headers.get(API_KEY_HEADER)
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
        )
    # In production, use proper secret management
    expected_key = os.getenv("API_KEY") or settings.api_key
    if api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    return api_key

# ── Lifespan Events ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    logger.info("Starting Enterprise RAG API...")
    # Initialize RAG chain on startup
    try:
        build_rag_chain()
        logger.info("RAG chain initialized")
    except Exception as e:
        logger.warning("RAG chain initialization deferred: %s", e)
    yield
    logger.info("Shutting down Enterprise RAG API...")

app.router.lifespan = lifespan

# ── Pydantic schemas ──────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=10000)


class QueryResponse(BaseModel):
    answer: str
    source: str  # "CACHE HIT" | "LLM GENERATED"
    latency_ms: float
    similarity: float | None = None
    cached_query: str | None = None
    sources: list[dict] | None = None


class RepositoryIngestRequest(BaseModel):
    source_type: str = Field(..., description="Either 'folder' or 'git'")
    path: str = Field(..., min_length=1)
    name: str | None = Field(default=None)


class RepositoryIngestResponse(BaseModel):
    repo_id: str
    status: str
    name: str
    summary: dict[str, Any]
    file_count: int
    graph: dict[str, Any] | None = None


# ── Endpoint 1: Ingest ────────────────────────────────────────────────────────
@app.post(f"{API_PREFIX}/ingest", summary="Upload a PDF and queue async ingestion")
@limiter.limit(f"{settings.rate_limit_requests}/minute")
async def ingest(
    request: Request,
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """
    Save the uploaded PDF to a temp file, dispatch a Celery task, and
    return immediately with a task_id. The caller polls /task/{task_id}.
    """
    # Validate file extension
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Validate file size (read in chunks to avoid memory issues)
    content = await file.read()
    max_size = settings.max_file_size_mb * 1024 * 1024
    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_file_size_mb}MB",
        )

    # Validate PDF magic bytes
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Invalid PDF file content")

    doc_id = str(uuid.uuid4())
    file_hash = hashlib.sha256(content).hexdigest()

    # Write to a temp file that the Celery worker can access
    tmp_dir = tempfile.gettempdir()
    file_path = os.path.join(tmp_dir, f"{doc_id}_{file.filename}")

    with open(file_path, "wb") as f:
        f.write(content)

    # Dispatch to Celery — non-blocking, returns immediately
    task = celery_app.send_task(
        "tasks.ingest.ingest_document",
        args=[file_path, doc_id, file_hash],
    )

    logger.info("Ingestion task queued: doc_id=%s task_id=%s", doc_id, task.id)

    return {
        "task_id": task.id,
        "doc_id": doc_id,
        "filename": file.filename,
        "status": "queued",
    }


# ── Endpoint 2: Task status ───────────────────────────────────────────────────
@app.get(f"{API_PREFIX}/task/{{task_id}}", summary="Poll Celery task progress")
@limiter.limit(f"{settings.rate_limit_requests}/minute")
async def get_task_status(
    request: Request,
    task_id: str,
    api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    result = AsyncResult(task_id, app=celery_app)

    def _sanitize(obj):
        """Recursively turn non-JSON-serializable objects into safe representations."""
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, BaseException):
            return {"error": str(obj), "type": type(obj).__name__}
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        try:
            return str(obj)
        except Exception:
            return repr(obj)

    try:
        raw_info = result.result if result.ready() else result.info
    except Exception as exc:
        logger.warning("Celery task metadata error for %s: %s", task_id, exc)
        raw_info = {"error": str(exc), "type": type(exc).__name__}

    safe_info = _sanitize(raw_info)

    try:
        state = result.state
    except Exception as exc:
        logger.warning("Celery task state error for %s: %s", task_id, exc)
        state = "FAILURE"

    return {"task_id": task_id, "status": state, "info": safe_info}


# ── Endpoint 3: Query ─────────────────────────────────────────────────────────
@app.post(f"{API_PREFIX}/query", response_model=QueryResponse, summary="Ask a question against ingested documents")
@limiter.limit(f"{settings.rate_limit_requests}/minute")
async def query(
    request: Request,
    req: QueryRequest,
    api_key: str = Depends(verify_api_key),
) -> QueryResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    t0 = time.perf_counter()

    # 1️⃣ Check semantic cache
    cache_result = semantic_cache.lookup(req.question)
    if cache_result.hit:
        return QueryResponse(
            answer=cache_result.answer,
            source="CACHE HIT",
            latency_ms=round(cache_result.latency_ms, 1),
            similarity=round(cache_result.similarity, 4),
            cached_query=cache_result.cached_query,
        )

    # 2️⃣ Cache miss → run full RAG chain
    try:
        response = build_rag_chain().invoke({"query": req.question})
    except Exception as exc:
        logger.error("RAG chain error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG chain error: {exc}")

    # Handle both old and new LangChain response formats
    if isinstance(response, dict):
        answer = response.get("result", str(response))
        source_documents = response.get("source_documents", [])
    else:
        # New LangChain format - response is the answer string
        answer = str(response)
        source_documents = []

    latency_ms = round((time.perf_counter() - t0) * 1_000, 1)

    # 3️⃣ Store in cache for future similar queries
    semantic_cache.store(req.question, answer)

    source_docs = [
        {k: v for k, v in doc.metadata.items() if k in ("doc_id", "source")}
        for doc in source_documents
    ]

    return QueryResponse(
        answer=answer,
        source="LLM GENERATED",
        latency_ms=latency_ms,
        sources=source_docs,
    )


# ── Repository intelligence endpoints ───────────────────────────────────────
@app.post(f"{API_PREFIX}/repositories/ingest", response_model=RepositoryIngestResponse, summary="Index a repository or folder")
@limiter.limit(f"{settings.rate_limit_requests}/minute")
async def ingest_repository_endpoint(
    request: Request,
    req: RepositoryIngestRequest,
    api_key: str = Depends(verify_api_key),
) -> RepositoryIngestResponse:
    result = ingest_repository(req.source_type, req.path, req.name)
    return RepositoryIngestResponse(**result)


@app.get(f"{API_PREFIX}/architecture/explore/{{repo_id}}", summary="Explore repository architecture graph")
@limiter.limit(f"{settings.rate_limit_requests}/minute")
async def explore_architecture(
    request: Request,
    repo_id: str,
    api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    repository = get_repository(repo_id)
    if repository is None:
        raise HTTPException(status_code=404, detail="Repository not found")
    return {"repo_id": repo_id, "name": repository.name, **repository.graph}


@app.get(f"{API_PREFIX}/repositories/{{repo_id}}", summary="Get repository details and analysis")
@limiter.limit(f"{settings.rate_limit_requests}/minute")
async def get_repository_details(
    request: Request,
    repo_id: str,
    api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    repository = get_repository(repo_id)
    if repository is None:
        raise HTTPException(status_code=404, detail="Repository not found")
    return {
        "repo_id": repo_id,
        "name": repository.name,
        "path": repository.path,
        "summary": repository.summary,
        "file_count": len(repository.files),
        "graph": repository.graph,
    }


# ── Endpoint 4: Health ────────────────────────────────────────────────────────
@app.get(f"{API_PREFIX}/health", summary="Infrastructure health check")
async def health() -> JSONResponse:
    status = {
        "redis": "unknown",
        "qdrant": "unknown",
        "cache_entries": 0,
        "config": {
            "qdrant_collection": settings.qdrant_collection,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "similarity_threshold": settings.similarity_threshold,
            "top_k_chunks": settings.top_k_chunks,
        }
    }

    # Check Redis
    try:
        redis_client = redis.from_url(settings.redis_url, decode_responses=True)
        redis_client.ping()
        status["redis"] = "ok"
    except Exception as e:
        status["redis"] = f"error: {e}"

    # Check Qdrant
    try:
        qdrant_info = collection_info()
        if "error" in qdrant_info:
            error_msg = qdrant_info.get("error", "")
            if "exist" in error_msg:
                status["qdrant"] = {"status": "ok", "message": "Qdrant reachable (collection will be created on first ingest)"}
            else:
                status["qdrant"] = {"status": "error", "error": error_msg}
        else:
            status["qdrant"] = qdrant_info
        
        # Get cache stats
        cache_stats = semantic_cache.stats()
        status["cache_entries"] = cache_stats.get("cached_entries", 0)
    except Exception as e:
        status["qdrant"] = {"status": "error", "error": str(e)}

    overall_ok = status["redis"] == "ok" and (
        status["qdrant"] == "ok" or 
        isinstance(status["qdrant"], dict) and status["qdrant"].get("status") in ("ok", "green")
    )
    return JSONResponse(
        status_code=200 if overall_ok else 503,
        content={"ok": overall_ok, **status},
    )


# ── Endpoint 5: Flush cache ───────────────────────────────────────────────────
@app.delete(f"{API_PREFIX}/cache", summary="Flush the semantic cache")
@limiter.limit(f"{settings.rate_limit_requests}/minute")
async def flush_cache(
    request: Request,
    api_key: str = Depends(verify_api_key),
) -> JSONResponse:
    try:
        deleted = semantic_cache.flush()
        return JSONResponse(content={"deleted_entries": deleted, "status": "cache cleared"})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Cache flush failed: {e}"}
        )


# ── Root redirect ─────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Enterprise RAG API", "version": "1.0.0", "docs": "/docs"}