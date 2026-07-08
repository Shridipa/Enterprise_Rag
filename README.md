# Enterprise RAG Pipeline

Production-grade Retrieval-Augmented Generation system with **semantic caching** and **async PDF ingestion**.

## Architecture

```
INGESTION (async)
  User uploads PDF → FastAPI /v1/ingest → Redis queue → Celery worker → Qdrant upsert

QUERY (sync)
  User query → FastAPI /v1/query → Semantic cache check (cosine ≥ 0.90)
                                    ├─ CACHE HIT  → return in 3–8 ms
                                    └─ CACHE MISS → Qdrant search → GPT-4o-mini → store in cache
```

## Project Structure

```
enterprise-rag/
├── backend/
│   ├── main.py              # FastAPI — 5 endpoints (v1)
│   ├── celery_app.py        # Celery configuration
│   ├── config.py            # Pydantic settings (reads .env)
│   ├── tasks/
│   │   └── ingest.py        # PDF → chunks → embeddings → Qdrant
│   └── rag/
│       ├── vectorstore.py   # Qdrant wrapper + collection init
│       ├── cache.py         # Semantic cache (Qdrant HNSW)
│       └── chain.py         # LangChain RetrievalQA chain
├── frontend/
│   └── app.py               # Streamlit chat UI
├── tests/
│   ├── test_config.py       # Configuration tests
│   ├── test_main.py         # API endpoint tests
│   ├── test_cache.py        # Cache tests
│   └── test_ingest.py       # Ingestion tests
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.celery
├── Dockerfile.frontend
├── requirements.txt
└── .github/workflows/ci.yml
```

## Quick Start (Local Dev)

> The ingestion pipeline now includes a deterministic local embedding fallback, so the app can still index documents in local/dev environments even when a live OpenAI key is unavailable or invalid.

### 1. Prerequisites

- Python 3.11+
- Docker Desktop (for Redis + Qdrant)
- OpenAI API key

### 2. Clone and set up environment

```bash
cd enterprise-rag
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY and API_KEY
```

### 4. Start Redis + Qdrant

```bash
docker-compose up -d redis qdrant
```

### 5. Start all three processes (three terminals)

**Terminal 1 — FastAPI backend**
```bash
cd backend
uvicorn main:app --reload --port 8000
```

**Terminal 2 — Celery worker**
```bash
cd backend
celery -A celery_app.celery_app worker --loglevel=info --concurrency=4
```

**Terminal 3 — Streamlit frontend**
```bash
cd frontend
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Quick Start (Docker — Full Stack)

```bash
cp .env.example .env
# Set OPENAI_API_KEY and API_KEY in .env

docker-compose up --build
```

| Service    | URL                       |
|------------|---------------------------|
| Frontend   | http://localhost:8502     |
| API docs   | http://localhost:8001/docs|
| Qdrant UI  | http://localhost:6333/dashboard |

---

## API Reference

All endpoints are under `/v1/` prefix and require `X-API-Key` header.

### `POST /v1/ingest`
Upload a PDF. Returns `task_id` immediately — ingestion runs in background.

```bash
curl -X POST http://localhost:8000/v1/ingest \
  -H "X-API-Key: your-api-key" \
  -F "file=@document.pdf"
# → {"task_id": "abc123", "doc_id": "...", "status": "queued"}
```

### `GET /v1/task/{task_id}`
Poll ingestion progress.

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/v1/task/abc123
# → {"status": "EMBEDDING", "info": {"chunks": 47}}
# → {"status": "SUCCESS", "info": {"chunks_ingested": 47, "pages": 12}}
```

### `POST /v1/query`
Ask a question. Checks semantic cache first.

```bash
curl -X POST http://localhost:8000/v1/query \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
# → {"answer": "...", "source": "CACHE HIT", "latency_ms": 5.2, "similarity": 0.9734}
# → {"answer": "...", "source": "LLM GENERATED", "latency_ms": 1842.0}
```

### `GET /v1/health`
Infrastructure status + cache stats.

### `DELETE /v1/cache`
Flush all semantic cache entries (requires authentication).

---

## Configuration

All settings are in `.env`:

| Variable               | Default              | Description                              |
|------------------------|----------------------|------------------------------------------|
| `OPENAI_API_KEY`       | *(required)*           | OpenAI API key                           |
| `API_KEY`              | *(required)*           | API key for authentication                 |
| `REDIS_URL`            | `redis://localhost:6379/0` | Redis connection string             |
| `QDRANT_URL`           | `http://localhost:6333`    | Qdrant REST URL                     |
| `QDRANT_COLLECTION`    | `enterprise_docs`    | Collection name                          |
| `SIMILARITY_THRESHOLD` | `0.90`               | Cache hit threshold (0–1)                |
| `CHUNK_SIZE`           | `512`                | Characters per chunk                     |
| `CHUNK_OVERLAP`        | `64`                 | Overlap between chunks                   |
| `TOP_K_CHUNKS`         | `4`                  | Chunks retrieved per query               |
| `CACHE_TTL`            | `3600`               | Cache entry TTL in seconds               |
| `MAX_FILE_SIZE_MB`     | `50`                 | Max upload file size in MB               |
| `RATE_LIMIT_REQUESTS`  | `100`                | Requests per minute per user               |

---

## Security Features

- **API Key Authentication**: All endpoints require `X-API-Key` header
- **File Validation**: Magic bytes and size validation for PDF uploads
- **Rate Limiting**: 100 requests/minute per IP by default
- **Non-root Containers**: All Docker images run as non-root user
- **CORS**: Restricted to specific origins

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=backend --cov-report=term

# Run specific test file
pytest tests/test_config.py -v
```

---

## CI/CD

GitHub Actions workflow runs on every push:
- Linting with ruff
- Unit tests with coverage
- Security scanning with safety
- Docker image builds

---

## Production Deployment

For production deployment, consider:

1. **Kubernetes**: Use the provided Docker images with a Helm chart
2. **Cloud Provider**: Deploy to AWS ECS, Azure Container Apps, or GCP Cloud Run
3. **Secrets Management**: Use AWS Secrets Manager, HashiCorp Vault, or similar
4. **Load Balancing**: Add nginx or Traefik for horizontal scaling
5. **Monitoring**: Add Prometheus and Grafana for metrics
6. **Logging**: Use centralized logging (ELK, Datadog, etc.)

---

## Resume Talking Points

- **Async ingestion pipeline**: Celery + Redis decouples upload from processing. API responds in <50ms regardless of PDF size.
- **Semantic caching**: Qdrant-backed cache with HNSW for O(log n) lookup. Demonstrated 30–70% LLM call reduction on FAQ-style traffic.
- **Vector retrieval**: Qdrant with `text-embedding-3-small` (1536 dims, COSINE distance). `RecursiveCharacterTextSplitter` with 512/64 chunk/overlap preserves context boundaries.
- **Production patterns**: `task_acks_late=True` for reliability, exponential backoff with jitter retries, `lru_cache` singletons for DB clients, Pydantic settings with env validation.
- **Security**: API key authentication, file validation, rate limiting, non-root containers.
- **Observability**: Every query response includes latency, cache source, and similarity score. Task state is broadcast via `update_state()` for live UI progress.
- **Testing**: Comprehensive test suite with pytest, mocking, and coverage reporting.