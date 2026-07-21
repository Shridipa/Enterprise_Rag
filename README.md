# CodeMind AI

CodeMind AI is an AI software engineering platform for repository understanding, architecture exploration, code review assistance, and developer productivity. It preserves the existing FastAPI, Redis, Celery, Qdrant, Docker, authentication, semantic caching, async ingestion, CI, tests, and observability pieces while extending the system for software-repository workflows.

## What CodeMind AI does

- Ingests folders and Git repositories for analysis
- Supports Python, JavaScript, TypeScript, Java, Go, Rust, and C++ repository analysis
- Extracts repository structure, modules, imports, dependencies, and symbol information
- Builds a lightweight architecture graph for navigation and exploration
- Exposes repository endpoints for indexing and architecture discovery
- Keeps the legacy document ingestion pipeline available for compatibility

## Architecture

```text
REPOSITORY INDEXING
  Folder or Git repo → FastAPI /v1/repositories/ingest → repository indexer → summaries + graph

ARCHITECTURE EXPLORATION
  Repo ID → FastAPI /v1/architecture/explore/{repo_id} → module/file graph + dependency edges

LEGACY DOCUMENT PATH
  PDF upload → FastAPI /v1/ingest → Celery worker → Qdrant upsert → /v1/query
```

## Project Structure

```text
enterprise-rag/
├── backend/
│   ├── main.py                    # FastAPI app with repository + legacy endpoints
│   ├── celery_app.py              # Celery configuration
│   ├── config.py                  # Pydantic settings (reads .env)
│   ├── code_intelligence/
│   │   └── repository_indexer.py  # Repository indexing and graph builder
│   ├── services/
│   │   └── repository_service.py  # In-memory repository service for indexing
│   ├── tasks/
│   │   └── ingest.py              # PDF → chunks → embeddings → Qdrant
│   └── rag/
│       ├── vectorstore.py         # Qdrant wrapper + collection init
│       ├── cache.py               # Semantic cache (Qdrant HNSW)
│       └── chain.py               # LangChain RetrievalQA chain
├── frontend/
│   └── app.py                     # Streamlit chat UI
├── tests/
│   ├── test_config.py
│   ├── test_main.py
│   ├── test_cache.py
│   ├── test_ingest.py
│   └── test_codemind.py           # Repository indexing and architecture tests
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.celery
├── Dockerfile.frontend
├── requirements.txt
└── .github/workflows/ci.yml
```

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker Desktop (for Redis and Qdrant)
- Optional: OpenAI API key for the legacy LLM-backed query path

### 2. Create a virtual environment

```bash
cd Enterprise_Rag
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start infrastructure

```bash
docker compose up -d redis qdrant
```

### 5. Run the backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 6. Run the frontend

```bash
cd frontend
streamlit run app.py
```

## API Reference

All endpoints are under `/v1/` and require an `X-API-Key` header.

### `POST /v1/repositories/ingest`
Index a repository or folder.

```bash
curl -X POST http://localhost:8000/v1/repositories/ingest \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"source_type": "folder", "path": "./my-repo", "name": "my-repo"}'
```

### `GET /v1/architecture/explore/{repo_id}`
Explore the repository graph for an indexed repository.

### `GET /v1/repositories/{repo_id}`
Retrieve repository details and analysis metadata.

### `POST /v1/ingest`
Upload a PDF for the legacy document ingestion pipeline.

### `POST /v1/query`
Ask a question against the legacy RAG path.

## Testing

```bash
pytest -q
```

## Notes

The project now supports repository-aware analysis for software engineering workflows while retaining the original document-processing capabilities for compatibility.

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